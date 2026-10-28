# Selene pool protocol 2: state machines

Companion to [selene.proto](../selene/proto/selene.proto), which is normative for fields. This
file defines who may do what, in which order, and what each party must prove before memory
is reused. The manager implements the manager side of each flow (section 12 lists what it serves);
the instance side (`flare-selene`, `selene::pool`) implements the other.

## Decisions

1. **One stream per instance.** `Hello` is a stateless unary call; everything
   else rides one bidirectional `Session` stream per instance process and scope.
   The stream's order is the only ordering the protocol needs.
2. **Incarnations.** `manager_incarnation` is a random nonzero 64-bit value picked
   at manager start. `session_incarnation` is assigned by the manager at
   `SessionOpened` from a counter, so a client cannot forge or reuse one.
   `instance_id` is **stable across restarts**: BLAKE2b-128 over the
   pool name, model identity, layout fingerprint and the sorted UUIDs of the
   instance's GPUs. Instances on one host use disjoint GPUs, so it is unique per
   host. A process is told apart by a random `process_nonce` (fresh per process,
   unchanged across a reconnect) and by the session incarnation. A new session
   for a known `instance_id` supersedes the old one: the newer incarnation wins,
   the older is fenced, and the old session's index entries are replaced
   atomically when the new session's snapshot installs.
3. **Encryption is declared, not negotiated.** The client selects its connection
   mode before connecting; see [address forms and trust](docs/network-operations.md#addresses-and-startup-warnings).
   For TCP clients, `Hello.encryption` must equal what the client connected with,
   else the pool is disabled for the process (one warning, local tiers only).
   Unix-socket clients trust filesystem permissions and accept either network mode.
   RDMA is not listed in `data_transports` when the manager is TLS.
4. **Same host is proven by the socket; levels.** `shared_memory` is
   returned only on a Unix-domain socket whose peer uid and gid match the
   manager's. A session without it uses `DATA_CHANNEL` only. It may be an
   export source when the manager serves a data channel: its windows are relayed
   through the manager (section 7.5). Instance-sourced candidates
   (neighbour GPU = level 2, neighbour DRAM = 3, neighbour NVMe = 4) are offered
   to every session, whether the copy goes through shared windows or through the
   manager's relay. Manager-store candidates are offered
   to every session: a same-host manager's DRAM is level 3 and its NVMe level 4; a
   remote manager's records (DRAM or NVMe) are level 5. The manager computes the
   level from the requester's perspective.
5. **Content keys.** A source pins ready content by its scoped prefix key and
   validates the chain shape. Index entries are hints: late evictions can remove
   a fresh hint and cause a miss, but cannot grant ownership or change bytes.
6. **COPIED is the requester's word,** and only after exact coverage (section 5).
   Everything else terminal is `FAILED`.
7. **Window ownership and reclaim.** A window passes manager, source, requester,
   manager. Each acknowledgment clears only the sending party's obligations
   (`ExportEnd`, `FenceAck`, `WindowReleased`, `Goodbye{quiesced}` are all
   per-party). Memory is reclaimed only after every party that could have
   touched it has drained (section 7).
8. **Claim and put ids are requester-chosen and strictly increasing** per
   (instance_id, process_nonce). The manager keeps a high-water mark per
   (instance, process) and refuses any id at or below it (`INVALID_CLAIM`, or
   `PutRefused{INVALID}`), which is what makes a replayed `Claim` harmless after
   its record has expired (section 5).
9. **Fence-class messages cross sessions.** `FenceAck`, `ExportEnd`,
   `WindowReleased` and `ClaimRelease` are resolved by (instance_id, process_nonce, id)
   and are accepted from any current session of the same process, because the
   accesses they speak for outlive the stream that started them. A session of a
   different process (a restarted or competing one) never attests for another
   process's accesses; those windows and pins wait for the old process's exit
   (section 7.4).
10. **Readiness follows recovery.** The manager serves lookups, claims and puts
    once its own NVMe scan completes. Sources reconnect and announce independently;
    a missing or partial source index can cause a miss, never grant a pin.
11. **Transfer cost** uses the requester's rolling bandwidth for each copy level,
    seeded until a restore completes. The manager does not report per-source estimates.
12. **Logical split versus physical page offset.** `Claim.begin_tokens`
    is the resident prefix end and need not be page-aligned. The source
    supplies pages from `floor(begin_tokens / page_tokens)` to
    `floor(key.consumed_tokens / page_tokens)` plus the row, the same flooring
    Flare's `Match` uses. Zero pages with the row is a legal row-only tail. A
    candidate's `begin_tokens` is likewise exact; its `skip_pages` is the physical
    offset into the first record.
13. **Claims run after admission, inside `fetch`** (see "Claims run after admission" in the client's state machine). Lookup is asynchronous
    while a request queues; claims start when the scheduler has reserved the
    ticket. A failed claim drains before reporting failure; the request
    recomputes, or uses a local copy of the same endpoint when available.
14. **GPU exports need retained staging first.** A
    GPU-tier export is accepted only after staging for the whole export is
    reserved: the manager reserves it from the index's size before sending the
    `ExportRequest` (`reserved_bytes`), and the source checks the actual chain
    against it before pinning. Otherwise the claim fails `NO_WINDOWS` at once.
    The GPU pin is therefore bounded by the D2H, never by the requester's pace.
    Host-tier sources (DRAM, NVMe) may stream with `window_credits`.
15. **Export slots are per instance.** An export touches every rank, so the
    per-GPU count (default 2) is also the per-instance count. The same default
    bounds claims in flight per requester. A quarantined export keeps its slot.
16. **Headroom:** computed
    counterfactually against Flare's real eviction rules (leaf eviction needs
    `pins == 0`; compaction needs no direct pin and exactly one remaining child;
    snapshot aliasing), subtracting the newly protected resources once, and with
    the reservation arithmetic shared with admission rather than copied.
17. **Put is included** because the eviction cascade ends in a write into the
    manager's store. Refusals keep the source local and release its selection.
    Retrying requires manager progress, reconnect or a short backoff; the model
    worker never waits for room.
18. **Entries die with their session, or are replaced atomically by its
    successor.** When a session ends with no successor the manager removes every
    entry it announced. When a new session opens for the same `instance_id`, the
    old session's entries stop being offered at once (its source session is
    stale: a claim naming it is `STALE_SOURCE`) and are replaced, in one step with
    respect to lookups, when the new session's snapshot installs. A restarted
    instance re-announces what its local tiers recovered under the same id.
19. **Lookup input is bucket-keyed.** The request carries the history's aligned
    boundary keys (capped); the manager returns candidates from the matching
    buckets and the client re-verifies each node with `history.key_at`.
20. **Data-channel framing is outside this contract.** Only the invariants in
    `DataChannelRef` and section 7 are fixed here.
21. **Persisted checksums.** A DRAM or NVMe export serves the checksums
    committed with the record, never hashes of the bytes just read; the manager
    validates every page and row. When the manager reads a whole record from NVMe
    into DRAM it checks the record against those checksums and fails the claim
    `COPY_ERROR` on a mismatch; a window read straight from NVMe into staging is
    served as it was read, with the persisted checksums, and the requester's check
    is the only one. The manager validates Put metadata against the accepted
    geometry, scope identity, parent, length and its own verification of the
    received payload before indexing. Only a GPU source, having no stored
    checksum, hashes what it wrote.
22. **Ticket handling.** The pool lookup ticket belongs to one request; it
    is consumed explicitly at admission, not by dropping the `Incoming` (which
    moves into `Active`). Candidates are validated against the request's complete
    held plan, not GPU residency alone (the client's lookup tickets).

## 1. Roles, identity and scope

An instance is a SOURCE (announces entries, serves exports), a REQUESTER (looks
up, claims, copies) and a WRITER (puts) on one session. The manager is the index,
the broker of claims and windows, the owner of the shared-memory regions and of
its own DRAM/NVMe store.

Scope is `(pool_name, model_identity, layout_fingerprint, geometry)`. The manager
supplies its configured name in Hello; OpenSession supplies the model and
geometry. The manager keeps one index and one set of claims per `scope_id`;
no message can name another. A claim whose source belongs to another scope is
`UNKNOWN_SOURCE`, which is what an unknown name answers too. Rank geometry is part
of the scope, so TP widths never share records.

Every message after `open` carries `Header{manager_incarnation,
session_incarnation}`. Receivers drop messages with a wrong manager incarnation.
The manager closes the stream (`SessionClosing{STALE_SESSION}`) on a wrong session
incarnation, except for the fence-class messages of decision 9.

## 2. Session lifecycle

```
client                                         manager
  | Hello{protocol} ------------->|
  |<---- HelloReply{status, incarnation M, encryption, readiness, limits, shm?}
  |  check: status ok, encryption == connected_with
  | Session stream open                           |
  | OpenSession{M, instance_id, scope, slots} --->|
  |<---- SessionOpened{S, scope_id, slots}
  |  check: scope_id equal
  | ResyncBegin / Chunk* / End  (may be empty) -->|   applies hints live
  |<---- ResyncAck
  | NoticeBatch ... ----------------------------->|
  | Lookup / Claim / Put at any time after open (when READY)
```

- `STALE_MANAGER`: the manager restarted between Hello and open. Redo Hello.
- A second `Session` for the same `instance_id` supersedes the first: the newer
  incarnation wins. The manager sends `SessionClosing{CLOSING_SUPERSEDED}` on the
  old stream and treats it as lost (section 8.5). Two cases, told apart by
  `process_nonce`: the same nonce is a reconnect after a blip (the old stream's
  fence-class messages stay valid on the new session); a different nonce is another
  process (a restart, or two processes started with the same GPUs): the older
  process is fenced, nothing it attested is trusted from the new process, its
  windows stay quarantined until its exit is confirmed, and its pins are its own
  to release when it exits.
- `Goodbye{quiesced=true}` is the only close that lets the manager stop waiting
  for THIS process's acknowledgments: it clears only this party's obligations. A
  window or pin that the opposite party of a claim may still touch is reclaimed
  only after that party has drained too (section 7.2). Any other end of the
  stream is a disconnect. A source sends `ExportEnd` for every export before a
  quiesced `Goodbye`; a quarantined export keeps its pin and makes `quiesced`
  false.
- Reconnect uses exponential backoff with a cap; no more than one Hello in
  flight. While disconnected the client is in `Local` state (the client's state).

## 3. Index hints and resync

Each session announces its current GPU, DRAM and NVMe holdings. Notices upsert,
spill or evict hints on the ordered control stream. Claims still validate and pin
the actual source before copying; the index grants no ownership.

A snapshot consists of Begin, bounded chunks and End. Begin clears this source's
hints; each chunk is applied live. End is acknowledged. Lookups may see a partial
snapshot and miss during this short interval. A second Begin clears a partial
predecessor too. Session supersession and disconnection remove that session's hints.

The client takes holdings and clears its unsent queue under one lock. Changes
after that cut follow the last snapshot chunk. Queue overflow or lost GPU eviction
notices request another snapshot without blocking the worker. Entry, chunk and
batch bounds remain enforced. There are no notice sequences, delta buffers,
watermarks or snapshot timeout.

## 4. Lookup

Request: the history's aligned boundaries, `before_tokens`, `resident_tokens`.
The manager answers from its index only; it never contacts a source.

Manager algorithm for each candidate source entry (endpoint `E` with a row):

1. Skip the requester's own instance and sources the requester cannot reach
   (decision 4), including sources with no export slots. Busy sources remain eligible.
2. Compute the level from the tier and the relation to the requester.
3. `begin` = `resident_tokens` (exact, not page-aligned); the physical start
   is `floor(begin / page_tokens)`. Walk parents from `E` to the node containing
   that page and require a contiguous chain. A hole gives no candidate.
4. Drop it if it supplies no page and no row beyond `begin` (zero pages with the
   row is kept: a row-only tail).
5. Order by longest match, then fixed copy level and stable source identity; return at most
   `max_candidates`, each with at most `max_chain_nodes` nodes.

Answers: `NOT_READY` (with `retry_after_ms`), `OVERLOADED` (the bound on
concurrent lookups or the work queue, answered at once, never queued), `INVALID`.
The requester treats every non-OK answer, a timeout, and a lost session as "no
pool candidates for this request" and admits with its local match. Each request
owns its lookup ticket.

A candidate is a hint. After the reply the client verifies each node against its
history; any mismatch drops the candidate. Lookup reserves no export slot: another
claim can fill the source between lookup and claim, causing `SOURCE_LIMIT`. The
client retries that refusal within the existing size-scaled restore deadline;
a temporarily busy source does not cause a recompute. A quarantined export keeps
its slot until drained or its process exits. Pending host writes are not announced
as ready holdings; a GPU holding can be offered while its separate host copy is
still in flight.

## 5. The claim state machine

The manager holds one record per claim, keyed (instance_id, process_nonce,
claim_id), mutated
only by the manager's single claim task (so "one authoritative transition" is a
compare-and-set on that task, not a lock). Ownership and replay tracking are separate:

- **Fence records** live until every party that could have touched the claim's
  memory has drained or its process has exited, however long quarantine lasts.
- **Replay tracking** uses a per-(instance, process) high-water mark. Reusing a
  claim id is refused `INVALID_CLAIM`, without starting another lifecycle.
  Fence acknowledgments still resolve through records retained for their memory.
  A process entry is dropped after exit and after its final fence record is gone.

```
                      Claim
 (none) ---------------------------> REQUESTED
   |  refused at once: NOT_READY, CLAIM_LIMIT, SOURCE_LIMIT, UNKNOWN_SOURCE,
   |  STALE_SOURCE, INVALID_CLAIM  ==> FAILED  (ClaimTerminal, fence_required=false)

 REQUESTED --pin held, grant sent--> GRANTED
 GRANTED --exact CopyResult COPY_OK--> COPIED
 any non-terminal --> FAILED  refusal, loss, cancellation, copy error,
                              deadline or no-progress timeout (reason preserved)
```

Terminal means `COPIED` or `FAILED`. The first transition into a
terminal state wins and is final; later events for the claim only cause
idempotent replies and fence bookkeeping. On entering a terminal state the
manager, in this order:

1. records the state and reason;
2. sends `ClaimTerminal` to the requester once;
3. sends `ExportCancel` to the source if its export has not ended;
4. stops the data-channel transfer if any;
5. starts fence tracking (section 7).

Only the requester's `CopyResult{COPY_OK}` makes `COPIED`. The manager accepts
it only on **exact coverage** of the granted records: the windows issued, filled
and released together cover every page of every record in `records` and the
endpoint row if granted, once each, with no hole and no duplicate (for streamed
host-tier claims, after the last window has been issued, not merely after all
windows issued so far were released); on the data channel, the transfer ended
with exactly `total_pages` pages and the row. Anything else becomes
`FAILED{COPY_ERROR}`. The requester performs the same coverage check on its side
before sending `COPY_OK`.

Manager-store sources have no export: the manager takes its own reference
on the record inside `Claim` handling before granting it, so
refusals (record evicted) are immediate and identical on the wire.

### Message flow, normal GPU export (shared windows, retained)

```
requester          manager                    source (model worker via poll)
 Claim ---------->|  checks; reserve retained staging for the whole export
                  |  (none available: FAILED NO_WINDOWS now); REQUESTED
                  | ExportRequest{reserved_bytes}->| poll -> worker:
                  |                               |   revalidate chain, size <= reserved, headroom, pin_checkpoint
                  |<------------------ PinAck PINNED{records}
                  | pin held; allocate windows
 <--ClaimGranted--|  GRANTED (hold is now in force)
                  | ExportFill{windows} --------->| D2H into windows (lowest-priority stream)
                  |<------------ WindowFilled{i, xxh3}  (after the D2H drained)
 <--WindowReady---|  GRANTED
 H2D from window i
 WindowReleased-->|  (after H2D drained)
                  |<------------ ExportEnd{COMPLETE, DRAINED}   (pin released after last fill)
 CopyResult OK -->|  COPIED
 <-ClaimTerminal--|  fence_required
 FenceAck DRAINED>|  windows freed once source also ended
```

### Race: cancellation crosses the pin acknowledgment

The requester gives up (deadline of its own) while the export is in flight.

```
requester     manager                     source
 Claim ------>| REQUESTED, ExportRequest ->| queued in poll
 ClaimRelease>| FAILED{CANCELLED}
 <ClaimTerminal| ExportCancel ------------>|
              |                             | case A: request not started -> drop,
              |                             |   PinAck REFUSED{CANCELLED}, no ExportEnd
              |<--- PinAck PINNED (crossed) | case B: pin done, ack sent before the cancel
              | late pin: terminal claim    |   was processed -> manager re-sends
              | ExportCancel (idempotent) ->|   ExportCancel; source unpins after in-flight
              |                             |   copies drain, ExportEnd{RELEASED}
              |<------------ ExportEnd ----|
              | export slot freed           | case C: cancel processed between pin and ack
              |                             |   -> unpin, PinAck REFUSED{CANCELLED}
```

Source rule: exactly one `PinAck` per `ExportRequest`; if it is `PINNED`, exactly
one `ExportEnd` later. The manager frees the source's export slot on `PinAck
REFUSED` or on `ExportEnd`, never on its own terminal transition (the source may
still hold the pin). A `PINNED` ack for a terminal claim never produces a grant.

### Race: late grant

`ClaimGranted` arrives at a requester that has already released the claim, or
that does not know it (restarted, or the claim was cancelled and forgotten).

- The requester never touches the windows of a claim it does not hold. It
  ignores `ClaimGranted` and `WindowReady` for such a claim.
- It answers the following `ClaimTerminal` with `FenceAck{DRAINED}` at once (it
  has no accesses to drain). For a claim id it has never heard of it does the
  same. If it already started a copy before releasing, it drains first.
- A requester that restarted is a new instance; the manager has already failed
  the old instance's claims with `REQUESTER_LOST` (section 8.3).

### Idempotency

| Message | Duplicate or late arrival |
|---|---|
| `Claim` with a reused id | refused `INVALID_CLAIM` |
| `ClaimRelease` | no effect once terminal; unknown id answered `ClaimTerminal{FAILED, UNKNOWN_CLAIM}` |
| `WindowReleased` | each (claim, index) returns its credit once |
| `CopyResult` | first one counts; others ignored |
| `FenceAck`, `ExportEnd` | first per (claim, party) counts |
| `PinAck` | first per export counts; a late `PINNED` is released |
| `ExportCancel` | idempotent; unknown export answered `ExportEnd{RELEASED, DRAINED}` |
| `NoticeBatch` | applied in stream order |
| `PutCancel` | idempotent |

Nothing promises exactly-once delivery across a process death; every state above
is recoverable by timeout.

## 6. Export on the source

An `ExportRequest` reaches the instance's pool client, which hands it to its
owner:

- **GPU tier:** delivered through `poll` to the model worker as
  `StoreEvent::ExportRequested`. On the worker, atomically (one scheduler step):
  1. resolve the ready checkpoint by its scoped key; otherwise refuse `NOT_FOUND`;
  2. verify the chain from the physical start (`floor(begin_tokens / page_tokens)`)
     is contiguous and has the row if wanted (`CHAIN_GAP`), and that its size fits
     `reserved_bytes` (`NO_WINDOWS`);
  3. run the headroom check (the client's headroom check); refuse `NO_HEADROOM`;
  4. check this instance's own export slot count; refuse `SOURCE_BUSY` (the
     manager's own pre-check answers `SOURCE_LIMIT`);
  5. `pin_checkpoint(id)`.
  The refusal is returned at once. It never waits for pages to appear.
- **Host tiers (DRAM, NVMe):** selene takes a reference on the record chain inside
  the pool client thread. No model worker involvement.

After `PINNED` the source waits for `ExportFill`. A GPU export receives all its
windows in one `ExportFill` (retained staging was reserved before the pin), so the
pin lasts only for the D2H. It copies each window (GPU: D2H on the lowest-priority
copy stream, behind the instance's own restores and offloads; host: memcpy of the
record bytes, with the persisted checksums, decision 21), and sends `WindowFilled`
only after the copy has drained. It releases the pin after its last fill, on
`ExportCancel`, on session loss, or on a failed fill, and reports
`ExportEnd{fence = DRAINED}` once the pin is released. A DMA already submitted is
never revoked: cancellation lets it drain first.

If the source cannot prove a copy drained it sends `ExportEnd{fence = QUARANTINED}`
and **keeps the pin** (a GPU checkpoint stays pinned until the process exits;
unpinning memory a DMA may still read is never allowed). `QUARANTINED` does not
release the export slot.

If the source sees its session lost it releases pins itself after draining and
reports `ExportEnd` on the next session.

A source without shared memory (a TCP session) is asked the same way, and the request also
carries `channel`: a data-channel reference for this export. After `PINNED` the source connects to
it and writes the windows there, in the direction of a put (section 7.5). The manager sends it no
`ExportFill` and expects no `WindowFilled`: each window is granted on the channel, the source copies
it (GPU: D2H into a slot of its own registered staging; host: the persisted bytes) and writes it
with the checksums it states, and `ExportEnd` follows as for any source. The pin is released once
every copy has drained, exactly as before; the manager's windows are written only by its own
handler, so they are not the source's to fence.

## 7. Copy, ownership and fencing

### 7.1 Logical invalidation and physical fencing are different

- **Logical invalidation** stops new work and rejects stale messages: a new
  `session_incarnation`, a terminal claim state, a cancelled export. It says
  nothing about copies already running.
- **Physical fencing** is the only thing that allows memory to be reused. Memory
  (windows, staging, GPU pins, registered regions) returns to service only after
  every access to it has provably stopped.

Fence parties for a claim: the requester (its reads of windows or its receive
buffers), the source (its writes into windows, its GPU pin), and the manager (its
data handler and registered regions). The manager reuses memory only when every party
that could have touched it is `DRAINED`; each acknowledgment (`ExportEnd`,
`FenceAck`, `WindowReleased`, `Goodbye{quiesced}`) clears only the sending party's
obligations, never another party's. A source finishing does not free a window the
requester is still reading, and a requester finishing does not free one the source
is still writing.

### 7.2 Window ownership

```
FREE --ExportFill--> SOURCE_WRITE --WindowFilled(OK)--> REQUESTER_READ --WindowReleased--> FREE
        (manager)        (source)        (source drained)     (requester)     (requester drained)
```

- A window is `SOURCE_WRITE` until the source says it drained its write. It is
  `REQUESTER_READ` until the requester says it drained its read.
- `WindowFilled` means the D2H/memcpy completion was observed, so the requester
  may read; `WindowReleased` means the H2D/memcpy completion was observed.
- A manager-store DRAM source serves ARENA windows aliasing immutable bytes: no
  `SOURCE_WRITE` phase. NVMe sources are read into STAGING by the manager first.
- At terminal, a window in `SOURCE_WRITE` needs the source's fence (`ExportEnd`
  with `DRAINED`); one in `REQUESTER_READ` needs the requester's `FenceAck`; a
  `FREE` window needs nothing. Windows are reclaimed individually.
- A window is reclaimed when the last party that held it has drained. The cases:

| State when the claim ends | Reclaimed after |
|---|---|
| `SOURCE_WRITE` | source `ExportEnd{DRAINED}` (requester never read it) |
| `REQUESTER_READ` | requester `FenceAck{DRAINED}` (source drained earlier, by `WindowFilled`) |
| `SOURCE_WRITE`, source `QUARANTINED` | source process exit, or manager exit |
| `REQUESTER_READ`, requester `QUARANTINED` | requester process exit, or manager exit |
| either, with `Goodbye{quiesced}` from one party | still the opposite party's acknowledgment |

### 7.3 What "drained" means per medium

| Medium | Party | Drained when | Not drained means |
|---|---|---|---|
| CUDA (D2H into a window, H2D out of it, GPU pin) | requester or source | the copy stream or event covering every submitted copy has completed on every rank, including a submission that failed part-way (Flare's `PageCopier` contract: one aggregated completion after all ranks drain) | quarantine |
| Disk (NVMe read into staging, writes in a put) | manager | every submitted I/O completion has been reaped | quarantine |
| TCP data channel | both ends | the data handler task has exited and the socket is shut down | quarantine |
| RDMA data channel | both ends | the queue pair is moved to the error state and destroyed successfully (`shared::rdma::Qp::close`) | quarantine |
| Local process that exited | requester or source | the manager confirms the process is gone (pidfd) | quarantine until exit |

### 7.4 Quarantine

If drain cannot be established (a peer is unresponsive, or completion state is
unknown), the party reports `QUARANTINED` (or the manager infers it). The memory
is not reused until the owning process is confirmed exited, or for the lifetime
of the manager process if exit cannot be observed (remote sessions). A
disconnected local process that is still alive keeps its windows quarantined
(`local.rs`).

- The manager counts quarantined bytes in `selene_pool_quarantined_bytes` and
  logs each event once. Claims that cannot get windows fail `NO_WINDOWS`
  immediately. There is no waiting for quarantined memory.
- On the client, a registration handle whose drain failed keeps its region alive
  and mapped (`HostRegistrar` contract); a GPU pin whose copy cannot be drained
  is never unpinned (no `ExportEnded` event is produced for it); the worker stops
  rather than reuse it (existing `PrefixStore::poll` failure semantics).

### 7.5 Data channel (other hosts)

`ClaimGranted.channel` replaces windows. The requester connects, presents the
single-use token, and receives the records in order with per-page and row xxh3
checksums. It verifies every checksum before using the bytes. The manager ends the
transfer on any terminal state. Fencing: the manager's handler exits and its socket
shuts down (RDMA: queue pair quiesced, work requests flushed, regions
deregistered after that); the requester closes its end the same way and only then
sends `FenceAck`. An unresponsive peer means the manager's registered regions for
that transfer stay quarantined until it exits.

**Framing (this implementation).** One connection per transfer, after TLS when the manager
has a certificate. The instance sends the 32-byte token, then `HELLO` naming the carrier
(stream, or RDMA with its queue-pair description). Frames, little-endian, are defined in
`flare_selene::shared::frame` (the header of `selene/src/shared/frame.rs`): `WINDOW` (index, record,
first page, page count, row flag, length, row checksum, slot, one checksum per page) followed on the
stream carrier by `length` payload bytes, then `END`; for a put `GRANT` from the manager and
`FILLED` (with the instance's checksums) from the instance, payload following on the stream
carrier. On the RDMA carrier the payload never travels on the connection: the manager writes a
claim's window into a private slot the instance lent (`CREDIT`) and then names it in `WINDOW`.
Puts and rank legs use the stream carrier. Windows arrive in index
order and are checked by the receiver's coverage exactly as shared windows are, so a transfer that
ends short or over is a `COPY_ERROR`; `COPY_OK` waits for `END`.

The manager's windows are read or written only by its own handler, so its exit is the manager's
fence for them (the requester's `FenceAck` is accepted and idempotent, but no window waits for it).
A stopped handler retains its windows until its exit is observed. These outstanding windows are
counted in `selene_pool_quarantined_bytes`, including handlers still draining normally.

**Relay.** A claim whose source has no shared memory, or whose requester asks for
`DATA_CHANNEL`, goes through the manager's staging in both cases; the two legs are independent
transfers on the data channel (or shared windows for a local end):

- *Source leg.* The manager opens a transfer for the export and names it in `ExportRequest.channel`.
  The source connects after `PINNED`; the manager `GRANT`s each window it issues (retained staging
  for a GPU source, reserved before the pin, as for any export; credits otherwise) and the source
  answers with `FILLED` and the payload. Checksums are the source's: persisted ones for a host record,
  computed over the bytes it wrote for a GPU checkpoint. The manager passes a window on only after
  its `FILLED` and payload arrived whole, and never trusts a count: a window that does not match the
  pinned chain's plan fails the claim.
- *Requester leg.* A filled window goes to the requester as `WindowReady` (shared windows) or on its
  own channel as `WINDOW` (`ClaimGranted.channel`). On the channel the window is free once written;
  otherwise when the requester releases it. The manager, not the source, ends the requester's
  transfer (`END`) once every window has gone.
- *Coverage and verdict.* Exact coverage is checked by the requester as for any claim and by the
  manager at `COPY_OK`. A source that ends its connection before every window arrived fails the claim
  (`SOURCE_LOST`); a requester that does fails it (`COPY_ERROR`) and the source is sent `ExportCancel`.
- *Fencing.* A window being filled is the source leg handler's to write, so the handler's exit is
  the fence for it, not the source's `ExportEnd`; a window being read by a remote requester is that
  leg's handler's. Each leg is stopped when the claim ends. A handler not seen to exit within five
  seconds of the stop quarantines the windows it may touch, counted in
  `selene_pool_quarantined_bytes`, until its exit is seen. A source whose session ends is fenced
  by that: its claims fail, its export is released, and nothing waits for an `ExportEnd` it cannot
  send.
- *Levels.* A relayed copy keeps the source's tier level (neighbour GPU, host DRAM, host NVMe); the
  requester observes completed restores for its cost estimate.

### 7.6 Flare's view

Flare sees `Fetched` only after every destination write has drained: selene
aggregates its H2D submissions (one per part) and reports a single result when
all have completed, including on failure. Destination pages are discarded by the
scheduler's existing failed-ticket path only after that.

## 8. Failure and restart

### 8.1 Manager death (seen from an instance)

Detected by stream error, EOF, a `SessionClosing`, or a different
`manager_incarnation` in a later Hello. The instance:

1. enters `Local` state: no new claims, exports, puts or lookups are issued;
   lookups already in flight complete as "no candidates";
2. fails every claim it has outstanding locally (all-or-nothing plans fail);
3. drains its own CUDA and transport accesses (it cannot ask the manager; the
   `PageCopier` completions tell it);
4. then releases its export pins (`unpin_checkpoint`) and drops host-tier
   references, quarantining whatever it could not drain. It keeps the shared
   regions mapped and registered: they are mapped and registered once, at
   startup, and for the life of the process (section 8.6);
5. continues on its local tiers. Evictions that targeted the manager's NVMe pass
   the record one tier further down (design: unavailable destination);
6. reconnects with backoff (Hello, open, resync). The regions are the same
   files at the same offsets when the restarted manager reports the same
   `SharedMemory.regions_generation`; a different generation disables the pool
   until the process restarts (section 8.6). Opening the new session is the
   instance's statement that step 3 is done.

### 8.2 Manager restart

All sessions, claims, exports, windows and in-flight puts are gone. The new
process picks a new incarnation and becomes `READY` after scanning its NVMe
records.

- A `ClaimRelease`, `FenceAck` or `ExportEnd` that mentions a vanished claim or
  export is answered (`UNKNOWN_CLAIM`) or ignored. Old windows were never
  reusable by the new manager: it reuses the same region files and holds them
  until each process that had a session has reconnected or exited
  (section 8.6).
- Records the dead manager held in DRAM are lost; NVMe records are recovered.
  A writer dropped its source copy only at `Stored`, so there is nothing to
  resend; the record is simply gone and will be recomputed on demand.
- Readiness does not wait for previous instances; source hints return as they resync.

### 8.3 Requester restart

The restarted process has the same `instance_id` (decision 2) and a new
`process_nonce`. If the old stream is still open when the new session opens, the
newer wins and the older is fenced (section 2); otherwise the manager first sees
the old stream end. Either way:

1. fails each claim of that instance `REQUESTER_LOST`, sends `ExportCancel` to
   the sources;
2. windows in `REQUESTER_READ` state, and the instance's put windows, stay
   quarantined until its process exit is confirmed (SIGKILL: immediate; a hung
   process stays quarantined);
3. stops offering the old session's entries immediately; they are replaced
   by the new session's live snapshot chunks, or remain absent if no
   successor appears.

The new process has no pool state: it opens a session under the same id, resyncs
what it recovered, and its first lookup asks the manager directly. Its claim and
put counters start afresh (new nonce), and it attests nothing for the old process.
There is no catalog mirror to rebuild and so no catalog-after-restart race. A request that arrived
before the session was open is admitted without pool candidates.

### 8.4 Source death

On stream end or process exit the manager:

1. removes the source's entries immediately (decision 18) so no new lookup offers
   them;
2. fails every claim served by it `SOURCE_LOST`;
3. quarantines the windows in `SOURCE_WRITE` until the source process is
   confirmed exited. Windows already in `REQUESTER_READ` hold valid bytes the
   source no longer touches and follow the requester's fence;
4. never trusts partially filled windows: the requester's all-or-nothing rule
   discards the restore.

If the stream dropped but the process lives (a blip), the process reconnects;
its `ExportEnd{fence}` for old exports arrives on the new session
(decision 9) and releases the quarantined windows. No hold survives the source
dying: "a hold protects against eviction only".

### 8.5 Session loss summary

| Event | Claims as requester | Claims as source | Entries | Windows |
|---|---|---|---|---|
| stream ends, process gone | `REQUESTER_LOST` | `SOURCE_LOST` | removed | reclaimed (exit confirmed) |
| stream ends, process alive | `REQUESTER_LOST` | `SOURCE_LOST` | removed | quarantined until fence acks arrive on a new session or the process exits |
| superseded by a reconnect (same process) | `REQUESTER_LOST` | `SOURCE_LOST` | hidden, replaced at the new snapshot | as above; the old stream's fence acks are accepted on the new session |
| superseded by another process (same `instance_id`) | `REQUESTER_LOST` | `SOURCE_LOST` | hidden, replaced at the new snapshot | quarantined until the old process's exit is confirmed; the new process attests nothing for it |
| manager restart | all dead | all dead | gone | the same region files, held until the previous run's processes have reconnected (drained) or exited (8.6) |

### 8.6 Persistent shared regions

An instance may register host memory with CUDA only at startup (registration
consumes device memory, and serving allocates none). So the manager's arena and
staging are not per-incarnation: they are fixed-size files, `arena` and
`staging`, in a directory named by the pool name and a hash of the absolute Unix
socket path. A per-name lock beside the region directories permits only one
manager per pool name on a host; its holder records its pid and socket. Changing
the listener path selects separate regions after the previous manager stops. An
instance maps and registers them once, when it first connects, for the life of
its process.

**Reuse.** At start the manager takes the instance lock, then reuses the existing
files when both are regular files of its user with exactly the configured sizes,
and a readable generation. Their paths, inodes and offsets are then unchanged,
and `SharedMemory.regions_generation` (Hello) is unchanged. Windows name a region
(`ARENA` or `STAGING`) and an offset, both stable across the restart. Otherwise
(first start, `--memory` or `--staging` changed, a file truncated or replaced) it
recreates both files, chooses a new random nonzero generation and logs that
running instances must restart to use the pool. The lock decides ownership, so a
live manager's files are never reclaimed and another instance's directory is never
read. Startup refuses region files owned by another user or symbolic links, and
removes only its unfinished `generation.tmp` and `holders.tmp` metadata writes.
Other files are left alone. Shutdown retains the regions for reconnecting clients.

**Client.** A client uses the pool only with the regions it registered:

- same generation, paths and sizes after a reconnect: it continues with its
  existing mappings and registrations;
- a different generation, path or size: its mappings are of files that no longer
  back the pool. It disables the pool for the process (local tiers only, one
  warning) and never registers late. Restarting the instance picks up the new
  regions;
- manager unreachable at the first connect: it waits a bounded time, then runs
  local-only for the life of the process and never attaches later.

**Why reuse is safe.** Reusing the files gives up what new files used to
guarantee: that no access under the dead manager can alias a window of the new
one. Physical fencing (7.1) is restored by the following rule instead. The
manager records the pid and start time of every process with a local session, in
a file that survives a crash. When it starts on reused regions, each recorded
process that is still the same live process is a previous-run holder, and the
manager takes the whole of both regions out of service (no window, put extent or
record can be allocated in them) until every such holder has either

1. exited, as its pidfd reports (the 7.4 quarantine rule: memory returns when
   the owning process is confirmed gone), or
2. opened a new session. A client opens one only after step 3 of 8.1: it has
   failed its claims and every copy it submitted under the old incarnation has
   completed on every rank. Old-epoch windows are not referred to by the new
   incarnation (stale messages are rejected by `manager_incarnation`), so after
   that nothing of the old epoch can write.

While the hold lasts a claim is refused at once with `REGIONS_HELD`, not `NO_WINDOWS`. It ends as soon as the last recorded holder has reconnected or exited, not on a timer. A requester that gets it claims again after a short wait (250 ms), within the restore's own deadline, and recomputes only when that passes.

A holder that is alive but never reconnects, or whose exit cannot be observed,
keeps the regions held: the same outcome as a quarantined window in 7.4, at region
size, and visible as `REGIONS_HELD`. A process that was not recorded cannot have a
window of the old epoch, so it needs no wait. Recreated files need no hold: they
are different files that no old mapping can reach. When the hold ends the
allocators are fresh, so every window of the old epoch is free by construction.

## 9. Single-source restores

The client chooses the longest matching prefix, then the closest copy level,
then a stable source identifier. One source supplies every missing page and the
endpoint row. Its chain must cover the resident prefix through the endpoint;
local and pool prefixes are never stitched together.

The page count is `floor(endpoint/page_tokens) - floor(resident/page_tokens)`.
A zero-page tail with a row is legal. Destination pages start at zero. Every key
in the chain must match the request history and scope.

After admission, the client obtains one claim and validates page and row checksums
before completing the restore. A failure drains all submitted copies and reports
one failed `Fetched`. A local copy of the same endpoint can be tried after the
pool copy drains; other pool sources are not replanned. Nothing partial is
published. Waiting restores use the bounded claim-slot queue.

## 10. Limits

| Limit | Value | Enforced by | On exceeding |
|---|---|---|---|
| Exports at once per instance (per-GPU count) | 2 (`export_slots`) | manager (pre-check) and source (re-check) | `SOURCE_LIMIT` / `SOURCE_BUSY`, immediate |
| Claims in flight per requester | 2 | manager | `CLAIM_LIMIT`, immediate |
| Headroom at the source | per check, section 6 | source's model worker | `NO_HEADROOM`, immediate, never waits |
| Retained staging for a GPU export | whole export reserved before the pin | manager reserves, source checks | `NO_WINDOWS`, immediate |
| Pin answer time | `pin_deadline_ms` 250 | manager timer | `SOURCE_TIMEOUT`, late pin released |
| Window credits (streamed claims and puts) | 2 | manager | no more windows until a release |
| Claim deadline | setup clamped to 30 s + bytes at 8 MiB/s | manager | `FAILED{DEADLINE}` |
| No progress | 3 s | manager | `FAILED{NO_PROGRESS}` |
| Concurrent lookups | 8 | manager | `OVERLOADED` |
| Candidates, nodes, boundaries | 8, 4096, 8192 | both | trimmed (client) or `INVALID` (manager) |

- Exports run on the lowest-priority copy stream.
- A refusal is a normal outcome. The requester moves to its next candidate for
  the same endpoint or recomputes; it does not retry the same source for 1 s.
- None of these scale with max-concurrency.
- The limits are defaults carried in `Hello`; configuration can override them
  because they come from hardware and the scheduler, not flags.

## 11. Put (spill into the manager's store)

```
writer                          manager
 PutBegin{put_id, key, ...} --->| validate, budget
 <------ PutAccept{first_page, parent, record_bytes, channel?}
 <------ PutGrant{window i}     |  at most window_credits unfilled
 write window, drain
 PutFilled{i, xxh3} ----------->|  verified by the manager
 ...
 PutCommit{metadata} ---------->|
 <------ Stored{bytes}          |  committed, indexed: the writer may now drop its copy
 or  <-- PutRefused{reason}     |  PRESENT is success; refusals keep the source local
```

- `PutCancel` is legal only before `PutCommit`. Unfilled windows stay leased to
  the writer until `FenceAck{put_id}` or process exit.
- "Already held means committed": `Stored` and `PRESENT` are the only answers
  that let the writer remove its local copy after a handover. All refusals,
  synchronous submission errors and repeated failures retain the source.
- **Pressure policy.** `BUDGET` may mean temporary staging or arena pressure,
  not a full manager. Every refusal releases selection and retries only after
  a manager or put event, a reconnect or a short backoff. Queue bounds stop
  selection before pinning. Unavailable or busy managers leave local LRU eviction
  as the end of the chain; only the manager's own LRU drops its committed data.
  Manager loss releases waiting selections through the failed-move path. Moves
  already started keep their move and export pins until the drained terminal
  event. The model worker never waits for these operations.
- **Validation.** Before indexing, the manager checks the
  `PutCommit` metadata against the accepted geometry, the scope identity, the
  accepted parent and first page, the record length, and the checksums it
  verified over the payload it received; a mismatch is `PutRefused{INVALID}` and
  nothing is indexed. On later reads the stored checksums are served unchanged.

## 12. This implementation (selene-manager)

The manager embeds `store::Store` as a library through its command interface.
The store owns record admission, storage, durability and recovery; the pool actor
owns sessions, hints, claims and fences. There is no second store wire service.

- **Transports.** See [network operations](docs/network-operations.md) for listeners,
  address forms, TLS, RDMA and startup warnings. `shared_memory` is returned only
  to a Unix-socket peer with the manager's uid and gid.
- **The data channel.** A session without shared memory can open, resynchronise, look up, see
  level-5 candidates (the manager's own records), claim them and put into the manager's store,
  with the bytes on the data channel (`--listen-data`; section 7.5). An instance may be the source
  too: a session without shared memory serves exports (`export_slots` as asked, up to the limit)
  and the manager relays its windows; two instances on other hosts, or one on each side, copy
  through the manager's staging. On a manager with no network listener `DATA_CHANNEL` is refused
  (`INVALID_CLAIM`;
  `PutRefused{INVALID}` saying so). A remote instance has no pid, so the store gives it a
  connection of its own (`RemoteHello`); it is told apart by its session, not watched, and no
  window is ever leased to it.
- **Lookup timing.** Lookups are answered on the manager's 10 ms clock, not inline, so a
  burst beyond `max_concurrent_lookups` is refused `LOOKUP_OVERLOADED` at once. Candidates
  come from the index only; the manager never contacts a source to answer one.
- **Manager-store entries.** The manager's own store appears in each scope's index as the
  manager's entries (tier DRAM or NVMe; records recovered after a restart start on NVMe).
- **Windows.** An instance-to-instance claim reserves one staging block for the whole export
  (its size rounds up to the allocator's size class), cut into windows by the pinned records.
  A host-tier source whose export does not fit streams with `window_credits`. Windows for the
  manager's own records are the store's: arena leases or staging reads, at most
  `window_credits` unreleased at a time. A store-sourced claim whose requester reports an exact
  `COPY_OK` before the store reports its own reads drained completes when the store does.
- **Put.** Records go through the store's put path (parent-first chain durability, budget,
  deduplication). The manager checks the commit metadata against the accepted put before the
  store sees it. A refused or cancelled put leaves its unfilled windows leased to the writer
  until its `FenceAck{put_id}`, `Goodbye{quiesced}` or exit.
- **Processes.** A process is identified by its Unix-socket credentials (pidfd). A stream that
  ends while the process lives keeps the windows quarantined until the same process reports on a
  new session; a process confirmed gone releases everything only it could touch.
- **Persisted state.** Committed NVMe records and shared-region files, generation
  and holders survive a restart. The store rebuilds its index from records; instances
  resynchronise their hints. Live claims and window grants are not recovered.

## 13. Multi-process instances: rank agents

An instance whose tensor-parallel ranks are separate processes is one instance of the pool.
Rank 0 alone opens the `Session` (`OpenSession.rank_secret` set, `scope.geometry.ranks` one
entry per rank, at least two); it announces, looks up, claims, puts and answers exports.
Every other rank runs an agent that attaches with the `Rank` stream and moves only its own
slice of each window. Records at rest are the ordinary composite records (one checksum per
page, one for the row; the scope and its identity are those of a single-process instance of
the same shape, so the two share records). Slices exist only in flight.

**Slices.** A window holds whole page records then optionally the row record; within a record
the ranks follow each other, each rank's planes padded to a block. Rank r's slice of page i of a
window is contiguous at `i * page_record_bytes + (the sum of the padded sizes of ranks before
r)`; its row slice likewise in the row record. `flare_selene::shared::SliceLayout` is the one
implementation of that arithmetic.

**Modes.** The manager decides each agent's mode when it attaches (`RankAttached.mode`):
`RANK_SHARED` (the agent reads and writes the manager's shared windows in place; only when it
is on the manager's host and rank 0 is in shared memory too, because window coordinates reach
an agent only through rank 0) or `RANK_CHANNEL` (it has its own data channel per transfer, named
in a `rank_channels` entry of `ClaimGranted`, `ExportRequest` or `PutAccept`; rank 0's own is the
message's `channel`). A channel carries only that rank's slice: `Shape.length` is the slice's
length, the frames are unchanged, and the channel is the stream carrier.

Rank 0 and each peer use one persistent `RankLink` for control, copy jobs and
completions. Losing that link stops the pool on that rank; completion still waits
for every submitted copy to drain. The manager's `Rank` stream supplies attachment
and fencing, and channel-mode copies use the per-transfer data connection above.

**13.1 Attachment.** `RankAttach{manager_incarnation, instance_id, rank0_nonce, rank,
agent_nonce, rank_secret}`. The manager checks the incarnation (`RANK_STALE_MANAGER`),
a live session of that instance run by that process (`RANK_NO_SESSION`: retry), the secret in
constant time (`RANK_BAD_SECRET`), the rank (`RANK_BAD_RANK`), and that a channel-mode agent has a
data listener to reach (`RANK_NO_DATA_LISTENER`). It assigns a generation from the session counter.
Claims, exports and puts of the instance, and claims naming it as a source, are refused
`RANKS_DETACHED` (put `PUT_NOT_READY`) unless every rank 1..W-1 is attached and not drain-only. A
new attachment of a rank supersedes the live one, which stays a drain-only party; the same
process reconnecting its session keeps its agents, another process taking the instance over does
not.

**13.2 Identity and fencing.** What an agent may still be touching is its own obligation under
`(instance_id, agent_nonce)`, never "rank r": a replacement inherits nothing and clears
nothing. Only the agent's own `RankFence{kind, id, DRAINED}` for the transfer, its own
`RankGoodbye{quiesced}` or its process's exit discharges it. Stream loss, replacement, timeout
and cancel are logical invalidation only. A channel leg is fenced by the manager's handler's
exit, which proves the manager's memory quiet, never that a remote rank's GPU copies drained: that
is the rank's own `Done` to rank 0, and rank 0 releases nothing and reports nothing before every
rank's.

**13.3 Drain-only attachments.** When rank 0's session ends its agents are not closed: they can
be granted nothing, but their existing streams can attest and say goodbye. A new attachment
requires a live rank-0 session. Stream loss retains shared-window obligations until the
process exits. After a manager restart all obligations are gone with the old regions.

**13.4 Windows and checksums.** A multi-process source states, per rank, slice transfer
checksums (`RankSums`): rank 0's `WindowFilled.rank_sums`/`PutFilled.rank_sums` for itself and
its shared peers, each leg's frame for its rank. A window is whole only when every accessor
has reported once (a bit per accessor, repeats are no vote). The manager then runs the
**assembly pass** on the window (off the actor, over several threads): it verifies the stated
slice checksums and computes the stored whole-page and row checksums for a single-process
requester or a put's commit, or verifies the stored (or stated whole) checksums and computes
each rank's slice transfer checksums for a multi-process requester (`WindowReady.rank_sums`,
the leg frames). A window is released to the store or the requester's pool only after the
pass is over, a window is free only when every reader has reported (the requester's session
for rank 0 and its shared peers, one leg per channel rank), an exact `CopyResult{COPY_OK}` that
arrives before a leg has finished reading is latched, and a put's `PutCommit` carries the
record the manager rewrites with the checksums it computed.

**13.5 RDMA.** Experimental RDMA carries only unsliced claim windows as signalled WRITEs.
Rank legs and puts use TCP. Each transfer owns its queue pair, whose destruction fences all
outstanding work before its handler reports `Closed`. The manager never advertises its region
keys or addresses; only the instance's private transfer staging is remotely writable.
