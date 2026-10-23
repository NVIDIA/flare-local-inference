# Direct execution: reproducing model faults without Helios

`ignition serve --direct` is a diagnostic and qualification path, requested by
project owner Luke Alonso, for reproducing and bisecting model-level faults. It
is not production serving and never replaces it. Production serving loads an
exported package in Helios, which owns scheduling, input packing, graph replay
and output without Python ([AGENTS.md](../AGENTS.md), "Python prepares; Helios
serves"). Direct execution prepares the model in the Ignition process, then
drives the prepared graphs from Python: a small fixed scheduler decides each
step, packs the same graph inputs Helios packs, and replays the captured graph
on every rank. With `--eager` it calls the forward each graph captured instead.
Nothing selects either implicitly: `--direct`, and `--eager` with it, must be
given, and neither `run`, `export` nor Flare accepts them.

A fault that reproduces through direct graph replay does not depend on Helios's
scheduler, transport frames or native replay. A fault that reproduces under
`--eager` does not depend on CUDA graph capture or replay; one that disappears
under `--eager` points at what replay does differently (see "Eager forwards"
below). Preparing with `--no-compile` additionally separates compiled forwards
from the recipe's Python forwards. Every run prepares from the checkpoint;
there is no package.

## Usage

On one host, GLM-5.3 Flash at TP1 on a GB300:

```sh
CUDA_VISIBLE_DEVICES=1 .venv/bin/ignition serve glm-5.3-flash --gpus 0 \
  --max-concurrency 8 --port 8000 --direct
CUDA_VISIBLE_DEVICES=1 .venv/bin/ignition serve glm-5.3-flash --gpus 0 \
  --max-concurrency 8 --port 8000 --direct --eager
```

Across DGX Sparks, rank zero adds `--ranks N --rendezvous ADDR:PORT` and each
peer runs `ignition join` exactly as for in-process serving
([multi-host.md](multi-host.md)); peers learn the execution mode from rank zero:

```sh
.venv/bin/ignition serve glm-5.3-flash --ranks 4 --rendezvous 10.200.0.12:29630 \
  --gpus 0 --max-concurrency 16 --host 0.0.0.0 --port 8000 --direct
.venv/bin/ignition join 10.200.0.12:29630 --rank 1 --ranks 4   # and ranks 2, 3
```

Every host needs the same Flare, Ignition and FlashInfer sources, as for any
multi-host preparation. Preparation flags (`--max-concurrency`,
`--prefill-chunk-size`, `--mtp-draft-tokens`, `--no-mtp-adaptive-verification`,
`--no-mtp`, `--no-context-parallel`, `--no-stream-overlap`, `--no-compile`)
apply as they do to `ignition serve`; direct execution runs what they prepared.
`--verbose` also prints every step's graph, row count and GPU time.
`--graph-times PATH` makes graph replay save its estimate of every graph's time
to `PATH` every 10 s and when it stops; `--eager` with the same file chooses
decode rounds from those times (see "Eager forwards"):

```sh
CUDA_VISIBLE_DEVICES=1 .venv/bin/ignition serve glm-5.3-flash --gpus 0 \
  --max-concurrency 8 --port 8000 --direct --graph-times glm-tp1-c8.json
CUDA_VISIBLE_DEVICES=1 .venv/bin/ignition serve glm-5.3-flash --gpus 0 \
  --max-concurrency 8 --port 8000 --direct --eager --graph-times glm-tp1-c8.json
```

The listener answers non-streaming `POST /v1/chat/completions`, `GET
/v1/models`, `GET /health` and `GET /metrics`, so
`scripts/repro_glm_batched_speculation.py` runs against it unchanged. Replies
split `<think>` reasoning from content. A request whose completion record
carries a device error fails with HTTP 500 and the message Helios uses,
`prepared model reported device error N`, with the code in
`flare_direct.device_error`. `/metrics` reports
`flare_speculative_rounds_total{batch,width}` as Helios does, plus steps by
kind, device errors by code, request outcomes and prefix-cache counters. Each
retired request prints one line on stderr: prompt and reused tokens, generated
tokens, speculative rounds and outcome.

## Design

**Scheduler** (`ignition/direct/scheduler.py`, CPU only). FIFO admission
places a request in a free slot when its pages fit: every page its prompt,
output limit and widest verification round can reach, plus the draft page, as
Helios reserves them. Pages are reference counted and handed out smallest
first. Admitted requests are initialized together, or restored together when a
cached checkpoint's consumed prefix is exactly their prompt, then a restored
request runs the resume graph. Prompts prefill one chunk per step, padded to
the smallest prepared capacity. Decode rounds and prefill chunks alternate
while both have work. A prompt not yet cached is snapshotted at its
endpoint and published as a checkpoint that owns the prompt's complete pages;
later identical prompts share those pages and restore the partial page into a
private one, as in Helios.

**Decode rounds** (`ignition/direct/policy.py`) for rectangular catalogs are
chosen as Helios chooses among its decode candidates (`select_plan` in
`helios/src/scheduler/policy.rs`, described in [scheduling-runtime-design.md](scheduling-runtime-design.md)),
so a fault that depends on particular graphs meets them as often as under
Helios. Every prepared batch capacity up to the concurrency's power of two
competes at every loaded verification width, or at the fixed width without
adaptive verification. Each carries its highest-valued decoding requests, so a
round can leave requests out or run in a larger, padded batch when that graph is
faster. A request's value is its expected tokens, from Helios's per-depth
acceptance estimate, priced by its token and delivery fairness, plus its output
age; the graph with the best value per graph time wins, with Helios's tie
breaks. Graph times are Helios's estimates: unknown until a step of that shape
runs, then a 7/8 moving average of observed graph time, forgotten after 30 s
without an observation. Unknown and forgotten graphs are explored before any
measured one, the most valuable first, so every batch and width runs at
startup, and a neglected one, such as a wide verification that lost to a
narrower one, runs again every 30 s. Service prices are initialized when a
step is planned, charged each completed step's graph time and credited with
its committed tokens and deliveries, as in `helios/src/scheduler/service.rs`.
`tests/direct/test_helios_decode_rounds.py` checks every round against the
rounds Helios chose for the same requests, graph times and acceptance.

For separate proposal and verification catalogs, a direct step runs the proposal
at the loaded depth followed by its selected verification graph. Its timing
covers both graphs. Width selection uses the preceding round's native confidence
curve, and proposal execution reserves the complete draft block's randomness.
The diagnostic scheduler does not retain proposals across steps or regroup
requests using fresh confidence. Those behaviors are exercised through native
Helios serving.

For combined packed-round catalogs, a candidate retains its graph operation,
request capacity, logical verification width, physical target-row capacity, and
proposal depth. Timing and cost estimates identify that complete shape. Cohort scoring uses conditional
confidence from completed rounds; width-one graphs compete independently. The
host freezes each request's prefix ceiling from its pending input, captured
width and depth, remaining output, and context budget. The combined graph
generates proposals and allocates actual prefixes on the GPU within those
ceilings and its aggregate physical capacity. Before committing a successful
batch, direct execution validates terminal completion and confidence for all
submitted requests. It does not retain proposals across steps or regroup requests
using fresh confidence.

The scheduling clock, which ages requests and expires estimates, advances by
each step's graph time, and by wall time only while the loop waits for
requests. Helios's clock is wall time, but its host work between rounds is
small; the direct loop's Python packing, uploads and bookkeeping are not, and
would otherwise age every request and expire estimates sooner than under Helios.

**Packing.** Each step's uploads follow the prepared graph's typed value bindings,
as in Helios's shared packer: state-row indices, row slots and
generations, page and draft page tables, token IDs, query lengths, sampling
flags, state-transfer commands and admission rows. Bytes it does not write
keep their resident values, as with native partial uploads. Combined packed
rounds bind `SPECULATIVE_PREFIX_LIMIT` and leave query lengths to the captured
GPU allocator.
Completion records are validated against the step before any state changes;
a record that contradicts it stops execution.

**Execution** (`ignition/direct/executor.py`). A rank copies each upload through
pinned staging on one stream, writes the frame index and submission sequence,
replays the graph, and waits for it before the next step. It checks any
collective runtime that reports transport failures only when asked, as Helios
does after each frame. Each rank times the graph with CUDA events from after its
uploads through the graph, as Helios's frame events do, and peers report their
time to rank zero; a step's graph time is its slowest rank's, as Helios takes it.

**Eager forwards.** With `--eager`, every rank releases its captured graphs
and, for each step, calls the forward that step's graph captured, compiled
where preparation compiled it, on the same buffers after the same uploads.
Scheduling, packing and completion handling do not change. The forwards are
built for capture and make no per-step host decisions, so they launch the
kernels the graph recorded, with the same arguments except for the addresses
of temporaries. Eager steps are bound by the host: on a GB300, GLM-5.3 Flash's
compiled forwards took 0.26 to 0.34 s per prefill chunk or speculative round
at every capacity exercised. What graph replay does and eager execution does
not reproduce:

- Replay runs the recorded launches from fixed addresses in the graph pool;
  eager forwards run their Python each step and take temporaries from the
  caching allocator, at other addresses. A fault that reads stale or
  uninitialized temporary storage can therefore behave differently.
- Replay schedules captured branches by their recorded dependencies, without
  launch gaps; eager execution orders them by stream and event as the Python
  launches them, so the timing of any race differs.
- A collective that orders itself across streams or checks transport health
  per call does so in eager execution; under replay its stream order is the
  graph's and health is checked after the step.
- An eager step's time is Python's launch time, 0.3 s where its graph takes a
  few milliseconds, and says nothing about the graph's cost. Eager execution
  therefore never schedules from it. With `--graph-times`, each eager step
  counts as the time graph replay saved for its graph, both as the observation
  Helios's estimates take and on the scheduling clock, so eager execution
  explores, expires and chooses graphs as graph replay did with those times.
  Choices still diverge when acceptance does, and graph replay's times follow
  its context lengths while saved ones are fixed. Without `--graph-times` no
  graph is ever measured, and every decode round carries every decoding request
  at the widest verification, which is Helios's choice among unmeasured graphs.
  Preparing with `--no-mtp-adaptive-verification` fixes the width in both modes.

**Ranks.** Rank zero decides every step and sends it to each peer as one
message over that peer's executor connection, the connection Helios uses for
per-step coordination, then executes it. Every rank therefore runs the same
steps in the same order with the same inputs and the same execution mode, and
the collectives inside each graph or forward join their GPU work. Rank zero
alone reads completion records. Stopping rank zero (SIGINT or SIGTERM) takes
effect between steps; it ends every peer's loop, then the group closes as after
in-process serving.

## Limits

- Prefix reuse covers exact full-prompt hits only. Partial prefixes, system
  boundaries and terminal checkpoints are not captured.
- One step at a time: no pipelined submission, and no mixed prefill and decode
  graphs.
- Step kinds follow the fixed order above, not Helios's value comparison across
  kinds; only the choice among decode-round graphs is Helios's. Prefill chunks,
  admission, restore, resume and snapshot steps keep their fixed shapes.
- No streaming, grammar masks, tool-call parsing, raw completions, Responses or
  Messages APIs, images or copied proposals.
- Eager forwards allocate their temporaries from the caching allocator after
  the graphs release their pool; preparation leaves only its headroom free
  beyond that, so `[memory] reserve` may need raising for large prefill chunks.
- GLM-5.3 Flash retains its forwards for `--eager`; other recipes support graph
  replay only, and `--eager` with them fails once preparation has finished.
