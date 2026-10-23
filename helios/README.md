# Helios

Helios (`flare-helios`) is Flare's Rust runtime library. It supplies a native
frontend, prefix cache, live-request scheduler, CUDA package loader, and graph
execution through direct driver calls. The `flare/` crate's binary links this library
and serves a prepared package in the same process. Neither depends on Python
or the internal preparation tools.

The internal `ignition` command imports the `flare_heliopause` CPython extension
to prepare and export a model in its own process. Heliopause depends on Helios;
its crate owns the Python bindings and export allocator and builds no executable.
`ignition export` writes the package and closes the prepared model.
`flare run PACKAGE` loads it in a separate serving process. See
[runtime packages](../docs/runtime-package-design.md) for the resource contract.

The `server` feature supplies HTTP/SSE and model scheduling. Native package
serving uses the typed prepared contract for single-rank and tensor-parallel
execution. Raw replay and two-frame graph execution are available through
Rust APIs and heliopause's diagnostic Python bindings. In-process HTTP
bindings require heliopause's `qualification` feature; normal installations
serve only through a package.

During export, Ctrl-C and SIGTERM terminate Ignition directly, bypassing
Python finalizers and CUDA graph destruction. While serving, `flare run`
stops admission, gives active responses five seconds to finish, drains native
execution, and exits 0. Unexpected runtime failure exits 1. The library's
terminal dashboard reads native metrics without contacting the GPU.
`flare run` starts it after loading when stderr is a usable terminal; redirected
output remains plain startup logs.

Left/right arrows or Tab switch between the throughput overview and scheduler
details. The scheduler pane groups graph completions over two seconds by graph
kind, captured batch capacity (`b`), prefill/token capacity (`t`), or speculative
draft depth (`d`, verification width minus one). It shows call counts, mean and
maximum GPU duration, and each shape's share of observed graph time. Wider
terminals also show actual requests per call, which can be below the captured
batch capacity. GPU duration is the maximum participating rank's CUDA event
interval; these shares are not device utilization. Missing timings are counted
as untimed and excluded from duration averages.

Request rows identify the native slot and generation, keeping reused slots
separate. Their ten-second history marks completion observations, not GPU start
or end timestamps. Each cell shows the latest model-work completion; setup is
shown only when the cell contains no model work. Up/down arrows scroll requests.
Wide terminals show consumed prompt tokens, generated tokens, time since model
progress (`Gap ms`), time since visible output (`Out ms`), and useful-progress
deficit (`Debt ms`). Retired requests remain visible while their history remains.
Gaps and deficits refer to the request-state sample, whose age is displayed.
The last submitted graph displays its frozen estimate; `?` means indeterminate.

Dashboard telemetry is enabled only while the dashboard runs. The execution
thread writes to a preallocated 1024-completion buffer with a nonblocking lock;
the dashboard retains at most 16384 completions and 128 request rows. Overflow or
lock contention increments the displayed dropped-update count. The display
marks history truncated by its retention limit. Request state is sampled on
demand every 500 ms, and its age is displayed. Terminal input preserves signal
handling, so Ctrl-C follows normal server shutdown.

Prefill throughput counts consumed prompt tokens, excluding padding and prefix
reuse. Generation throughput counts committed output tokens, including EOS and
tokens withheld by text stops. Drafter steps count batched learned-proposal
forwards inside completed speculative graphs; verification steps count completed
speculative graphs. Copy-only proposals contribute no drafter steps. Mean draft
acceptance length is accepted draft tokens per progressing request round with
at least one offered draft, excluding the bonus token and width-one rounds.
DEPTH is allocated draft positions per progressing decode or verification request
round, including zero-draft rounds. Packed rounds contribute their allocated
per-request prefixes; the graph's maximum width and padded batch size do not
determine this average. Rejected drafts count toward depth.
The display also shows active/queued requests, completed/failed requests, reused
tokens, and cached/free KV pages. Token counters are available through `/metrics`.

Native APIs return `flare_helios::Error`, with separate `AdmissionError`,
`ExecutionError`, and `FrontendError` variants. Validation, capacity exhaustion,
unsupported features, grammar failures, image decoding, manifest violations,
worker failure, poison, and drain failures remain distinct through worker channels.
Errors retain their underlying source and CUDA operation/status code. Cloned
errors share their underlying cause.

HTTP request validation and unsupported features return 400. A prompt and
output limit beyond the prepared context return 400 with both token counts and
the context. Exhausted HTTP ingress capacity returns 429; exhausted model
admission capacity and unavailable workers return 503. Context and capacity
rejections append the configured `context_hint` or `capacity_hint`, which names
the setting that raises the limit. A missing or wrong bearer token returns 401
when the server configuration sets `api_key`; `/health` needs no token.
Execution and presentation failures return 500. SSE terminal failures and diagnostic Python exceptions serialize the same display
messages. Failed CUDA drains retain prepared resources. Worker recovery is unsupported.

The server exposes `/health`, `/v1/models`, `/v1/chat/completions`, and `/metrics`.
`/v1/models` lists one card, using the first nonempty `served_names` entry as its
`id`, or the checkpoint identifier when none is set. The card includes the
effective context limit in `max_model_len` and prepared capabilities, with the
checkpoint identifier in `root`. `/v1/models/{model}` returns the same card for
the checkpoint identifier or any alias, including repository IDs with slashes.
Unknown names return a JSON 404.
See [the API reference](../docs/api.md) for discovery fields and generation routes.
HTTP admission reserves at most `queue_capacity + max_active` requests before
reading JSON bodies. Admitted requests await a bounded CPU preparation worker;
the default two workers can prepare four simultaneous generation requests.
`ignition` packages set the queue to four times the concurrency limit, one worker per
concurrent request, and a body limit of the recipe's images plus 1 MiB.
An admission permit survives CPU preparation and the streamed response lifetime.
Disconnects cancel model scheduling, and slow consumers still face the bounded
per-request event queue. HTTP capacity and physical GPU slots are independent.
Tool-call history accepts OpenAI JSON-string arguments and object arguments;
strings must parse as JSON objects before checkpoint-template rendering.

`/metrics` emits Prometheus text without request labels, prompt text or token
IDs. Counters cover model admission/completion/cancellation/failure, prefix
lookups/hits/misses/reused tokens, successful launches by graph kind, ready
checkpoint publications and logical evictions. Gauges expose live/free/retiring
request slots, reusable pages/snapshots, cached checkpoints/pages, model pending
requests, HTTP/frontend capacities, seconds since the listener started, and tokens
generated during the most recent minute, sampled every five seconds. Prefix lookup counters include admission
attempts later rejected for resource pressure. Publication counts unique
checkpoints, excluding duplicate proposals. Eviction precedes GPU retirement;
free-resource gauges count only IDs whose fences have completed. Scrapes read
independent atomics and may span concurrent worker updates. Per-request deltas
require terminal responses followed by model quiescence, with no other traffic.

```sh
cargo test --manifest-path helios/Cargo.toml --offline
cargo build --manifest-path heliopause/Cargo.toml --features extension-module
```

The `images` and `server` features require native **libturbojpeg >= 3.0** and
`pkg-config` at build time. Install the development package that provides
`libturbojpeg.pc`; `pkg-config --atleast-version=3.0 libturbojpeg` must succeed.
The Cargo dependency enables only system `pkg-config` discovery, with no bundled
codec or decoder fallback. A missing or older library fails the build. Default
system linking is dynamic, so the matching `libturbojpeg.so.0` must also be
available to the runtime loader.

Image admission accepts base64 `image/png` and `image/jpeg` data URLs. JPEG uses
accurate IDCT, smooth chroma upsampling, and at most 100 progressive scans.
Baseline/progressive Huffman JPEG is supported; lossless and arithmetic JPEG
are rejected. Encoded byte and decoded pixel limits apply before pixel allocation.
PNG and JPEG EXIF orientation is applied before resize; orientations 1–8 include
mirrored and axis-swapped images. Invalid orientation metadata is rejected.
RGB conversion discards alpha and does not apply embedded color profiles.
Pillow is used only to generate test fixtures, never by native image admission.

Heliopause builds the internal Python extension named `flare_heliopause`. It requires the NVIDIA driver at runtime
and dynamically loads `libcuda.so.1`; building and running native unit tests does
not require CUDA libraries. Tests exercise queueing and teardown using explicit
Rust test functions. Production always calls the CUDA driver.

The `GraphReplayWorker` and `ReplayCompletion` Python bindings require the
`qualification` Cargo feature. Production package builds omit them. The server
uses its native execution transport directly.

Raw replay qualification entrypoints:

```text
current_context() -> (context_handle, device_ordinal)
GraphReplayWorker(owner, context, device, graph_execs, buffers, queue_capacity=8)
worker.submit(graph_index, repetitions=1) -> ticket
worker.poll(timeout_ms=0) -> ReplayCompletion | None
worker.outstanding -> accepted commands whose completion has not been consumed
worker.close()
```

`buffers` is a list of `(address, nbytes)` spans. Registration checks span
arithmetic and executable-handle uniqueness; it cannot prove that raw addresses
refer to live allocations. Preparation must supply valid handles and an `owner`
that strongly retains every graph, captured tensor, module, mapped allocation,
and CUDA context. The caller must finish warmup and synchronize preparation
before registration. Re-instantiation, tensor resizing, and concurrent external
mutation of captured storage are forbidden until `close()` completes.

One submission repeats a selected executable entirely in Rust, synchronizes the
worker stream, and emits a completion ticket. All submissions use the same
nonblocking stream, so aliased graph scratch is serialized. The completion queue
and pending command queue share a capacity bound: an unconsumed completion still
occupies admission capacity. `poll()` releases the GIL while waiting. This is a
replay primitive, not the model-serving loop's two-frame transfer pipeline.

Serialization alone does not preserve tensors allocated inside a shared graph
memory pool. Preparation must verify that the captured allocation lifetimes
permit the submitted variant order. Outputs or state that survive another
variant's replay require independent persistent storage; pooled intermediates
cannot escape their graph unless the preparation contract protects their
lifetimes.

Raw replay bypasses PyTorch RNG bookkeeping. Prepared graphs must use explicit
device RNG state and cannot contain generator-backed PyTorch random operations.

`close()` rejects future submissions, executes queued submissions, synchronizes
the stream, joins the thread, and then releases the Python owner. It is
idempotent. Launch failure prevents later launches and is reported on subsequent
completion tickets and on close. If stream drain cannot be confirmed, the owner
is retained and ultimately leaked rather than freeing potentially referenced
device storage. An unclosed object's destructor attempts the same drain; callers
must close explicitly before interpreter finalization. No device context is
created or destroyed by this crate.

Qualify real PyTorch-to-Rust graph replay on an explicitly assigned idle GPU:

```sh
PYO3_PYTHON=/path/to/python cargo build --manifest-path heliopause/Cargo.toml --features extension-module,qualification
CUDA_VISIBLE_DEVICES=GPU-UUID /path/to/python helios/tests/test_cuda_graph_replay.py \
  --extension heliopause/target/debug/libflare_heliopause.so
```

The integration cases capture real PyTorch graphs, change device inputs between
completed submissions, replay different graph capacities, check native repetition
counts, and verify graph-owner lifetimes during close and destructor draining.
They qualify interoperability and resource handling, not model correctness or
serving throughput. The selected Python environment is read without installation
or modification.

## Two-frame execution

[manifest.md](manifest.md) specifies the prepared execution ABI.
`execution::ExecutionManifest::validate` checks graph capacities, required
completion geometry, byte extents, contiguous strides, alignment, unique names
and graph handles, and nonoverlapping host/device I/O spans before startup.
The owner supplies live pointers on the declared context; metadata validation
cannot establish allocation lifetime or graph numerical behavior.

`execution::GraphExecutor` owns two pinned input frames, two pinned completion
frames, a model stream and a copy stream. `submit` copies selected input
subranges into a free host frame, enqueues asynchronous uploads and graph
execution, records a production event, and queues the completion transfer on
the copy stream behind that event. Omitted input ranges preserve device state.
`poll` returns ordered `FrameCompletion` records after transfer completion and
then permits frame reuse. A third submission receives backpressure; a grammar
submission requires an empty window and blocks further submissions until its
completion is consumed. The direct executor has no Python calls or per-step
CUDA allocation.

An executor belongs to one native OS thread. A Rust scheduler can call it
directly. `execution::ExecutionWorker` adds a synchronous native-thread control
channel for bootstrap and interoperability tests. The optional Python binding
uses that worker and releases the GIL around native operations:

```text
GraphExecutor(owner, manifest)
executor.submit(graph_index, [(buffer_name, byte_offset, bytes), ...],
                grammar_constrained=False) -> fence | None
executor.poll() -> ExecutionCompletion | None
executor.outstanding -> number of occupied frames
executor.drained -> whether close confirmed both streams drained
executor.close()
```

`ExecutionCompletion.records` contains four physical-slot records in the ABI's
eight-field order. `consumed_tokens` and `generated_tokens` are cumulative.
Submission failure poisons the executor. `close` drains both streams before
releasing pinned memory or the Python owner; failed drain retains both. Owner
release happens outside the ownership mutex, permitting a Python finalizer to
reenter `close`. Production request processing belongs in Rust; Python control
methods provide a qualification surface.

```bash
PYO3_PYTHON=/path/to/python cargo build --manifest-path heliopause/Cargo.toml \
  --features extension-module --offline
timeout 60s env CUDA_VISIBLE_DEVICES=GPU-UUID /path/to/python \
  helios/tests/test_execution.py --extension heliopause/target/debug/libflare_heliopause.so
```

The four CUDA integration cases cover physical-slot scatter, different graph
capacities, partial uploads, two occupied frames, grammar serialization, changed
resident inputs, rejected uploads, asynchronous owner retention, reentrant
finalizers and destructor draining. They use real PyTorch-captured executables.

## Live requests

`scheduler::RequestScheduler` owns up to 64 live request slots within the selected
concurrency and allocates private page tails and exact checkpoint snapshots from
prepared capacities.
Request handles contain a physical slot and a positive int64-safe generation.
Each physical request slot owns a fixed range of execution-state rows, leaving
state row zero for padding. Cached pages remain immutable and shared through pinned ancestors.
Resource pressure evicts unpinned cache leaves; retirement fences determine
when their page and snapshot IDs become reusable.

`RequestScheduler::plan` returns an immutable `ScheduledPlan` for initialize,
restore, snapshot, vision, prefill, resume-sampling, decode or speculative work.
The plan retains its scheduler identity and each participating request's state
epoch. Admissions of unrelated requests and timing observations preserve its
validity; mutation or retirement of a selected request invalidates it.
Prefill selects among loaded capacities up to `prefill_chunk_size`, using graph
duration, request age, and useful-progress deficit. Logical lengths pad to the
smallest loaded capacity that covers them. The limit must be 64-aligned, at least 64, and within the context capacity. Decode batches use
power-of-two capacities through the selected concurrency.
Prefill, decode cohorts, speculative widths, and mandatory stages compete in
one duration-aware selection policy. Full prompt cache hits restore state and
resume sampling from saved hidden state without replaying prompt tokens.

Before CUDA enqueue, callers check `GraphExecutor::can_submit`, obtain its
`next_fence`, and call `RequestScheduler::commit` with the plan to reserve pages
and record use of cache ancestors. Commit validates the plan's epochs and
resources without rerunning scheduling policy. Completion validation checks
generations, physical slots, monotonic counters and per-step logical bounds before
mutating request state.
Two decode steps can reference a request concurrently; a repeated terminal
completion does not emit its sampled token twice. A CUDA submission failure
requires confirmed stream drain before `release_after_shutdown` releases
resources associated with the attempted submission.

Each request owns conditional draft-acceptance statistics by depth and proposal
source. Validated speculative completions update only that request. The emitted
accepted prefix supplies positive observations. When a correction token was
emitted and an offered draft remained, the next draft supplies a rejection.
Termination within an accepted prefix censors the remaining depths. Expected
yield is clipped by the remaining output limit, context capacity, and available
proposals. A fixed captured width may therefore contain fewer active tokens.
Acceptance learning and accepted-draft metrics exclude proposals beyond the
lane's logical token limit even when the completion reports physical padding.

`estimated_duration_ns` returns one estimate per captured graph, initially zero
(unmeasured). A valid completion sets its cost directly; further fresh observations
update it with an EMA of weight 1/8. After 30 seconds without a valid observation,
the cost becomes stale but remains available for selection. The next valid
observation replaces the stale estimate. Tensor-parallel observations use the
maximum positive rank interval. Isolated executions also measure host handoff;
queued executions exclude queue wait from the timing updates. Unmeasured
candidates compete by useful work and can beat measured candidates. Both
unmeasured and stale physical dispatches require an empty queue and completion
before another submission.

Ordinary valid execution earns refresh credit at `0.002 / 0.998` times its
measured duration, capped at 5 ms. Full credit permits a stale alternative to
displace the ordinary winner, ordered by the later of its last valid observation
and last attempted dispatch.
Refresh pays for every physical capture, including an enabling proposal and its
verification captures. Required verification can continue through debt; another
refresh waits for full credit. Idle discards positive credit and preserves debt.
This bounds aggregate refresh expense, not individual stalls; see
[funded timing refresh](../docs/scheduling-runtime-design.md#funded-timing-refresh).

Speculative width selection combines this cost with request-local expected
prefix survival. Combined graphs learn from completed verification; split
graphs keep proposal-confidence forecasts separate. Ordinary selection uses
historical means, while a separate 1% raw-time budget funds optimistic
acceptance probes. Observational weight ages without changing unobserved means;
width-one rounds still measure cost. See
[funded acceptance learning](../docs/scheduling-runtime-design.md#funded-acceptance-learning).
Each request accrues equal entitlement while GPU work is outstanding;
completion earns useful-progress credit at frozen reference rates. Unknown
reference rates initialize from runtime observations and credit accumulated
progress once. Timing freshness does not alter those accounting units.
The latency filter includes queued work and prerequisite stages, preserving
separate clocks for prompt progress and published output.

`PolicyConfig` defaults to a 20 ms graph slice, 20 ms submission window, 40 ms
service-gap target, and 20 ms deficit scale. An overdue graph stops further queueing
until completion. Unavoidable atomic singleton work may exceed a duration target
with an empty queue. These are estimated-time preferences, not hard latency
bounds. Residency rotation and mixed captures require further implementation;
see [the native scheduling design](../docs/scheduling-runtime-design.md).

Admission reserves checkpoints for unreused system, prompt and terminal endpoints.
Prefill stops at the exact rendered system boundary and prompt end. Snapshot graphs
copy live state and partial cache tails to immutable snapshot rows after the
selected requests' model submissions complete. System and prompt publication retain the live physical
page sequence while transferring complete pages to the cache. Admission reuse
counts stay fixed. Duplicate publication never remaps the live request.

`finish` publishes exact terminal consumed state after its successful snapshot
copy, excluding the final pending sampled token. A child at T after parent S owns
`floor(T/64)-floor(S/64)` complete pages. Partial payloads restore from the snapshot
into private pages. Failed copies prevent publication. `cancel` stops scheduling
immediately and retains slot/page/snapshot ownership through final GPU use.
Generation reuse waits for that retirement.

This scheduler operates on logical IDs and completion records. `model::PreparedContract`
declares typed graph value sources, storage views, dependencies, and frontend syntax.
Shared `model::Inputs` binds and packs admission, state, vision, and forward inputs.
`server::ModelRuntime`
owns the scheduler and executor on one native model thread. Its private `Worker`
owns ingress, pending and active requests, model admissions, and reusable input
storage. Admission precedes ordered completion processing, terminal publication,
copy-proposal planning, and submission. Completion-triggered retirement runs
before polling another completion; exact system and prompt publication remains
atomic inside `RequestScheduler::complete`.

The model thread retains the executor outside the `catch_unwind` closure that
runs `Worker`. Worker failure notifies active, pending, and queued consumers
before draining the executor;
scheduler resources release only after confirmed drain. Input storage remains
owned through that drain. Model packing initializes prompt penalty counts once
and uploads only newly assigned page-table entries during decode.
[model-contract.md](model-contract.md) specifies the concrete handoff.

## Native frontend

The `frontend` Cargo feature provides `Frontend::load` over explicit tokenizer and
chat-template assets with expected SHA256 fingerprints, plus typed token settings.
Each asset is hashed and parsed from the same bytes. No model lookup, downloading,
Python import, or Python callback occurs in this API.

`Frontend::render_chat` preserves JSON message content and tool argument order,
applies the prepared recipe's template, and returns prompt bytes, token IDs,
and complete-message token endpoints. Template defaults fill missing options;
explicit request options win. `frontend::jinja` compiles Jinja statements and
expressions to compact bytecode at asset load. Constants, names, call arguments,
and jump targets belong to the immutable compiled template; operand stacks,
lexical scopes, namespaces, and output buffers belong to each render. Rendering
borrows JSON input values, emits concatenations directly into the output, and
appends namespace lists without copying their accumulated contents. Python
methods and the JSON filter preserve Hugging Face spacing, Unicode, insertion
order, and float formatting. MiniJinja 2.24.0 is a development-only differential
and benchmark reference.

Recipe templates annotate complete preambles and messages in their actual
rendered order with `{% mark %}`, which records the UTF-8 output byte position
without emitting text. Segmented templates use `{% mark expression %}` with a
nonnegative segment count, translated through the segment text lengths after
rendering. Marks are forbidden inside macros, captured assignments, and filter
blocks; a root call site marks output after a macro returns. Generation headers
remain outside completed messages. Annotations follow Jinja whitespace controls.

The exact deployed Qwen3.8 and inference-transformed Step templates are replaced
in memory with their annotated recipe assets. Other unannotated templates fail
to load. Checkpoint-derived annotations must reproduce the source bytes when
removed. Gemma 4 selects one of two annotated Google canonical templates by model
size across all supported quantization publishers; see
[recipe frontend metadata](../docs/model-development.md#sampling-and-frontend-metadata).
Package files and metadata formats remain unchanged.

DeepSeek V4 and V4.1 use the canonical native encoder, patched to record byte
positions in its normalized-message append loop. Adjacent users share a merged
message endpoint, and reordered tool results receive endpoints in output order.
Continuation truncates the rendered text before tokenization and clips later
positions. Existing text, ordinary-token, and image-placeholder encoding divisions
are preserved. The single encoding supplies original byte offsets; a cut is
retained only if every preceding token ends at or before it and every following
token starts at or after it. Overlapping Unicode spans and BPE tokens crossing a mark cannot
create partial-token cuts. Token endpoints are sorted, unique, and strictly
inside the prompt. Image expansion adjusts them with one ordered pass.

`Frontend::encode`, `decode`, and `incremental_decoder` use Hugging Face's Rust
tokenizers library pinned to the fixture environment's version. Incremental
decoding preserves incomplete UTF-8 between chunks and flushes remaining bytes
at completion. It does not parse reasoning or tool-call events.

Run exact-checkpoint CPU parity qualification:

```sh
cargo run --manifest-path helios/Cargo.toml --features frontend,test-tools --bin test_frontend -- \
  /absolute/path/to/model/snapshot tests/frontend/qwen38.json
```

This compares rendered bytes, token IDs, template exceptions, full decoding,
incremental decoding, and configured-default precedence with golden outputs
generated by the pinned Transformers environment. Image contents render the
template's image placeholders; image decoding, patch expansion, and preprocessing
are separate model-processor work. The frontend does not expose HTTP sockets,
run a model, support arbitrary Python tool callables, or claim compatibility
with every Transformers template extension. JSON inputs use UTF-8 strings,
64-bit integer ranges, and finite double-precision numbers.

The Jinja environment and continuation semantics follow Transformers
`utils/chat_template_utils.py` at revision
`4cc2aa84301c9aa210b5513dab9fefae03981f6e` (Apache-2.0).

`cargo bench --manifest-path helios/Cargo.toml --features frontend --bench template --
NAME=/absolute/path/to/chat_template.jinja` compares annotated bytecode rendering
against the equivalent unannotated MiniJinja template. It measures compilation,
warm median and p95 latency, allocations, peak live bytes, compiled storage, and
four-thread throughput on short prompts, long histories, large messages, tools,
and images. Tokenization is excluded. `--check` runs output comparisons without
timing; `--owned-only` measures the owned renderer while retaining output checks
and concurrent comparisons against MiniJinja.

## Embedding row cache

Disk-backed embedding tables use a bounded host-RAM row cache, defaulting to
4 GiB per host on discrete GPUs or 1 GiB when any selected GPU shares host memory.
`flare run --embedding-table-cache-size SIZE` overrides the budget; `0` disables
caching. RAM table storage allocates no separate row cache. The loader accounts
for its host storage before sizing KV pages and continuation snapshots. Entries hold immutable
table rows shared across requests; they copy into existing pinned row staging,
whose consumer fences remain unchanged. Approximate usage counts protect frequently
requested rows; recency breaks ties, and periodic count decay adapts to workload
changes. Each server start creates an empty row cache; request gathers populate
it without persistent cache files. `/metrics` exposes `flare_engram_` gather
time, disk traffic, cache hits, misses, evictions, occupancy, capacity and bytes.

## Prefix checkpoint policy

`cache::PrefixHistory` incrementally hashes 64-token blocks with BLAKE2b-128.
Exact partial endpoints hash the aligned parent plus the remaining tokens and
conditioning, preserving the existing complete-block identity.
The root and each block include a 32-byte model/recipe identity. Conditioning
events carry absolute token positions and 32-byte identities, retaining their
order even at the same position. An event at position N first affects prefixes
that consume that position; appending a later image preserves earlier keys.
Exact endpoint keys remain stable when later tokens or conditioning are appended.

`cache::PrefixCheckpointCache` owns metadata for immutable exact checkpoints.
A checkpoint privately owns its tail pages and one complete state snapshot;
ancestor pages are shared through parent links. Snapshot consumed length must
equal the endpoint exactly. The parent must match the supplied history,
and pages and snapshot IDs must fit prepared capacities without overlapping
existing or retired ownership.

Publication has two outcomes. `Created` transfers the proposed resources to the
cache. `Existing` keeps the proposal's unused resources with its caller; any
outstanding work on those resources must complete before the caller reuses them.
A created checkpoint whose readiness fence is incomplete reserves resources but
does not participate in lookup. `advance_completed` publishes it after the
reported fence completes. Pending-publication cancellation also observes that
fence before releasing resources.

Pinning increments the selected checkpoint and every ancestor. Direct pin
accounting prevents an ancestor unpin from consuming a descendant's reference.
The scheduler records GPU use through `mark_used` before submission; releasing
logical pins does not imply completion of that use. Leaf LRU eviction accepts
separate page and snapshot pressure targets and produces `RetiredResources`,
which explicitly carry their final-use fence. The cache continues reserving
those IDs. Only `advance_completed(...).reusable` grants permission to reuse
the corresponding storage.

Fences describe a single monotonic completion timeline covering model execution
and associated transfers. The caller supplies truthful completion information;
this metadata library never queries CUDA or performs state copies. Checkpoint
IDs are monotonic within one cache and are not reused after eviction. Resource
IDs are zero-based logical ordinals within the prepared page/snapshot capacities;
the prepared graph contract maps them to backing storage and padding sentinels.

## Native image preparation

The `images` feature adds `frontend::images::ImageProcessor` and PNG, JPEG,
and static WebP decoding; it also enables `frontend`. Typed package metadata
selects the preparation profile and token layout independently. Profiles cover
Qwen, MiMo, DeepSeek, DeepSeek row-major, Moonvit, Step, and RADIO inputs.
The Qwen processor uses an RGB8, bicubic-antialias profile with
16×16 patches, temporal duplication by two, spatial merge by two, a 48×48
learned position table, and fused normalization `(pixel - 127.5) / 127.5`.

`process` produces packed `pixel_values[N,1536]` float32, image grids `[1,H,W]`,
vision positions `[N,2]` int64, interpolation indices `[N,4]` int64 and weights
`[N,4]` float32, and per-image cumulative patch offsets int32. Patch features
are ordered by channel, temporal copy, pixel row, and pixel column; patches
are ordered by 2×2 spatial merge blocks. `expand_prompt` consumes the rendered
chat text, expands each image placeholder, tokenizes the result, and returns
modality IDs and four unpadded position rows `[text,T,H,W]` plus the mRoPE delta.
The model scheduler can transpose the last three rows into `[tokens,3]`.

Prompt construction describes each image without building its tensors. The
description includes geometry, semantic identity, token dependency span, and
canonical feature range. Admission validates each image against its captured
patch capacity and feature-window capacity. Total conversation image tokens
are bounded by context, with encoded-body limits enforced separately.
Feature-slot concurrency limits requests needing vision computation.

After prefix lookup, only images whose dependency spans extend past the
consumed endpoint require tensors or vision graphs. A fixed CPU worker prepares
those inputs. Host retention can reuse source descriptions and prepared
tensors through the configured memory and disk tiers. Whole images are grouped
into windows that fit the prepared feature arena. Each window's features are
packed contiguously from zero without changing canonical prompt positions or
indices. Prefill stops before the next window's first image dependency; atomic
image spans remain indivisible.

A request spanning multiple windows reserves one private snapshot row. After
prefill consumes a window, a snapshot preserves complete continuation state,
including partial cache pages and recurrent histories. Restore installs that
state under a fresh execution generation before vision writes reuse the arena.
The snapshot stays private, does not require host retention, and does not add
cached-token usage. GPU fences protect the snapshot row, feature slot, and
inputs through cancellation and retirement. Preemption and failed host restores
retain encoded sources and replan from the restored endpoint. See
[host retention](../docs/prefix-cache-offload.md).

`RgbImage` holds the decoded representation required by its profile, including
alpha and CMYK data where preprocessing consumes them. Semantic identities
include these channels and resize mode as applicable. Encoded-source keys
include the declared media type, source bytes, and processor settings.
Animation, video, and remote URL fetching are unsupported. Decoding and
preprocessing execute in Rust.

The resize coefficient generator and integer convolution follow PyTorch
[`UpSampleKernel.cpp`](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cpu/UpSampleKernel.cpp)
and its uint8 antialias implementation at `cf30153c4c131c8164ee7798e5022d810682e2cb`;
the applicable license is [licenses/PyTorch.txt](licenses/PyTorch.txt).
The scalar Rust implementation preserves PyTorch's adaptive coefficient
precision and intermediate uint8 rounding. The `llm-multimodal` implementation
at `f0985ef65967615db2c79279aa07818499301bfd` uses Pillow's different coefficient
precision and serves as an interface reference, not a pixel reference.

Image fixtures in `tests/frontend/qwen38_images.json` pin Transformers
`4cc2aa84301c9aa210b5513dab9fefae03981f6e`, PyTorch `2.13.0+cu130`, Torchvision
`0.28.0+cu130`, Pillow `12.3.0`, and all checkpoint asset hashes. They use the
actual `AutoProcessor` Torchvision backend and canonical Qwen4Exp position
helpers. SHA256 checks compare every resized RGB byte and every little-endian
float/integer tensor byte, with no numerical tolerance. The fixture set covers
19 complete pipelines and six geometry cases, including the 16,384-patch
boundary, half-even rounding, independent-axis resize, PNG color modes, and
mixed/repeated image batches.

```bash
cargo test --manifest-path helios/Cargo.toml --features images --offline
cargo run --manifest-path helios/Cargo.toml --features images,test-tools --offline \
  --bin test_images -- "$SNAPSHOT" tests/frontend/qwen38_images.json
.venv/bin/python tests/frontend/generate_image_fixtures.py \
  "$SNAPSHOT" tests/frontend/qwen38_images.json \
  --binary-dir /tmp/flare-image-reference
```

The qualifier accepts an optional third argument for writing mismatching
native tensor bytes. Fixtures contain fingerprints rather than the full
100 MiB largest patch tensor; the generator's `--binary-dir` records full
reference tensors for diagnosis. These checks qualify CPU preparation; vision
graph execution and decoder consumption require separate model integration.
