# Prepared model contract

Helios serves captured programs through one `ModelRuntime` and one `Server`.
Ignition recipes declare their inputs, storage, frontend syntax, and capacities
in `ModelIndex.contract`, a typed Protobuf `ServingContract`. Its versioned
`ModelContract` is decoded as `model::PreparedContract`; there is no per-model
native implementation or model-name dispatch.

The [Protobuf schema](proto/runtime_package.proto) defines the wire contract.
The [execution manifest](manifest.md) describes physical buffers, graph handles,
transport, and completion records. The
[runtime package design](../docs/runtime-package-design.md) specifies resource
ownership, signatures, loading, and cache allocation.

## Ownership and bootstrap

Recipes in `ignition/recipes/` declare semantic metadata beside their captured
catalogs. `ignition/preparation/contract.py` provides reusable declarations for
shared admission, sampling, state transfer, and paged-text components. A recipe
may bind different buffer names to the same value source. It cannot install
native scheduling or state-update code.

Build Helios with `server,runtime-package`. `NativeModel::load` selects graph
capacities, loads each rank's resources, and backs declared cache storage.
`NativeModel::manifests` exposes the resulting execution geometry.
`Bootstrap::load` validates the prepared contract and loads the tokenizer, chat
template, and typed frontend settings. `Server::start` starts the native model
workers and listener. Tensor-parallel peers use `RankPeers::Threads`.

The native owner retains graphs, CUDA contexts, weights, cache storage, mapped
allocations, and workspaces through confirmed execution drain. Serving performs
no Python calls, CUDA allocation, compilation, or graph capture. Missing graph
variants and inconsistent contracts fail startup; requests outside prepared
capacities fail admission.

Model index version 4 is required. Execution ABI 2/3 and protected envelope
version 4 have separate meanings. Packages with a different model-index version
require re-export; Helios has no schema-name fallback or migration loader.

## Serving metadata

`ServingContract` holds the served identifier and aliases, cache identity,
context and prefill extents, prepared page and snapshot capacities, selected
concurrency, speculative widths, and serving defaults. Load-time selection may
reduce captured concurrency and speculative width. Physical cache capacities
are chosen after graph loading and replace preparation's working-set counts.

The optional package rebuild manifest and per-rank weight catalogs describe
preparation inputs and stored tensor views. They do not select serving behavior.
Missing fields, or catalog versions unknown to a serving runtime, leave the
serving contract and replay path unchanged.

`ModelContract` contains:

| Field | Meaning |
|---|---|
| `version` | Semantic contract version, currently 1 |
| `vocabulary` | Valid sampled token range and histogram/mask geometry |
| `input_vocabulary` | Valid input range, including declared nonsampled image sentinels |
| `vision` | Feature-slot and per-window feature capacities, patch-to-feature ratio, and captured patch capacities |
| `cache` | Logical page tokens, attention-history storage, and page-table views |
| `graphs` | Per-program operation, value bindings, persistent-state accesses, and resource dependencies |
| `frontend` | Independent parser and image implementation identifiers plus their syntax and layout data |

Every enum has an unspecified zero value, which cannot select an implementation.
Unknown identifiers, unsupported versions, and unknown semantic JSON fields are
rejected. Enum numbers are stable; removed wire fields and values must be
reserved.

## Host values and graph state

Each `GraphContract` names one captured program and declares its operation.
The operation must agree with the execution catalog. A `ValueBinding` names a
registered transport destination and an enum-selected source:

| Sources | Values supplied by shared packing |
|---|---|
| Request identity and lane geometry | Slot, generation, live-row count, query length, sampling decision, and mixed prefill row |
| Tokens and positions | Prompt tokens, completed pending token, prompt positions, and image feature/context indices |
| Cache bindings | Page tables, live state rows, restore source, snapshot destination, and consumed-token endpoints |
| Admission sampling | Seed bits, sampling parameters, output limit, EOS IDs, full prompt histogram, and grammar mask |
| Copied proposals | Candidate IDs, count, generation, consumed endpoint, and per-position grammar masks |
| Constants | Declared integer values and inactive-row padding |
| Image tensors | Named outputs of the selected image layout implementation |

Indexing is explicit: scalar, submission row, active row, physical request row,
or packed token. Active rows omit inactive tails; submission rows initialize
padding across the request capacity. Packed-token indices follow each lane's
actual token count. Integer offsets and padding are declared separately.
Omitted transfers retain device bytes; scratch buffers need no host value source.
The execution transport owns frame index and submission sequence.

Shared packing resolves sources, types, extents, and indexing once at startup.
It uses bounded host staging, initializes admission histograms once, and uploads
only newly assigned page-table entries. Physical slot generations invalidate
remembered page-table lengths. Buffers remain alive through execution drain.

A `StateView` describes a byte offset, row stride, and byte extent within a
registered persistent buffer. Each graph lists read, write, or read/write
accesses. Device-owned pending tokens name a readable state view; initialization,
restoration, and prefill must declare producers of that view. A host-owned
pending token instead declares a value binding. Shared scheduling derives its
completion dependency from these concrete inputs.

### Device-allocated speculative prefixes

`PACKED_SPECULATIVE` declares one captured proposal, causal prefix allocation,
target verification, and accepted-state commit. Its execution descriptor carries
the request capacity, `draft_tokens`, logical `verify_tokens`, and
`packed_token_capacity`. The configured proposal depth is fixed across dispatches.
The host chooses physical target capacity before generating the round's proposals,
using completed observations rather than current draft values. The graph generates
the complete proposal block before allocating verification prefixes.

The `SPECULATIVE_PREFIX_LIMIT` input contains each submitted lane's immutable
upper bound, including its pending token. The graph initializes proposal
validation from those bounds, then writes actual lengths to device-owned
`query_lengths`; this operation cannot bind host `QUERY_LENGTH` values.
Helios fixes the cohort, original request slots, generations, pending tokens,
page tables and continuation rows for the entire submission. Page reservations
cover each lane's upper bound, including the cache contract's draft-page extent;
dividing the total target budget evenly among lanes is insufficient.

Every submitted lane starts with one pending position. Allocation repeatedly
selects the highest immediate-frontier survival probability, multiplying FP32
confidence values sequentially in FP64. Ties prefer the shorter prefix and then
the original request slot; floating-point total ordering distinguishes signed
zeros. Later confidence cannot influence admission of its predecessors. Zero
probability does not stop allocation while a permitted frontier remains.
The sum of admitted lengths cannot exceed the captured physical target capacity.
Rank-zero confidence is broadcast before allocation; validation agreement
suppresses commitment on every rank if any participating rank rejects the cohort.

Allocation consumes no random draws. Valid rounds preserve the proposal depth's
draft RNG reservation and the verifier width's target RNG reservation, regardless
of the admitted prefix. Terminal completion reports actual offered drafts,
accepted drafts and emitted tokens. Helios validates every offer against its
reserved lane ceiling and validates the aggregate target budget before changing
request state. Invalid ownership cannot mutate a replacement request.

There is no intermediate proposal completion. The target completion still
precedes host admission, stop processing and reservation of another round.
Cancellation suppresses publication but retains slots, pages, mailboxes and
scratch until the terminal fence drains. Embedding delivery uses one
GPU-produced request per table and submission, followed by the existing compact
row transfer; a round contains only one target pass. Host-stop and constrained
requests retain their ordinary width-one barriers. Packages containing only the
separate proposal and verification operations retain their existing semantics.

`state_rows_per_request` is a per-graph storage requirement. The runtime uses the
maximum among selected graphs to partition execution and snapshot rows. It is
not a model capability flag. Captured programs perform state updates and accepted
speculative-state commits; native code manages the declared storage.

## Attention histories and cache policy

`CacheContract.page_tokens` is the logical token count represented by one page
ID. History regions identify paged key/value, paged latent, sliding-window,
recurrent, or continuation storage. Paged regions name their actual cache planes
and compression ratio in tokens per record. Sliding windows declare their token
extent. Continuation rows preserve partial pages, recurrent and compression
state, saved hidden vectors, and other histories computed by captured programs.

Each page-table view declares its destination buffer and page offset. A shifted
draft view therefore expresses the extra page it references without selecting
policy by model family. Shared allocation, prefix reuse, eviction, preemption,
readmission, and offload consume the same storage contract.

A completion cursor names committed state and its lag behind target consumption.
For a successful speculative lane, the reported draft cursor must equal
`max(consumed_tokens - lag_tokens, 0)`. A graph with no draft cursor reports -1.
Sampled IDs must fit the vocabulary; the ABI3 reserved model field remains -1.
The scheduler independently validates generations, submission order, accepted
counts, and request endpoints before retiring any lane.

Snapshots preserve complete committed state at exact token endpoints. Immutable
checkpoints own complete pages; partial-page payloads remain in continuation
storage. Restore copies those payloads to private writable pages. Cancellation
and preemption retain resources through their final GPU fence. Failed execution
cannot publish a successful checkpoint.

## Frontend implementations

Tool parsers, reasoning parsers, image preparation, and image layouts each have
an independent enum selector. Implementations are reusable syntax or image
algorithms, not model identities. Qwen Coder function tags, GLM argument tags,
and DSML tool syntax may be paired independently with think-tag reasoning.
Wrapping newlines, tag separators, assistant boundaries, and reasoning terminal
markers are explicit syntax data.

The shared prompt path determines initial reasoning state from an open assistant
envelope, excluding completed user, tool, and assistant messages. Prefix-boundary
rules declare message roles and rendering constraints. Image token layout data
declares feature IDs and atomic spans; custom image layout code produces token
positions, feature mappings, and vision tensors. The generic graph packer binds
those tensors to declared destinations.

`OutputPipeline` selects reusable parsers. The shared presentation layer emits
JSON or SSE while preserving streamed tool IDs and argument fragments. The same
reasoning/tool envelope constrains structured output through llguidance; literal
tags inside a structured answer remain answer text. EOS, output limits, user
stops, cancellation, and execution failures retain distinct termination semantics.

The only runtime frontend assets are tokenizer data and the chat template.
EOS tokens, template tokens, image settings, and complete sampling defaults are
typed Protobuf metadata. Recipes declare model-card sampling recommendations;
Helios reads no checkpoint config files or model cards. Explicit request values
win over prepared defaults.

Frontend identifiers cannot select scheduling, cache policy, text-graph input
packing, or completion behavior. Boolean flags, callbacks, schema names, and
other enum values cannot disguise per-model runtime policy. Every field must
state a concrete syntax, storage, geometry, or dataflow requirement whose meaning
is independent of model identity.

## String stops

`stop` accepts null, one nonempty string or an array of up to four nonempty
strings totaling at most 4096 UTF-8 bytes. An empty array disables string stops.
String stops cannot be combined with token-stop IDs/sequences or structured
output. Matching uses incremental decoded text, including reasoning and tool
envelopes, before presentation. The first completed match removes the matched
string and its suffix, including text within the same sampled token. Partial
UTF-8 and overlapping stop prefixes remain buffered until resolved. Normal
EOS or output-limit termination flushes an unmatched prefix.

String stops use the host-stop scheduler barrier: width one and one outstanding
round while such a request is active. The worker determines the match before
submitting another round. Usage and raw-token observation count the complete
sampled token crossing a stop; visible text excludes the stop. The response
finishes with `stop`, including responses containing earlier completed calls.
A matched stop cancels the private publication candidate after its GPU fence;
previous immutable cached prefixes remain unchanged.

## Token and timing observation

Chat requests may set `return_token_ids=true`. JSON responses include
`prompt_token_ids` at the top level and raw generated IDs in
`choices[0].token_ids`. SSE includes prompt IDs in its first chunk and emits
one metadata-only chunk per generated ID, with `delta={}` and
`choices[0].token_ids=[id]`. Text, reasoning and tool deltas carry no duplicate
IDs. Observations include EOS and tokens omitted by stop-token presentation.
They retain normal EOS termination, prefix lookup, output limits and scheduling.

`return_token_timestamps=true` requires `return_token_ids=true` and adds an
aligned `choices[0].token_timestamps_ns` array. These host timestamps mark
validated active-request completions immediately after `scheduler.complete`,
before grammar advancement, stop handling or output parsing. They do not mark
GPU kernel completion or socket delivery. Nanoseconds are relative to a
process-local monotonic origin initialized at native runtime startup. Unobserved
requests read no per-token clock and emit no observation events.

Observed terminal responses include `reused_prompt_tokens` and ordinary token
usage. Timed responses also include one top-level `performance` object with
`ingress_ns`, `enqueue_ns` and `reused_prompt_tokens`. Ingress is chat-handler
entry after HTTP JSON decoding, before waiting for frontend preparation;
enqueue is immediately before submitting the prepared request to the model
queue. Both timestamps use the token timestamp origin.

`stream_options.include_usage=true` emits a final usage-only SSE chunk with
`choices=[]` before `[DONE]`; preceding chunks have `usage=null`. Otherwise an
observed stream includes usage on its terminal choice chunk. Observation flags
and `include_usage` accept only booleans or null; null is equivalent to false.
The existing bounded event queue also bounds observation events. A slow reader
can cancel an observed request through the same backpressure policy as normal
generation. Observation changes no CUDA graph, device allocation or replay ABI.

`stop_token_ids` and `stop_token_sequences` accept explicit vocabulary IDs.
At most sixteen sequences of at most 64 tokens are supported. Matching tokens
are omitted from presentation; pending prefixes are buffered until resolved.
A host stop cancels remaining scheduling and discards the private unpublished
candidate. It does not rewrite consumed model history or free in-flight storage.

HTTP admission allows at most `queue_capacity + max_active` requests, including
CPU preparation, queued generation, and streamed responses awaiting delivery. Admitted
requests wait asynchronously for one of `frontend_workers` CPU preparation
slots. The native ingress channel and pending admission deque each hold at most
`queue_capacity` requests. A dropped response cancels its request. A slow reader that fills its
bounded event queue cancels generation and receives an error if still
connected. Neither case blocks graph scheduling. HTTP shutdown allows five
seconds for connection drain, then drops remaining connections; CUDA ownership
remains governed by its separate stream-drain confirmation.

Full HTTP ingress returns 429; invalid request preparation returns 400.
Scheduler admission rejection returns 503 for nonstream requests. Temporary
page shortages wait for shared cache reclamation and preemption. Admission
rejection does not cancel other requests or mark the model unhealthy. Model execution, device and output errors
return 500. After SSE headers have been sent, either admission rejection or
execution failure emits one terminal `error` event under HTTP 200, without
`[DONE]`. Admission rejection uses a distinct native event; error text does not
determine the status code.

`/metrics` exposes native Prometheus counters and gauges for request lifecycle,
graph launches, prefix reuse, snapshot publication and eviction, and resource
capacity. Metric labels contain no request identifiers, prompt contents, or token IDs.
`flare_mixed_grammar_decode_steps_total` counts successful decode submissions
whose actual request lanes include both constrained and unconstrained requests.
`flare_constrained_max_inflight` records the maximum outstanding executor
frames immediately after a successful constrained submission. It is zero before
any constrained submission and must remain one afterward. Both measurements
cover the lifetime of the native model worker.
