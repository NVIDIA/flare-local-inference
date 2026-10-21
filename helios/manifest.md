# Prepared execution manifest

ABI versions `2` and `3` describe graph execution and transport over one existing CUDA
context. A resource owner retains the context, graphs and allocations through
shutdown. `NativeResources` reconstructs these resources from a runtime package written
by Ignition before native serving starts. The model recipe defines the numerical behavior of each executable.

The Python representation is a dictionary with these fields:

```python
{
    "abi_version": 2,
    "context": int,                 # Existing CUcontext handle.
    "device": int,                  # Visible device ordinal.
    "max_active": int,              # 1 through 64.
    "input_frame_bytes": int,       # Positive, multiple of 8.
    "buffers": [
        {
            "name": str,
            "address": int,
            "nbytes": int,
            "dtype": str,
            "shape": [int, ...],
            "strides": [int, ...],  # Element strides; contiguous row-major.
        },
    ],
    "graphs": [
        {
            "name": str,
            "kind": str,
            "sampling_mode": str,   # general.
            "handle": int,          # Instantiated CUgraphExec handle.
            "batch_capacity": int,
            "token_capacity": int,
            "input_bindings": [
                {
                    "buffer_name": str,
                    "host_offset": int,
                    "nbytes": int,  # Entire named buffer extent.
                },
            ],
        },
    ],
}
```

Supported dtypes are `bool`, `uint8`, `int32`, `int64`, `float32`, `float16`, and
`bfloat16`. Shapes are nonempty and every extent is positive. `nbytes` equals
the contiguous element count times dtype size. Device addresses and host
offsets are aligned to that size. Names are nonempty and unique within their
lists. Graph executable handles are nonzero and unique. Registered device
spans do not overlap; captured weights and scratch need not appear in this I/O
manifest but must remain in the resource owner.

Graph kinds are `init`, `restore`, `snapshot`, `prefill`, `decode`, `resume`,
`vision`, and, in ABI3, `speculative`. Decode and speculative batch capacities are powers of two through 64.
Decode has one token per packed lane. Speculative graphs have an integer
verification width from 1 through 8 in `token_capacity`.
Prefill has batch capacity 1. Token capacities are 1, 16, or a multiple of 64
within 64..=262144. The model configuration separately requires the captured
catalog for its `prefill_chunk_size`; transport validation does not infer a
scheduling policy or substitute a missing graph.
State operations
and resume use power-of-two batch capacities through 64. Vision batch capacity is 1, and its
token capacity is one of 256, 1024, 4096, or 16384 patches. Batch capacity can
exceed `max_active` for padding, but must not exceed its next power of two.
Request storage capacity is `max_active.next_power_of_two()`. The manifest records
capture capacity. Runtime admission may use a smaller `SchedulerLimits.max_active`,
while `SchedulerLimits.request_slots` remains the captured transport capacity.
Transport buffers retain their shapes, strides, and completion-frame extent.
Continuation storage has one null row, an execution partition sized by selected
concurrency and verification width, and a prefix-cache partition. Request-to-row
bindings address those partitions independently of transport slot indices.
A loaded manifest contains only the selected graph
variants, and scheduling chooses verification widths from those actual graphs.

The sampling mode is `general` for every graph. Requests require a positive
normal FP32 temperature and use the same stochastic sampling executable.
Unsupported mode strings reject manifest parsing before CUDA initialization.
A model contract defines the required graph capacities.

Two graph inputs are reserved and required in every executable's bindings:

| Buffer | Dtype | Shape | Writer |
|---|---|---|---|
| `frame_index` | int32 | `[1]` | Native executor |
| `submission_sequence` | int64 | `[1]` | Native executor |
| `completion_records` | int64 | ABI2 `[2,capacity,8]`; ABI3 `[2,capacity,24]` | Captured graph finalizer |

`completion_records` is an output and cannot appear in input bindings. Each
buffer has one fixed host offset shared across all graph bindings that name
it. Different buffers' host spans do not overlap. A command can upload any
subset of its graph's non-reserved bindings, including aligned subranges;
omitted ranges preserve resident device values. Preparation initializes all
device inputs before handoff. The native executor always uploads the reserved
frame index and positive, monotonic submission sequence. Command callers
cannot override those two values.

An optional `readiness_timestamps` output has dtype int64 and contiguous shape
`[2,3*N]`, with `1 <= N <= 64`. Each frame holds one triple per embedding table:
GPU timestamps before its external readiness wait, after readiness, and after a
prepared all-rank reduction. Table indices follow the embedding catalog. Every
graph consuming embedding tables writes all triples for its selected frame.
Other graphs leave them untouched. The executor retains each frame's readback
through its transfer fence. Packages without this output retain raw replay-cost
estimation.

Across ranks, the estimator takes the minimum readiness-through-reduction span
and the minimum reduction-only span separately for each table. Their difference
estimates delivery delay exposed after the last rank arrives, retaining the
reduction's own cost. This uses local durations and requires no cross-device
clock alignment. Differences between ranks' reduction completion instants remain
measurement error. The reported CUDA intervals and service charges retain all
elapsed time.

Each completion lane begins with eight signed 64-bit fields in this order:

1. Positive request generation, bounded by `INT64_MAX`.
2. Submission sequence.
3. Consumed-token count.
4. Generated-token count.
5. Sampled token ID, or `-1` when no token was sampled.
6. Status: `0` inactive, `1` running, `2` finished, `3` error.
7. Error code, `0` on success.
8. Reserved, `0`.

ABI3 appends these fields at zero-based indices:

| Index | Meaning |
|---|---|
| 8 | Emitted token count, 0 through verification width |
| 9 | Offered draft count, 0 through width minus one |
| 10 | Accepted drafts within the emitted prefix |
| 11 | Bitmask of offered drafts supplied by copying; bit zero is the first draft |
| 12 | Reserved field; every prepared model contract requires `-1` |
| 13 | Committed draft-cache length, `max(consumed_tokens-1,0)`; `-1` for a copy-only speculative round with no draft cache |
| 14 | Reserved, `0` |
| 15 | Verification width, an integer from 1 through 8 |
| 16–23 | Emitted token IDs in order, with unused entries `-1` |

The scalar token in field 4 equals the last vector token, or `-1` for an empty
vector. Accepted drafts number either `emitted_count-1` or `emitted_count`;
the latter occurs when EOS or the output budget clips inside the accepted draft
prefix. It never exceeds the offered count. Ordinary sampling emits at most
one token and reports width one. Maintenance and non-sampling prefill emit
zero tokens. Every graph in an ABI3 catalog writes the 24-field layout.
The executor validates vector bounds, padding, epochs and transport-reserved
fields before delivering records. The model contract validates field 12 before
request state advances. A speculative or verification completion requires a
token vector whose verification width (field 15) equals the submitted graph's
`token_capacity`, and its record's consumed-token increase since submission
must equal its emitted count; such a completion without a token vector is
invalid. Each speculative or verification graph declares one fixed
`token_capacity`, so a finalizer writes that value into field 15 of every
record it emits.

The lane index is the physical request slot within the configured capacity. A captured finalizer scatters
packed model rows through the request-slot mapping. Every launch writes the
selected frame's records, marking unused lanes inactive. Records for
active lanes carry the submitted sequence and the corresponding request
generation. Finished requests suppress further mutation inside subsequent
graphs. The model scheduler requires an active completion for every submitted
live maintenance or vision lane; an inactive row cannot confirm a state copy.

The executor allocates two pinned input frames and two pinned completion
frames. Input updates, graph execution and production events use the model
stream. A copy stream waits for each production event, transfers the selected
completion row, and records its completion event. A frame remains occupied
until that transfer completes. Ordinary work allows at most two submissions
in flight. Every speculative graph uses a submission barrier, including direct
executor calls without an explicit grammar flag. Grammar-constrained work and
host token stops also run with one submission in flight,
so host token processing finishes before the next generation step.

Each graph descriptor carries `estimated_duration_ns`; zero denotes an
indeterminate duration. Ignition leaves this field at zero. Helios starts every
graph indeterminate and learns its cost from GPU completion intervals, including
tensor-parallel execution and excluding host queue wait. A first measurement sets
the estimate; fresh measurements update it with weight 1/8. Estimates expire after
30 seconds without a valid observation and are learned afresh on the next replay.

Every submission uses preallocated CUDA timing events around the entire
captured graph. `FrameCompletion.timing.gpu_ns` retains each rank's local interval,
excluding input uploads and the completion transfer. Events and frames remain
occupied through the transfer fence. `timing.wall_ns` spans host dispatch through
observation of all participating ranks' completions, including input staging,
rank coordination, device queueing and completion delivery. A maximum of local
GPU intervals is not the rank group's wall interval. `timing.queued` identifies
single-rank frames dispatched with another frame outstanding; these host
intervals must not be summed as GPU busy time or treated as isolated service cost.
The Heliopause `ExecutionCompletion` exposes `gpu_ns`, `wall_ns` and `queued`.

Submission sequences also identify the ordered completion fence. Resource
retirement waits for the completed fence covering every operation that can
reference the resource. Shutdown retains graph owners, context owners, and
pinned host frames until both streams drain. An unconfirmed drain preserves
ownership and reports failure; it cannot authorize memory reuse.

The manifest describes transport and graph capacities. The model recipe's
request-state, page-table, positional and sampling input names are additional
named buffers and bindings. A scheduler adapter checks those semantic names
against the recipe before serving requests.

## Native runtime package

`ignition export MODEL --gpus N[,N...] --output model.safetensors` prepares a
native package for `flare run model.safetensors`. Each TP rank has its own
resource bundle inside the file. Helios serves ranks as threads in one Rust
process. Ignition exits or replaces its Python process through `execv` before
the native server starts. The CUDA driver creates context, module, function,
graph, and executable handles from the package descriptions.

The safetensors file contains two one-dimensional `U8` tensors:
`flare.payload` holds all resource bytes and alignment padding;
`flare.metadata` holds the Protobuf `ModelIndex` at the end of the file.
The JSON header is limited to 64 KiB, padded to place the payload on a 4 KiB
boundary, and declares `__metadata__: {"format": "flare"}`. Stored allocations
and embedding tables from all ranks precede execution resources. Resource
spans are byte offsets relative to `flare.payload`, bounded by its size.
The index is limited to 16 MiB. It includes:

- Ordered rank bundles and peer-memory ownership and access declarations.
- Architecture, context flags, limits, cache settings, and address reservations.
- Allocations with packed payload spans, or runtime sizing by continuation rows
  and KV pages with fixed virtual bases and maximum virtual extents.
- Modules, globals, function symbols, parameter sizes, and function attributes.
- Graph declarations, function dependencies, and graph-body resource spans.
- Serving configuration, typed sampling and frontend settings, and tokenizer/chat-template resource spans.
- Offloaded Engram geometry, encoding, table spans, compact input names, and
  named external event waits.

Native manifests for models with Engram tables carry a shared host reader and
rank-local event bindings. Table rows use separate preallocated GPU inputs;
ordinary graph uploads contain the CPU-computed hash IDs. One step may be in
flight while the reader owns the shared pinned row buffers. Each rank queues
its readiness wait, H2D copies, and event record before launching the graph.
Consumer event waits preserve the dependency between those transfers and the
captured lookup. Reuse requires completion on every rank. Shutdown drains row
copy streams in addition to model and completion streams.

Graph declarations specify kind, sampling mode, batch capacity, token capacity,
draft source, draft count, and verification count. Verification count includes
the pending input. Selection uses these fields before reading graph bodies or
loading kernels. MTP alternatives for one batch and verification width are
selected exclusively. Weights, including MTP weights, remain loaded.

Graph bodies, module images, globals, tensor payloads, and frontend assets are
separate indexed spans in the file. Each graph body contains its declared
number of nodes followed by its edges, encoded as u32 little-endian lengths
and Protobuf records. Records are limited to 1 MiB. Helios decodes one record
at a time and drops CUDA graph templates after instantiation. Tensor loading
uses 16 MiB of pinned staging memory across all ranks; module images are
mapped from the package while loading.

The schema is `proto/runtime_package.proto`. `prost-build` generates Rust
bindings during builds with `runtime-package`; these builds require `protoc`.
Regenerate checked-in Python bindings with:

```sh
protoc --proto_path=helios/proto --python_out=ignition/preparation helios/proto/runtime_package.proto
```

Allocation references name an allocation and byte offset. Explicit relocations
support 64-bit pointers and 128-byte CUDA tensor maps. Model export preserves
recorded addresses for opaque bindings. Device allocations use CUDA VMM;
request mailboxes use registered host storage. N-gram consumers read compact
GPU buffers populated by the shared table reader and per-rank copy streams.
Address collisions fail loading.
Runtime-sized allocations map physical prefixes after fixed resources and
selected graph executables have loaded. The configured headroom, selected
concurrency and verification width, and remaining memory determine row and
page counts. Serving performs no allocation or remapping.

Supported graph nodes are kernels, empty nodes, memset operations, named
external event waits, and contiguous device-to-device copies. Other node types and device-updatable
kernels reject export. Qwen and GLM use the shared native loader and scheduler.
Packages are trusted executable input. Listener configuration and API keys are
supplied when launching. Resources remain owned until replay drains; a failed
drain retains them until process termination. Incompatible files require
re-export; there is no directory or JSON package loader.
