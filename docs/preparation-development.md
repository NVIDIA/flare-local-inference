# Preparation development

Ignition prepares weights, tunes operators, plans storage, compiles forwards,
and captures graphs before Helios serves them. Follow
[repository guidance](../AGENTS.md) and the relevant
[core preparation design](core-design.md#2-model-preparation) sections.
For running preparation, use [model preparation and serving](model-preparation.md).
Recipe and execution-component changes also require
[model development](model-development.md).

## Operator declaration, tuning, and binding

- Use b12x's autotuning and warmup facilities during preparation. Do not pin
  kernel plans or tuning choices without a demonstrated requirement.
- Constructors declare b12x requests through `ignition.preparation.tuning.declare`;
  they must not compile or prime default plans. Submit all operators across the
  complete serving shape catalog to one preparation batch, then bind selected
  plans through `register_finalizer`. Scratch sizing and serving bindings must
  follow tuning.

## Startup progress

- Show startup work through `ignition.progress.phase`. Weight loading and kernel
  preparation retain a phase activity bar while their detailed b12x panels are
  transient and share the phase's Rich console. Use `DirectWeightLoader` for
  loading progress, including draft-model weights. Only the root TP rank
  renders progress.
- Heliopause reports native export phases on stderr without the Flare banner,
  including scanned, hashed, copied, installed, and reused byte counts. Capture
  and frontend rows render only on rank zero. The enclosing Python export phase
  uses `live=False` while native rows animate and retains its elapsed result.
  Keep bulk export progress in native code so transfers do not call back into
  Python.

## Complete-forward compilation and capture

- **Compile complete prepared forwards.** Use
  [capture_forwards](../ignition/preparation/capture.py), which restores compiled
  forwards before tracing and compiles missing variants with static shapes and
  autotuning. Workers perform Dynamo tracing, storage lowering, and Inductor
  compilation from tensor and selected-plan metadata; CUDA resources remain in
  the owning rank. Preserve native mutations, storage aliases, and stream dependencies;
  bind process-local kernel handles and caller-owned tensors at startup. Complete
  b12x preparation, tuning, binding refresh, and compiled warmup before capturing
  each serving graph. Retain compiled kernel owners until native replay has drained.

## Runtime package export

`ignition.preparation.runtime_package` hands live graph handles, typed storage
and execution metadata, tensor-view roles, compiler signatures, frontend assets,
and retained owners to `flare_heliopause`. Recipes describe file-backed table
ranges and generated frontend text without creating package files or payload
offsets. Heliopause records CUPTI resources, traverses graphs and pointer-bearing
storage, downloads retained bytes, serializes resources, and owns asset writes,
hashing, encryption, signing, conversion, rank payload transfer, and publication.
Keep captured buffers, contexts, graph templates, and kernel owners alive until
native work confirms a drain; an unsuccessful drain retains their owners.

Native transfer pools bound file buffers and pinned CUDA download storage from
available host memory and preparation rank count. Preserve direct bulk I/O,
exact unaligned tails, cache verification before reuse, and hash-before-write
publication. `--no-weights` must fail on any missing exact stored-data blob;
correctness and cache checks remain required. See
[runtime package transfers](runtime-package-design.md#native-export-transfers-and-progress)
for the transfer budget and progress contract.

## Scratch storage and concurrent lifetimes

- Supply caller-owned scratch through the kernel planning interfaces. Reuse
  storage across sequential work; preserve separate storage for overlapping
  operations and persistent state. Aliasing must follow actual lifetimes and
  stream dependencies, including overlap captured inside a single graph.
- **Share scratch across the complete graph catalog.** Vision, prefill, decode,
  speculative, and resume graphs execute serially and should reuse caller-owned
  activation and private buffer pools. Allocate per-variant views with
  `GraphBufferPool` cursors; keep simultaneously live values disjoint within a
  graph, including retained verification state and overlapping expert branches.
  Use the shared operator arena for kernel workspaces and one CUDA graph pool for
  compiler-created temporaries. Inspect constructor allocations as well as captured
  allocations; sharing the CUDA graph pool alone does not share recipe scratch.
  Constants and request-owned state must retain independent storage.
  Prime indexed operators with private valid metadata: aliased serving scratch
  is initialized by the complete graph, not by each operator's constructor.
  Keep shared buffer-pool layouts identical across TP ranks, including storage
  reserved for designated-rank sampling, so shared compiled forwards retain the
  same storage aliases on every rank.

## Physical cache backing and preparation memory

- **Bound preparation cache backing by tuning and capture needs.** Keep virtual
  serving capacity separate from physical preparation backing. Back only the
  KV pages touched by tuning and capture, plus the null row, simultaneously live
  request/verification rows, and disjoint snapshot destinations. Do not physically
  allocate the planned prefix-cache capacity or fill remaining VRAM with KV during
  export; Helios sizes serving caches after graph upload. Check every fixture,
  state-clear operation, and snapshot/restore path against the backed extent.
  Describe the full virtual shapes with CUDA tensor metadata while compiling. Pass a
  `materialize` callback to `capture_forwards` to bind a bounded physical working
  set after compiler workers exit and before warmup and capture. Validate compiled
  guards against those tensors. Helios sizes the serving cache independently;
  temporary compiler contexts must not reduce its budget.
- **Release preparation temporaries at phase boundaries.** Retire loading,
  conversion, tuning, and warmup pools when their temporary owners are gone;
  retain graph pools and captured addresses through export and replay. A live
  tensor can pin otherwise unused space in a shared allocator segment; retiring
  a pool cannot release a segment that still holds a weight. Account for
  simultaneous scratch, cached allocator segments, physical cache backing,
  and graph-executable/driver allocations. Report allocated and reserved tensor
  bytes separately from device-free memory. Validate the complete requested graph
  catalog near capacity; a smaller concurrency does not establish builder fit.
- **Derive preparation memory policy from hardware.** Shared preparation code
  detects integrated GPUs and budgets compiler workers and tuning races against
  available system RAM. Recipes pass byte headroom, not hardware flags. Forward
  compilation uses memory-bounded parallel workers on both integrated and discrete
  GPUs; artifact binding starts no additional compiler pool.

## Tensor-parallel compilation

- **Compile tensor-parallel forwards once, sharded across ranks.** Pass the
  preparation coordinator to `capture_forwards`. Each forward variant compiles
  once, on a rank that selects it, from that rank's own forward; on one host, a
  rank without variants left, or without room for a compiler, compiles another
  rank's variant from that rank's traced forward metadata. Compiler pools are
  budgeted for their host's ranks. Ranks on one host share its cache directory;
  ranks on other hosts receive the artifacts they lack. Every rank loads the same
  artifacts, binds its tensors and native handles, and participates in ordered
  warmup and local capture. Keep shard offsets and native handles in runtime
  bindings when they do not change the computation. For distinct rank behavior,
  select a variant key through `compilation_variant` on each rank. Every
  selected variant must compile before collective warmup; ranks must not retrace
  or compile on a binding mismatch.

## Shape and graph-transition validation

Packed verification catalogs declare physical row capacities independently of
logical per-request widths. Use `packed_speculative_shapes` to distribute a
graph-count budget across the request buckets while preserving each bucket's
physical endpoints. Size target workspaces from the largest physical capacity
and sampling/state workspaces from their declared logical widths. Validate
capacities that are not multiples of batch size, including a single aggregate
draft row, inactive lanes, and replay transitions between physical capacities.

- **Prepare the whole serving shape range.** Include batch-one decode, packed
  decode, partial/inactive rows, and configurable chunked prefill through at
  least 4096 tokens. Test alternating graph variants with shared scratch and
  changing cache/recurrent state. Small-batch success does not establish stride,
  alias, or workspace correctness for prefill.

Follow [model development](model-development.md#speculative-decode) for width-one
speculative execution and [GPU performance work](performance-work.md) for complete
GPU-step measurements and repeated native replay checks.
