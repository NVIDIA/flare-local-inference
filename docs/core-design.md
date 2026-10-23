# Flare core design

Flare is a local inference server for NVIDIA SM120 and SM121 GPUs. It serves
agent workloads with long contexts, repeated tool-result append, shared-prefix
branches, and model-supported image inputs. Ignition prepares models in Python;
Helios serves them in Rust through captured CUDA graphs.

## Why Flare exists

Flare exists to make a local machine's memory, bandwidth, and compute go
further. It targets long conversations and responsive generation on
workstations, compact systems such as DGX Spark, and embedded deployments.
Building another inference server carries a substantial maintenance cost.
Flare must earn that cost through better use of these machines and a smaller
integration burden for the stack it serves. Its maintainers already develop
the model checkpoints, compressed scale encoding, kernels, and communication
subsystem around the same hardware and workloads. Flare extends that control
to memory ownership, request state, and scheduling while preserving the
development speed of the Python ML ecosystem.

### Keep the ecosystem's development speed

A wholesale “vLLM in Rust” rewrite is a poor trade when it severs access to
the Python model and kernel ecosystem. It takes on a much larger obligation
than building a scheduler. It must continually reproduce model architectures,
checkpoint conventions, quantization rules, multimodal processors, compiler
integrations, and numerical fixes. A working implementation of one model is
the beginning of that maintenance burden.

Python's value here is the accumulated work available through it. PyTorch,
Transformers, Triton, CuTe DSL, FlashInfer, and the surrounding tools provide
model definitions, compilers, optimized operators, loaders, references, and
ways to inspect failures. Much of the difficult integration work happens at
the boundaries between them. A separate implementation inherits responsibility
for those boundaries every time an upstream contract changes. Each upstream
improvement it needs becomes another port to schedule and verify. A small
project would be maintaining its own counterparts to work contributed across
the industry. That recurring cost consumes the time that could have improved
inference.

Flare makes Transformers and PyTorch compilation the foundation of model
preparation. Ignition preserves canonical model implementations where their
dataflow fits and replaces the smallest coherent unit where serving requires
a different layout or state contract. It uses `torch.compile` on complete
prepared forwards before warmup and CUDA graph capture. Model structure,
numerical semantics, compiler optimizations, and kernel preparation remain
connected to the Python ecosystem throughout execution construction.

vLLM offers a
[Transformers backend](https://huggingface.co/docs/transformers/main/en/transformers_as_backend),
whose attention, state, and execution contracts still need to fit each model.
Flare's preparation boundary lets canonical model code and PyTorch compilation
produce the execution that its native runtime serves. Access to upstream work
is a lasting development advantage; rebuilding a parallel model ecosystem
would spend that advantage before the serving loop runs.

Language choice alone also says little about GPU execution. A Python-defined
model can launch compiled kernels and replay captured graphs; vLLM itself
[uses CUDA graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/).
Translating host code does not reduce the weights read by a GEMM, fuse an
attention kernel, or repair an inefficient layout. Host overhead matters where
measurement exposes it. The GPU program, its memory traffic, and its
dependencies still determine the cost of the model step.

### Complete a stack with a common target

Flare builds on quantized QAD model checkpoints, losslessly compressed FP4
scales in the CSF format, and the b12x compute and communication library in
FlashInfer. These components share a focus on local inference and SM120/SM121.
Their maintainers can change the checkpoint representation, kernel contract,
communication plan, and serving integration together. The remaining work is
concentrated on composing those capabilities into a service; Flare does not
need to establish a competing model ecosystem or develop another kernel
library to become useful.

vLLM's device, backend, model, and distributed-execution breadth creates a
large compatibility matrix. Changes to cache layouts, compilation, offload,
and scheduling must coexist across those combinations. Flare limits that
obligation to selected model recipes, FlashInfer/b12x, NVIDIA SM120/SM121,
and supported single-host and cross-node tensor-parallel configurations.
It can use one prepared-execution contract throughout the
runtime because backend portability and general vLLM feature parity are
outside its scope.

Several benefits depend on cooperation across these interfaces. CSF saves
resident memory when scales remain compressed between uses and serialized
layers reuse expansion scratch. A fast loader preserves that saving only if
its staging and conversions stay bounded. A prepared kernel retains its
allocation advantage only if tuning operands are released and graph variants
share storage where their lifetimes permit it. A PCIe collective needs the
right communication buffers, launch dependencies, and ownership through graph
replay. Each local optimization places obligations on the surrounding service.

Flare makes these requirements part of the runtime contract. Prepared
execution, caller-owned storage, complete continuation state, and explicit
capacity limits apply throughout the service. Its narrow scope reduces the
combinations that those rules must accommodate. Owning the relevant interfaces
also makes coordinated changes practical: a kernel's workspace or state
contract can be carried through preparation and native replay by the same
maintainers, with correctness and performance checked through the service.

Flare also provides a complete application for the shared library. Repeated
graph replay, mixed request shapes, cancellation, prefix reuse, and memory
pressure exercise combinations that isolated kernel benchmarks cannot cover.
Reusable fixes and optimizations belong in FlashInfer/b12x, where other
engines can use them. Flare's maintenance cost must be justified by the serving
behavior it adds around that common foundation.

### Put the native boundary after preparation

Flare keeps Python where model work benefits from that ecosystem. Ignition
loads checkpoints, composes forwards, tunes kernels, plans scratch, compiles,
and captures the supported executions. It exports their kernels, graphs,
storage layouts, and frontend assets as a runtime package. Python can do
substantial work to construct that package because it finishes before serving.

Helios then owns a different problem: admit requests into bounded resources,
maintain complete continuation state, submit prepared work, and deliver
results. Rust provides explicit ownership and concurrency for sockets, queues,
cache pages, graph handles, transfers, and shutdown. The serving process needs
neither a Python interpreter nor Python callbacks. A kernel integrated through
Python remains usable after capture without acquiring a second Rust model
implementation or a Rust binding for each operator's preparation API.

This boundary carries real obligations. Every serving path must be prepared;
state, aliases, and resource lifetimes must be explicit; unsupported capacities
must fail admission. Model-specific native packing and frontend behavior still
need implementation and testing. The work stays concentrated on the serving
contract while model construction and kernel development remain connected to
their existing tools and maintainers.

### Give cache operations one ownership model

Decode context parallelism (DCP) distributes an attention history across ranks.
Correct execution requires agreement on token placement, page geometry,
global selection, collective ordering, and scratch lifetimes. Efficient
prefill and decode can require different exchanges. Flare's
[GLM context layout](../ignition/recipes/glm53/dcp.py) declares the rank
placement and page geometry; its prepared attention uses bounded query
exchange for decode and gathered context for prefill. The runtime package
carries the resulting storage and execution contract into Helios.

Keeping a conversation outside GPU memory imposes related requirements. A
resumable endpoint includes recurrent matrices, selector state, attention
tails, and saved hidden state alongside its KV pages. Every part must describe
the same committed token prefix. Transfer sources must remain immutable, and
destinations cannot be recycled while a copy can still write to them. Offload
integration spans frozen recurrent snapshots, padded selector tails, restore
admission, cancellation, and cache reset. A disk transport depends on all of them.

The [tiered prefix-cache design](prefix-cache-offload.md) extends the checkpoint
object Helios already owns. Stores retain published immutable page chains and
their complete endpoint rows. The runtime package declares every rank's byte
layout, so storage needs no model-specific interpretation. Memory and disk
stores use the same lookup and completion interface; staged I/O and direct
storage transfers preserve the same checkpoint semantics.

Restore resources belong to a scheduler-owned ticket. Requests sharing a
prefix can join one transfer, and cancelling a request detaches its waiter
without invalidating that transfer's buffers. Completion publishes an ordinary
GPU checkpoint before any request can execute against it. Disk retention
therefore changes where checkpoint bytes live while preserving the meaning of
publication, reuse, and retirement.

Flare can concentrate correctness in these shared rules. Context sharding,
branching, speculative rollback, and offload all operate on explicitly owned
state. Their interactions remain necessary engineering work, but a common
contract reduces the independently maintained interpretations of a page, a
completed copy, or a resumable endpoint. This is a concrete reason to own the
serving architecture alongside the kernels.

### Treat memory topology as part of the target

Embedded deployments and unified-memory systems make wasted capacity and
unnecessary traffic especially costly. On DGX Spark, CPU and GPU allocations
draw from shared system DRAM. A host copy of a GPU cache therefore consumes
the same physical capacity; calling it an offload tier creates no additional
memory. NVIDIA's
[DGX Spark memory guidance](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html#guidance-for-reporting-memory-resources-with-unified-memory-architecture)
also explains why device free-memory reporting alone does not describe all
reclaimable system memory on this platform.

Sharing physical memory does not make allocation types interchangeable.
CPU accessibility, GPU access, registration, and I/O support depend on the
platform and allocation mechanism. NVIDIA's
[Spark CUDA porting guidance](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)
documents constraints that differ from discrete GPUs. A generic policy that
treats host RAM and device memory as independent budgets, or assumes every
shared allocation supports the same access path, cannot account for those
constraints.

Flare's efficiency goal therefore includes the whole memory lifecycle. Bound
checkpoint-loading intermediates, reuse scratch according to actual lifetimes,
release compiler resources before allocating large caches, and avoid retaining
duplicate representations of model state. Account for host buffers, graph
storage, weights, continuation rows, KV pages, and operating-system headroom
against their physical backing. On systems using managed memory, include
residency and migration behavior in that accounting. Memory that can be made
allocatable through reclamation is not automatically suitable for predictable
token latency.

Exporting a runtime package lets compiler and model-construction resources
disappear before the native service starts. A narrow hardware target makes
the remaining decisions specific enough to measure. SM121 acceptance must
cover preparation and loading peaks, resident serving memory, memory traffic,
and latency under pressure. Discrete-GPU measurements cannot establish those
properties on Spark. The goal is useful context and completed work per
available byte and unit of bandwidth, with predictable response times.

### Make a smaller promise than vLLM

[vLLM's broad model, hardware, and distributed-serving support](https://docs.vllm.ai/en/latest/design/arch_overview/)
is valuable and expensive to reproduce. Flare chooses NVIDIA SM120/SM121,
FlashInfer/b12x, selected model recipes, and interactive workloads on one host
or supported cross-node tensor-parallel groups. [Multi-host serving](multi-host.md)
uses prepared RoCEnante collectives; [Orbit](orbit.md) manages the hosts,
preparation, packages, and rank processes needed for those deployments.
Interchangeable backends and general-purpose cluster orchestration remain
outside that serving scope. Flare does not aim for vLLM's feature matrix.

Within this scope, the priorities are concrete. A tool result should append
to an existing conversation without recomputing its history. Branches should
share immutable prefixes while preserving recurrent state. A long prefill
should leave room for responsive generation. Speculation should earn its
state and bandwidth cost through accepted tokens. The device's memory should
hold useful weights and history after accounting for the resources required
to execute them.

The vLLM fork provides a direct reference for this decision. Compare the same
checkpoint, numerical contract, kernel library, device, context lengths, and
concurrency. Measure completed work, response latency, physical memory use,
and the cost of preserving conversation state. Count the ongoing effort to
support model and kernel changes as well. A kernel improvement available to
both engines cannot by itself justify a separate server. If a small integration
delivers the same behavior with less maintenance, extending the existing engine
is the better choice.

Flare succeeds when this bounded serving contract makes the shared stack more
efficient to run and maintain for its intended workloads. Reusing the
industry's model tools and the same public kernel library keeps the investment
focused on that result. A useful local server needs to make those deployments
better; it does not need to replace vLLM across its supported applications.

## Architecture and principles

The design has three ownership boundaries:

| Component | Responsibility |
|---|---|
| Ignition, in `ignition/` | Checkpoint loading, model composition, kernel tuning, memory planning, forward compilation, graph capture, and package export |
| Heliopause, in `heliopause/` | CPython extension providing the preparation allocator, CUDA resource inspection, and native replay diagnostics |
| Flare and Helios, in `flare/` and `helios/` | Package loading, native frontend, request scheduling, cache ownership, graph replay, and output delivery |

Both Flare and Heliopause depend on Helios. Helios has no Python dependency.
The [runtime package design](runtime-package-design.md) specifies the
`.safetensors` boundary. The [CLI design](cli-design.md) defines configuration
and capacity selection; the [MTP design](mtp-design.md) defines speculative
state transactions; the [tensor-parallel design](same-process-tp.md) defines
rank ownership and coordination.

### Design principles

Helios's shared runtime policy and separation from model computation are
non-negotiable architectural invariants. Correct output, performance gains,
delivery pressure, and intended cleanup do not justify model-specific runtime
policy or native model computation. Temporary implementations must obey the
same boundary; violations must be resolved before committing.

- **Helios stays minimal.** Model-specific execution behavior, composition,
  state-update computation, and optimization belong in Ignition and its captured
  programs. Native specialization is limited to concrete cache storage
  requirements and graph cache bindings. Cache schemes and prepared layout
  metadata express those differences through contracts reusable by every model
  with the same requirements. Introducing a captured-graph operation family requires
  prior explicit approval from project owner Luke Alonso <lalonso@gmail.com>.
  Shape and capacity variants preserve an existing family's operation contract;
  a new operation contract requires approval even when it reuses an existing
  graph kind or enum value.
- **FlashInfer supplies GPU execution.** Preparation uses FlashInfer and its
  b12x operators directly. Flare has no interchangeable kernel backend.
- **Graphs define model execution.** Preparation resolves kernel calls,
  arguments, storage, and stream dependencies before serving. Helios uploads
  request metadata and launches graph executables without Python callbacks.
- **State has explicit storage and ownership.** KV pages, selector keys,
  recurrent matrices, convolution and n-gram histories, positions, and saved
  hidden vectors belong to declared page or slot layouts. Checkpointing,
  branching, and rollback preserve every state-bearing mechanism.
- **Admission reserves capacity.** Requests outside the prepared feature or
  capacity contract fail with a typed error. Steady-state model execution
  allocates no GPU storage, compiles no kernels, and captures no graphs.
- **Memory planning follows the device.** Account for physical capacity,
  residency, access constraints, and traffic across preparation, loading,
  and serving. Shared CPU/GPU memory is one budget; duplicate cache tiers
  must justify their capacity cost.
- **Recipes own model semantics.** Preserve canonical Transformers forwards
  where their layouts and state handling fit the serving contract. Replace
  the smallest coherent module needed to control dataflow. Reusable
  projections, experts, collectives, and state transfers belong in
  `ignition/components/`.
- **One native lifecycle serves every model.** Transport, scheduling, cache
  ownership, input packing, frontend processing, and request handling stay shared.
  Admission, page reservation and growth, eviction, preemption, readmission, and
  offload use shared policy and mechanisms. Model or recipe identity must not
  select those behaviors, including through capability flags that encode model
  allowlists. Cache schemes supply concrete page geometry, state storage, and
  restore constraints; shared policy consumes those contracts. Express other
  model-dependent behavior through prepared metadata and frontend assets.
- **Runtime fixes follow the affected contract.** Trace a defect through every
  supported cache scheme and apply the fix to every compatible contract. For
  example, incremental page reservation depends on cache growth and continuation
  requirements. A GLM reproducer does not justify a GLM-only scheduler path.
  Restrictions require concrete storage or execution constraints in the contract.
  Validate affected cache schemes and request lifecycles; untested models remain
  a validation limitation, not a reason for model-specific policy.
- **Prepared contracts cannot select runtime policy by model identity.**
  `PreparedContract` describes graph value sources, storage views, dependencies,
  capacities, and frontend syntax. Per-model boolean policy flags, including
  `supports_*`, `can_*`, and default-false opt-ins, are prohibited. Enums,
  callbacks, schema names, and metadata fields must not disguise the same
  allowlist. Shared code derives behavior from concrete geometry and dataflow.
  Independent enum identifiers select reusable tool/reasoning parsers and image
  preparation/layout implementations only. Review every field and all consumers.
- **Protocol Buffer changes preserve model package compatibility.** Every Protocol
  Buffer change **must** be accompanied by Helios code that keeps previously
  exported model packages loadable and usable without re-export. Shared code must
  handle absent fields and Protobuf default values while preserving older metadata
  semantics and explicit values. Wire-format compatibility alone is insufficient.
  Include regression tests for older serialized metadata and validate the affected
  package loading and serving paths across the affected model contracts.
- **Numerical and performance requirements are separate.** State copies and
  metadata are exact. Arithmetic comparisons follow each operator's precision
  and reduction semantics. Complete graph and service measurements determine
  performance.
- **Recipes declare model-card sampling defaults.** Agents must read the model
  card and manually verify recommendations for the default serving mode. Checkpoint
  config files remain authoritative for structural data, not sampling policy.
  Ignition declares complete defaults and disables unspecified filters; explicit
  request values override defaults. Runtime metadata contains typed sampling,
  EOS, tokenizer, and image settings, with no raw configuration-file assets.

Serving includes cancellation, streaming, tool-call presentation, structured
output, prefix reuse, and speculative decoding on one host or supported
cross-node tensor-parallel groups. Rank coordination and transport preserve
the same prepared-graph execution and shared runtime policy boundaries.
Disaggregated prefill, adapters, pooling models, and device portability are
outside this design.

## 1. Target and execution contract

Recipes declare the data consumed by the shared native contract:

| Recipe | Contract declaration | State and execution |
|---|---|---|
| [Qwen3.8 Flash](../ignition/recipes/qwen38/recipe.py) | [Prepared data](../ignition/recipes/qwen38/contract.py) | GDN recurrence, QSA sparse attention, PLE, vision, and learned or copied proposals |
| [GLM 5.3 Flash](../ignition/recipes/glm53/recipe.py) | [Prepared data](../ignition/recipes/glm53/contract.py) | KDA recurrence, indexed MLA, mHC, vision, and learned MTP |
| [DeepSeek V4.1 Flash](../ignition/recipes/deepseek41/recipe.py) | [Prepared data](../ignition/recipes/deepseek41/contract.py) | Compressed latent attention, sliding windows, Engram tables, and DSpark proposal composition |
| [DeepSeek V4 Flash](../ignition/recipes/deepseek4/recipe.py) | [Prepared data](../ignition/recipes/deepseek4/contract.py) | Compressed latent attention, sliding windows, vision, and DSpark proposals |
| [MiMo V2.6 Pro](../ignition/recipes/mimo_v2/recipe.py) | [Prepared data](../ignition/recipes/mimo_v2/contract.py) | Paged KV, sliding windows, vision, and DFlash proposals |

Recipes declare supported GPU counts, checkpoint revisions, and feature
capacities. Each architecture and tensor-parallel size requires independent
model and replay validation.

The Qwen geometry used throughout this document comes from
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`, revision
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`, on one 96 GB RTX PRO 6000
Blackwell. It contains 36 Gated DeltaNet layers, 12 sparse-attention layers,
512 routed experts per decoder layer, four hyperconnection streams, PLE
n-gram embeddings, and a vision encoder. Stored formats include NVFP4
experts with compressed scales, MXFP8 projections, and mixed-precision
vision weights.

[pyproject.toml](../pyproject.toml) and [uv.lock](../uv.lock) declare the
preparation dependencies. Transformers supplies canonical architecture and
processor semantics. FlashInfer targets `main` in
`local-inference-lab/flashinfer`, with its resolved commit recorded in `uv.lock`;
its b12x operators and FlashInfer sampling supply the prepared GPU computation.

### Runtime contract

- Every admitted model execution runs through prepared CUDA graphs.
- Preparation resolves kernels, allocates storage, warms execution paths, and
  captures every supported graph variant.
- Ignition performs bootstrap, model loading, preparation, and capture, then
  exports a runtime package and closes the prepared model. `flare run PACKAGE`
  loads it in a separate Rust serving process.
- Rust owns all runtime sockets, HTTP/SSE, queues, tokenization, chat-template
  rendering, image preprocessing, grammar advancement, sampling control,
  output decoding, cancellation, and health reporting.
- Rust never invokes Python to process a request, execute a model step, prepare
  attention metadata, update a cache, or sample a token. The serving process
  contains no Python interpreter or Python resource owners.
- Unsupported requests fail admission with a concrete capacity or feature error.
- Graph executable handles, their captured storage, and their CUDA context remain
  alive together.
- Steady-state serving performs no CUDA allocations, kernel compilation, or graph
  capture.
- Identical inputs to deterministic plans produce identical logits. Plans with
  atomic or otherwise variable reduction order use component tolerances derived
  from measured repeatability. State transfers and request metadata remain exact.
- A typed `PreparedContract` defines semantic value bindings, state views,
  graph dependencies, completion cursor relationships, and frontend syntax.
  Shared admission and packing code binds these to each rank's buffer geometry.

### Capacities and memory

For Qwen at concurrency four:

| Setting | Default |
|---|---|
| Decode request capacities | 1, 2, 4 |
| Prefill scheduling | One request per chunk, interleaved with decode |
| Maximum prefill chunk | 4,096 tokens |
| Prefill capacities | 128, 256, 512, 1,024, 2,048, 4,096 tokens |
| Tail placement | Smallest captured capacity containing the live tail |
| Context limit | 262,144 tokens |
| Main KV / GDN recurrence | BF16 / FP32 |
| PLE delivery | Compact GPU rows supplied by Helios from disk or RAM |
| Checkpoint endpoints | Rendered message endpoints, final prompt, and terminal consumed prefix |
| MTP draft limit | Five drafts plus the pending input |
| Speculative scheduling | One round in flight |
| Vision | Prepared image capacities; video validation remains separate |

Ignition's `--prefill-chunk-size` selects a multiple of 64 between 64 and the
context limit. Capacities are powers of two starting at 128 below that limit,
plus the exact limit; a limit of 64 captures one 64-token graph. Preparation
validates every operator geometry and workspace envelope. An unsupported
capacity fails instead of silently shrinking the chunk or omitting a variant.

At the default Qwen capacities, ordinary serving has 25 graphs. Five-draft
speculation replaces three decode graphs with eighteen verification graphs,
giving 40 graphs. Prefill, resume, vision, initialization, snapshot, and
restore share the same catalog. `--no-mtp` omits draft weights and speculative
state during preparation. Load-time `--mtp-draft-tokens 0` selects target-only
verification while retaining the package's MTP weights and history layout.

Flare can reduce concurrency and draft depth within the package's captured
capacities. Fixed verification loads one width for each required batch;
adaptive verification loads the permitted widths. Concurrency three requires
the batch-four bucket. Missing variants and capacity increases fail before
CUDA initialization.

Continuation rows and KV pages occupy separate fixed virtual ranges.
Helios backs those ranges after uploading selected graphs, retaining the
configured GPU reserve and a 64 MiB driver margin. The memory planner accounts
for null and live rows, verification width, admission checkpoints, and the
prefix cache before assigning the remaining physical memory to KV pages.
Every rank uses the same row and page counts, limited by the smallest rank
budget. Weights and fixed transport buffers retain their prepared layouts.

The context limit bounds one request; aggregate page capacity bounds all
active and cached histories. Cache management follows attention storage:

| Attention mechanism | Growing history | Continuation state |
|---|---|---|
| Causal full or sparse attention | Append logical KV or latent/index pages | Partial page and saved hidden state |
| Compressed latent attention | Append pages at the declared compression geometry | Compression interval, overlap, partial pages, and sliding window |
| Recurrent linear attention | No token-proportional pages for the recurrence | Recurrent matrices and convolution history |
| Sliding-window attention | Fixed window in a continuation row | Window contents and position |

A hybrid model combines these storage requirements. Every prepared model uses
the same page-growth policy; page size, draft-page headroom, and continuation-row
extent describe its storage. Model names and per-model support flags do not
participate in the decision.

For text requests, admission reserves the prompt and one page of output. The
output limit bounds generation without reserving all of its potential history.
Reservations grow a page ahead of decode rounds, evicting unpinned prefix pages
first; admission waits while pages are short. When a running request still cannot
continue, the newest eligible decoder publishes its terminal checkpoint when
backing allows and is readmitted with its history and remaining output budget.
The checkpoint restores all declared continuation state; eviction can instead
require recomputing the history. Shared handling extends text-position and
image-index metadata for generated tokens before readmission.

The output stream, stop and grammar state, and penalty counts continue. Random
streams restart on the device with a seed derived from the request's seed and
the number of tokens generated before preemption. A request whose history
exceeds the whole pool ends at that length. Requests with image inputs or
conditioning retain full reservations because readmission does not reconstruct
their image-feature execution. They wait in the bounded admission queue until
their full reservation fits. Waiting for initial capacity does not require
preemption support. A blocked reservation does not prevent smaller queued
requests from using available resources. Requests whose required reservation
exceeds the entire pool fail admission.

For Qwen's target cache, BF16 KV and compressed selector keys cost 24.75 KiB
per token; MTP pages, continuation rows, and graph resources add separate
costs. The loader reports actual capacities. The [CLI design](cli-design.md)
and [package memory layout](runtime-package-design.md#4-rebuild-and-serving)
specify sizing and load-time selection.

## 2. Model preparation

### Model composition and shared components

Instantiate the canonical Transformers model on the meta device, retain its
module tree for weight naming and configuration, and stream checkpoint tensors
into their execution layouts. Qwen replaces the text model, decoder layer,
sparse attention and indexer, GDN, PLE, and vision forwards with prepared
operations over packed inputs, fixed storage, positions, and cache metadata.

Those replacement boundaries control inline hyperconnection mixing,
host-driven sparse selection, PLE row delivery, growing position histories,
and per-image loops. Other recipes choose boundaries from their own forward
dataflow. Canonical Python dispatch during capture creates no per-token
overhead during native replay.

Recipes control checkpoint mapping, composition, state layouts, memory policy,
and graph construction. Shared components expose independent execution
mechanisms:

| Mechanism | Component and contract |
|---|---|
| Dense projections | [linear.py](../ignition/components/linear.py): BF16, FP32, and block-scaled plans, declared tuning, caller-owned outputs, and scratch |
| Vocabulary projection | [vocabulary.py](../ignition/components/vocabulary.py): contiguous TP weight shards, rank-local projection, and gathered logits |
| Embeddings | [embedding.py](../ignition/components/embedding.py): counted lookup with inactive-row zeroing |
| Routed experts | [moe.py](../ignition/components/moe.py): FP32 route weights, model-selected scoring, and tuned expert execution |
| Branch overlap | [expert_overlap.py](../ignition/components/expert_overlap.py) and [projection_overlap.py](../ignition/components/projection_overlap.py): explicit forks, joins, and concurrent scratch |
| Recurrence and mixing | [recurrent.py](../ignition/components/recurrent.py) and [mhc.py](../ignition/components/mhc.py): state bindings, checkpoint export, and fused mixing |
| Sampling | [sampling.py](../ignition/components/sampling.py): request epochs, Philox state, penalties, grammar masks, and completion records |
| Token consumption | [token_state.py](../ignition/components/token_state.py): pending tokens, positions, epoch validation, and committed spans |
| Continuation storage | [state.py](../ignition/components/state.py) and [state_transfer.py](../ignition/components/state_transfer.py): aligned rows, exact endpoint transfers, and alias checks |
| Sparse attention | [sparse_mla.py](../ignition/components/sparse_mla.py) and [indexer.py](../ignition/components/indexer.py): latent writes, selection metadata, and prepared attention |
| Collectives | [collectives.py](../ignition/components/collectives.py): captured b12x reductions with retained transport storage |

Model contracts use the shared [graph catalog](../helios/src/model/catalog.rs)
to validate capacities, operations, and bindings. Shared
[admission](../helios/src/model/admission.rs) owns request epochs, sampling
parameters, prompt tokens, budgets, and grammar masks. Qwen adds rotary
positions and vision-feature indices. GLM and DeepSeek supply their own
prompt and state metadata. Adapters participate in one scheduling and
completion lifecycle.

### Tensor-parallel model composition

GLM shards KDA and MLA heads, dense/shared MLP intermediates, routed-expert
intermediates, and vocabulary weights. Low-rank inputs, sparse indexers,
normalization, and mHC weights are replicated. Packed Q/K/V, beta, and
forget-gate input projections share their source; an independent output-gate
chain can overlap them. Multirow column slices use caller-owned contiguous
staging where the downstream GEMM requires it.

GLM's decoder composes mHC, KDA or sparse MLA, an attention all-reduce,
fused post/pre mixing, dense or routed experts, and a second all-reduce.
BF16 router weights produce FP32 logits. Routing uses sigmoid scores,
correction-biased top-eight selection, and weights scaled by 2.5. The
vocabulary projection runs on every rank, including resume sampling.

GLM's vision tower is replicated across ranks. Native image preparation matches
the checkpoint processor's aligned resize, black padding, normalization, and
grouped 14-pixel patches. Captured vision graphs merge four patches into each
4096-element feature and publish generation-scoped embeddings through the shared
image feature arena. Text prefill validates and injects those features before
the decoder layers. Vision and prefill reuse activation storage because their
graphs execute serially; index and router constants retain separate storage.

Each GLM continuation row retains KDA recurrence and convolution history,
unfinished C4 index groups, partial latent/index pages, consumed count, and
the final normalized hidden vector. Its logical pages cover 256 tokens.
Maintenance validates endpoints, epochs, aliases, and rank participation
before any persistent write; a rejection on one rank suppresses the command
on all ranks. Sparse selection order and atomic reductions require numerical
comparisons on matched operands and measured repeatability. GLM full-model
numerical acceptance remains incomplete for these paths.

DeepSeek's logical pages cover 128 tokens across compressed latent and index
regions with different byte sizes. Continuation rows retain sliding windows,
pending compression state, partial pages, and saved hidden state. Engram
lookup uses committed request history, including tokens covered by a prefix
hit. Its recipe composes DSpark drafting separately from target attention and
state maintenance. Each prepared variant requires the same native replay,
rollback, and numerical checks as the Qwen and GLM variants.

Preparation uses one Python process per rank and a Gloo coordinator for
declarations, readiness, compilation distribution, and ordered warmup.
Serving loads all rank bundles into one Rust process with one execution
thread per rank. The [rank layout](same-process-tp.md) specifies symmetric
preparation addresses, peer access, collective lifetime, and failure handling.

### Qwen operators

| Model region | Prepared execution |
|---|---|
| Quantized linear projections | b12x MXFP8 and NVFP4 operators |
| Routed experts | b12x fused MoE with prepared CSF weights |
| GDN recurrence | b12x GDN prefill and decode |
| GDN causal convolution | b12x short convolution over raw projected QKV and explicit history |
| Sparse attention and selector state | b12x QSA |
| Hyperconnections | b12x hyperconnection primitives and BF16 down projections; tuned up-projection, sigmoid, and stream-mean operator |
| PLE lookup | b12x NVFP4 PLE embedding over compact GPU rows delivered by Helios; explicit n-gram history |
| PLE mixing and convolution | b12x PLE operators |
| Vision | Transformers block structure; b12x varlen attention, MXFP8 attention projections, weight-only NVFP4 MLPs through the block-scaled `w4a16` entry |
| Vocabulary projection | b12x BF16 vocabulary projection |
| KV writes | Independent b12x paged KV-write entry supporting 256/256 head dimensions |
| Request metadata, state maintenance, sampling | Captured server operations |

Ordinary tensor composition may remain expressed in PyTorch during capture.
Decoder hyperconnections fuse residual injection with the following mixer's
RMSNorm using FP32 gains, an in-place BF16 residual, and caller-owned normalized
scratch. The PLE injection boundary requires a separate normalization. MTP
fuses the attention-to-MLP boundary; its inactive-row mask precedes the final
standalone normalization.

The up-projection and stream gate are tuned as one operation at each prepared
row capacity. b12x selects a fused kernel or an existing BF16 projection followed
by gate reduction. Both preserve BF16 rounding after the projection, sigmoid,
and product, with FP32 stream accumulation. Selection finishes before forward
compilation and graph capture; Helios replays the selected CUDA work. Gate
workspace is volatile scratch, so stage traces expose normalized inputs,
bottlenecks, injection logits, and mixed outputs without requiring materialized
gate logits.

GDN combines QKV/Z and B/A weights into two MXFP8 projections. Sparse attention
combines Q/K/V into one projection and keeps the indexer independent. MTP uses
the same Q/K/V grouping with BF16 weights. Loading concatenates serialized
values and scale rows before b12x packs the scales; it does not requantize
weights. Output slices share persistent packed scratch. The b12x convolution,
recurrence, and KV writer consume row-strided views without unpacking copies.
GDN and sparse-attention projection plans let b12x select BF16 or quantized
activations for each prepared capacity. Concurrent projections retain separate
workspaces sized from their selected plans.

Operators that allocate their outputs per call, including standalone
hyperconnection combine, are acceptable inside capture
because the allocation resolves into the graph memory pool. Graph variants share
one pool only for intermediates whose lifetime ends within their graph. Values
that survive a graph boundary live in caller-owned persistent storage outside
the graph-private pool: request and cache state, snapshots, saved hidden
vectors, vision embeddings, and completion frames. Each graph copies escaping
results into that storage before completing. Graphs execute serially on the
model stream, including any captured side-stream joins; serialization alone
does not establish safe pooled-output lifetimes. Qualification alternates
decode, prefill, maintenance, and vision variants in different orders.

### Checkpoint loading

Use the exact checkpoint's safetensors index and quantization configuration.

The loader must:

- Account for every required tensor through an explicit recipe mapping.
- Preserve packed NVFP4 values, global/input scales, MXFP8 scale interpretation,
  and PLE hashing constants.
- Feed compressed expert scales into b12x's CSF preparation path.
- Share CSF expansion scratch across serialized layer execution.
- Convert normalization weights only where the selected b12x contract requires
  it, and qualify the numerical effect.
- Exclude MTP tensors explicitly while MTP is disabled.
- Release loading intermediates incrementally.
- Append immutable PLE table slices to the runtime package without
  materializing the full table in preparation memory. Graphs consume compact
  GPU row buffers; Helios supplies their contents at replay.

Avoid materializing a second full model or a fully dequantized expert table.

NVFP4 compressed scales load directly from checkpoint tensors into CUDA storage
through b12x's direct loader. b12x performs TP slicing, gate/up joining, native
layout conversion, and exception indexing on the GPU. Temporary compressed
source storage is released after each layer is prepared. W4A16 scale repacking
also remains on the GPU. No prepared-scale disk cache is used.

The local snapshot's safetensors headers give this storage breakdown:

| Component | Stored GiB | Storage role |
|---|---:|---|
| Decoder routed experts | 59.94 | GPU |
| Other decoder tensors | 6.57 | GPU |
| Vision tensors | 0.30 | GPU |
| PLE tables | 26.82 | Package payload; disk or resident RAM at serving |
| MTP tensors | 1.49 | GPU when learned MTP is prepared |
| **Total checkpoint** | **95.12** | |
| **GPU-bound stored tensors without MTP** | **66.82** | |

These are stored tensor sizes. Prepared layouts, scratch, activations, graphs,
and caches require additional memory.

Preparation reports memory for persistent weights, table-delivery buffers,
operator workspaces, the shared graph pool, live and speculative state,
checkpoint snapshots, and pages.
The Qwen preparation API defaults to 4 GiB of GPU headroom and a 4 GiB
checkpoint-state budget. Ignition passes the [CLI memory plan](cli-design.md):
pages sized from measured free memory, a snapshot count derived from
concurrency, and headroom for capture plus the configured reserve. Helios
independently sizes physical rows and pages at load time. Preparation fails
if the requested capacities cannot fit.

### Gated DeltaNet convolution

b12x's GDN packages treat convolution as external. The prepared
`sequence.short_conv` operator comes from the LIL FlashInfer dependency.
The PLE implementation in `sequence/ple` supplies
reusable history and state machinery: caller-owned state sized by dilation and
kernel width, state-slot ids, a fresh-state flag, accepted-token counts,
device-resident sequence and token counts, and a mixed prefill-plus-decode mode.

The independent prepared short-convolution operator shares that machinery.
GDN consumes raw projected QKV, stores its raw input history, and uses dilation
1 with SiLU over the projection width. The complete PLE operator also performs
gating, normalization, and residual addition; those operations do not belong in
GDN. Preserve PLE's own input-history and residual semantics when sharing the
convolution machinery. Qualify convolution and activation rounding against the
GDN references. Compare the prepared operator against the PLE reference and measure its
contribution to complete graph latency.

The prepared interface supports:

- Packed prefill and single-token decode.
- Caller-owned convolution state.
- Explicit request-to-state-slot mapping.
- Fixed token and sequence capacities.
- Inactive lanes that perform no state writes.
- Capture-safe execution without CPU sequence-length extraction.

The [GDN component tests](../tests/recipes/test_qwen38_gdn.py) compare exact
checkpoint execution with the canonical Transformers implementation. The
serving recipe consumes the prepared b12x operator directly.

### KV writes

Flare owns the cache-write contract: write projected K/V to the page and offset
identified by request metadata, then execute attention. QSA does not write the
main cache; writes precede its call.

b12x's independent `attention.paged_kv_write` entry prepares the
capture-safe writer for Qwen's BF16 256-dimensional K/V heads. Its
`[pages, page_size, kv_heads, head_dim]` layout matches the QSA main cache.
Negative slot mappings suppress writes.
Slot mappings are contiguous int64 values `page * page_size + offset`; live
destinations must be unique and padded input rows must have valid backing
storage even when their negative mappings suppress writes. The writer is
captured inside the forward graph. Rust assigns pages; the captured GPU program
writes them.

### Numerical policy and tuning

QSA selection, KV writes, and attention preserve causal visibility and the
checkpoint's normalization and rotary semantics. Different attention
partitions can change BF16 probability rounding. Compare partitions on
identical operands, including cache contents and selected indices, and
calibrate cross-capacity bounds from measured arithmetic differences.

GDN retains BF16 intermediates and FP32 recurrent state. Its 16-token prefill
chunks are relative to each invocation, so changing a prefill partition can
change recurrence association even with sequential single-key-split execution.
Any consumed endpoint remains a valid state-transfer boundary. Snapshot and
restore preserve that endpoint exactly; comparisons against differently
partitioned prefill use numerical bounds.

Autotuning should select every kernel configuration compatible with the
model's arithmetic contract. Cross-capacity bitwise equality must not force
a slow configuration. Same-capacity replay is exact for deterministic plans;
atomic reductions use measured repeatability bounds. Whole-model logit
distance is diagnostic and cannot select a recurrence policy.

The GDN declaration in [gated_delta.py](../ignition/components/gated_delta.py) restricts
prefill to `recurrence_policy="sequential_k1"`. Removing that restriction
requires component and prefix-continuation checks, plus before/after
prefill throughput and kernel timelines. QSA's declaration uses the library's
default partition policy; [tuning.py](../ignition/recipes/qwen38/tuning.py)
still labels it as fixed at sixteen splits. Preparation policy metadata must
describe the submitted queries. Controlled arithmetic comparisons belong in
explicit preparation configuration, carried into the plan and measurements;
environment-variable switches are prohibited.

### Preparation lifecycle

`RecordedPreparationSession.declare()` collects requests without compiling,
priming, or binding kernels. Components use `ignition.preparation.tuning.declare()`
to submit requests and `register_finalizer()` to defer serving bindings. After
every admitted capacity has an owner, bootstrap prepares one b12x batch containing
weight operations and operations that depend on cache or recurrent-state
layout. The batch enables autotuning and coalesces equivalent declarations
across layers, target and draft models, and graph capacities. Compiler
concurrency is derived from the host; library environment settings are applied
by Ignition from its configuration before imports.

Qwen declares attention against bounded cache capacities and temporary page
pools. Its selected plans support replacement page pools with the same layout.
Bootstrap measures graph memory with sizing captures, then describes the
final cache geometry with CUDA tensor metadata for compilation. The compiler
pool exits before the final KV allocation is materialized. Real buffers must
satisfy compiled guards before warmup and serving capture.

Searchable operations use deterministic finite fixtures with active metadata at
their declared geometry. Weighted operations use the actual checkpoint weights.
Each timed trial has an activation producer and, where needed, state restoration.
Serialized GDN and routed-expert candidates each share one fixture per request
and use the shared operator scratch arena. GDN state fixtures borrow the unused
hybrid state pool while preserving the serving row strides. After trial owners are released,
preparation synchronizes, restores every hybrid state region and PLE history to
its admission value, and synchronizes again before capture. A restoration failure
prevents capture and freeze.

Hyperconnection priming borrows the decoder's zero-initialized residual
storage through capacity views. Priming only reads that source; normalized and
projected outputs occupy separate scratch. Autotuning retains private seeded
fixtures.

One `ignition.preparation.scratch.ScratchArena` supplies byte storage to quantized
projections, GDN, QSA, routed MoE, PLE, and vision attention across serialized
layers and graph executions. `arena.views(*shapes_and_dtypes)` returns contiguous
typed views with 256-byte-aligned, nonoverlapping regions for buffers needed by
simultaneous work. Serial invocations reuse the arena from offset zero.
`register_parallel(*plans)` declares operations whose scratch must coexist;
their aligned requirements are added within each group. Separate groups reuse
storage, so the reserved extent is the maximum across serial operations and
concurrent groups.
`reserve(nbytes)` grows the backing tensor during preparation. Candidate
measurement uses the maximum scratch requirement of the complete legal
candidate envelope, including alignment, rather than a sum of per-family,
per-layer, or per-candidate allocations. After selection and CUDA synchronization,
`trim(nbytes)` replaces that storage with the peak requirement of the selected
plans and concurrent groups. `freeze()` then rejects growth beyond the retained extent.

`RecordedPreparationSession.register_scratch(arena)` associates that storage
with the batched declarations. The session reserves the complete candidate
budget before measurement and keeps that extent stable throughout the trials.
It obtains the selected requirement from each plan's public
`memory_requirements()`, synchronizes before replacing storage, trims the
arena, and freezes the retained extent. `arena.finalize()` then refreshes
every binding and shared alias against that final allocation, including
singleton operations whose equivalent declarations share a prepared payload.
Standalone component tests using a plain b12x session prepare and bind eagerly;
model startup always uses the batched, tuned lifecycle.
Bound addresses must not change after capture.
The arena remains owned until all referencing graphs and native CUDA work drain.

One preparation-owned auxiliary stream overlaps independent GDN projections,
shared and routed experts, and QSA selection with its main QKV producer.
Concurrent branches use disjoint scratch views. Opaque operations preserve
forks, joins, input lifetimes, and tensor mutations through forward compilation.
Every branch rejoins before dependent outputs or reused scratch are consumed.
Python captures these dependencies inside one CUDA graph; Rust submits that
graph without calling Python or managing the internal streams.
Preparation with `--no-stream-overlap` creates no auxiliary stream: recipes
compose each branch on the main stream with the same kernels and register no
concurrent scratch, to test whether a fault depends on overlap.

Startup compiles complete text forwards with Inductor before capture, including
prefill absorption, ordinary decode, resume sampling, and speculative rounds.
Prepared b12x operations remain opaque calls; pointwise PyTorch work is visible
to fusion and autotuning. Forward tracing uses tensor metadata without launching
kernels. Prepared MoE calls expose the complete native operation, including
route packing and caller-owned scratch/output mutations. Structured return
values are flattened into tensor outputs for compilation and reconstructed by
the preparation callable. Compilation preserves shared-storage mutations and
retains prepared owners until graph teardown. Serialized Inductor callables
receive persistent tensors as runtime inputs; native call indices refer to each
forward's retained invocation table. The compiled-forward cache stores code
without checkpoint payloads or process-local kernel handles. Cache misses autotune with
the bound buffers before capture and serialization. Tests can set
`compile_decoder=False` to capture the same prepared operations without Inductor.
Serving never executes either Python forward after native handoff.

b12x retains priming `PreparedCall.owners` with prepared plans. Priming calls use
`prime_owners(arena, ...)` to retain the arena object without retaining obsolete
views or bindings from a larger allocation. Serving components own their
refreshed final bindings. Benchmark calls retain concrete trial views and
bindings through measurement and teardown. Explicit raw-tensor callers retain
their supplied storage owners. Replacing `arena.tensor` is insufficient to
release memory if another owner still references its previous storage; ownership
qualification checks those references as well as the reported arena extent.

Operator benchmark operands and trial-state arenas live in
[`preparation/fixtures`](../ignition/preparation/fixtures/). GDN, fused MoE,
hyperconnection, variable-length attention, blockscaled GEMM, and MTP feedback
each provide a concrete fixture factory. Recipes retain plan declarations,
priming, selected bindings, and finalizers. Fixture factories preserve checkpoint
weight identity, tensor layouts and aliases, producer order, candidate scratch
requirements, and cleanup before capture; they do not own serving execution.

Scratch contents are volatile across operator calls. Kernels may not depend on
bytes left by a previous call. Model state, cache pages, request metadata,
weights, CSF expansion storage, and outputs that remain live while another
operator runs have separate storage. Decoder and vision activation buffers
therefore retain their existing lifetime boundaries. Graph variants share the
operator arena only under the native model stream's serialization and captured
side-stream joins. The preparation report records the initial allocation,
candidate requirement and reservation, selected requirement, final reservation,
and pending persistent bytes under `scratch_arena`. The model memory report records the retained
allocation as `operator_scratch_bytes`.

Selection caches are separate from compiled executable caches. The serving
namespace includes checkpoint revision, configuration and tensor-header hashes,
workload schema, and numerical policy. b12x also keys selection by compute
capability, SM count, query/configuration/semantic/candidate-contract versions,
encoded query, invocation, explicit configuration, and selected dependencies.
Completed cache entries can avoid a search; incompatible identities cannot reuse
the same entry. Compiled kernel objects persist across restarts independently.

The search preserves activation precision and arithmetic constraints. BF16
projections search every eligible b12x backend and `rows_per_tile` configuration
using private activation and output buffers with the serving tensor strides.
Hyperconnection down and residual-injection weights share one packed projection;
the canonical Transformers parameters remain views of that storage. GDN and
sparse-attention MXFP8 projections search BF16 and quantized activation paths.
Quantized MXFP8 wrappers retain default inner dense-GEMM lowerings; their
benchmark callbacks do not expose an inner search. Vision NVFP4 projections remain A16;
routed MoE remains activation-quantized. The preparation report records
arithmetic constraints, selected plans, cache hits, timings, and memory use.

Preparation proceeds in this order:

1. Validate checkpoint architecture, quantization, tensor mappings, and hardware.
2. Construct the canonical model and required replacements. Load packed weights
   through `DirectWeightLoader`; declare immutable table payloads.
3. Describe persistent state, activation/output buffers, and the serving shape
   catalog. Use bounded cache storage for sizing and kernel preparation.
   Constructors declare requests without compiling or priming them.
4. Reserve the complete candidate scratch envelope and tune all declared
   operations in one batch. Synchronize, trim to the selected peak, refresh
   bindings, release trial owners, and restore borrowed state.
5. Measure graph memory with sizing captures and determine cache geometry.
   Compile complete forwards from tensor and selected-plan metadata on rank
   zero. Peers load the resulting variants without tracing or compilation.
6. After compiler workers exit, materialize final KV storage through
   `capture_forwards`' materialization callback. Bind live tensors and native
   handles, validate guards, and refresh cache-maintenance pointers.
7. Warm every variant in rank order and capture each rank's graphs. Preserve
   mutations, aliases, forks, and joins. Freeze storage and retain compiled
   kernel owners through native drain.
8. Validate the catalog and frontend assets, export the runtime package, and
   close the prepared model. `flare run PACKAGE` loads the exported package.

`--no-compile` skips forward compilation for development; b12x preparation,
tuning, warmup, and graph capture still run. Unresolved selections, failed
bindings, missing variants, or insufficient memory prevent export. Neither
serving nor package loading invokes the tuner or Python.

Weight loading and kernel preparation use `ignition.progress.phase`. Detailed
b12x panels share its console and remain transient; only the root TP rank
renders progress.

## 3. Helios runtime and graph interface

Helios is the pure Rust serving library, packaged as the `flare-helios` crate.
The `flare/` crate links Helios into the public serving executable. The internal
`heliopause/` crate also depends on Helios and builds the `flare_heliopause`
CPython extension used by Ignition. Helios has no dependency on either caller.

### Process and ownership model

Ignition prepares and exports a model in its own Python process.
`flare run PACKAGE` starts a separate Rust process, loads the package through
`NativeResources`, binds listening
sockets, and owns the server lifetime and request paths. The
[runtime-package design](runtime-package-design.md) specifies storage, kernel,
graph, and frontend reconstruction.

`Server` and `ModelRuntime` consume the package's
[`PreparedContract`](../helios/src/model/contract.rs). Shared code validates
vocabulary and capacities, resolves each graph's value sources and byte views
into a `BoundContract`, packs uploads, and validates completion records.
Neither Flare nor Heliopause dispatches on model identity or schema names.
Tool and reasoning parsers and image preparation/layout functions are selected
independently by typed frontend enums. Missing fields, unknown or unspecified
enum values, invalid storage views, and incompatible input geometry reject loading.

Pending-token input sources determine the scheduler's completion dependency.
Host-fed decode waits for its preceding completion; device-fed decode declares a
persistent state read and the admission and sampling graphs that produce it.
Each selected graph declares its continuation-row requirement and embedding-table
resource IDs. A speculative completion cursor references the graph's updated
history storage and a token lag, or is absent when the graph maintains no draft
history. These declarations preserve the existing completion ABI.

Captured operators use b12x's default numerical policy and tuned execution plans.
MoE reductions may use arrival-ordered atomic additions, so identical graph inputs
and RNG state do not guarantee identical logits or sampled tokens. State transfers
and request metadata remain exact; recomputed floating-point results use calibrated
component bounds.

`NativeResources` owns restored CUDA contexts, modules, graphs, device memory,
and host mappings until the server confirms that all replay has drained.
The live graph bundle contains process-local handles; packages serialize
descriptions and data from which the loader creates those handles.

Heliopause's optional `qualification` feature also supports in-process replay
and HTTP diagnostics. In that diagnostic path, a Python resource owner retains:

- Transformers modules and prepared b12x objects.
- PyTorch tensors and captured graphs.
- Mapped host allocations.
- Any captured library resources.

Rust retains a strong reference to this owner while borrowing its graph handles
and buffer addresses. The worker sets the same CUDA context used during
preparation and executes without acquiring the GIL. Native tokenizer, template,
processor, and parser instances are constructed during startup and used without
Python on every request.

The diagnostic graph bundle is an in-process execution object. Startup consumes the prepared model in
this order:

1. Validate bundle metadata, context, capacities, and native frontend assets;
   retain the strong resource owner.
2. Start the native workers and wait for their ready acknowledgement.
3. Bind the server socket and admit requests.

Any startup failure joins started workers and drains submitted work before
releasing the resource owner. A failed startup exposes no serving endpoint.
Shutdown stops admission, closes listeners, completes or cancels requests,
drains CUDA work and transfers, and joins native workers before releasing the
Python owner with the GIL. Interpreter finalization with live workers is an
error. Python can block in the native wait operation with the GIL released;
it has no request-processing or server-control callback loop.

PyTorch exposes the instantiated executable handle through
`raw_cuda_graph_exec()`. Re-instantiation can invalidate that handle, so
preparation must finish before Rust registration and no subsequent
re-instantiation is permitted.
[PyTorch CUDA graph API](https://docs.pytorch.org/docs/main/generated/torch.cuda.CUDAGraph.html)

Use direct CUDA bindings inside a small Rust module. The core does not require a
Rust tensor framework or LibTorch model execution.

### Diagnostic Python interface

Heliopause's qualification binding exposes preparation and server lifetime:

```text
prepare_model(model_spec, capacities) -> PreparedModel
start_server(PreparedModel, ServerConfig) -> ServerHandle
ServerHandle.wait()
ServerHandle.close()
```

`start_server` consumes the prepared model; a second handoff is rejected.
`ServerConfig` declares native listener and request-processing capacities.
`ServerHandle` retains the runtime and resource owner until shutdown finishes.
Request submission, cancellation, event consumption, and stats are internal
Rust interfaces exposed through the native server, not Python methods.

[Direct execution](direct-execution.md) is a separate diagnostic path,
requested by the project owner, for reproducing model faults without Helios:
Ignition schedules, packs and replays the prepared graphs from Python, or
calls their captured forwards with `--eager`. It is never production serving.

A normalized native request contains token IDs, generation parameters,
positional/conditioning metadata, any preprocessed image inputs with their
position-embedding interpolation tables and per-image sequence offsets, and an
optional compiled grammar. It contains no executable Python callbacks.

Events carry stable request IDs, generated token ranges, completion reasons, and
errors. Rust incrementally decodes tokens, parses reasoning and tool calls, and
formats HTTP/SSE responses. CPU-heavy tokenization and image processing run on
bounded preparation workers so they cannot block socket service, health, or
cancellation. Streaming disconnects cancel the associated request; an event
channel closing without a terminal event reports an error.

The native model thread uses a private `Worker` for admission, completion,
publication, retirement, planning, and submission. Every completion is validated
before scheduler mutation. Exact message and prompt checkpoints become
visible atomically in scheduler completion; stopped requests retire before another
completion is processed. Submission reserves scheduler ownership before model
input packing and graph enqueue. The executor remains outside the worker's
unwind boundary, and reusable inputs remain owned until stream drain completes.

Native [`Error`](../helios/src/error.rs) distinguishes admission, execution,
and frontend failures. Cloneable failures retain their underlying source chain;
CUDA failures retain the operation, numeric code, name, and description. Dispatch
and error assertions use variants rather than message substrings. Request
validation maps to HTTP 400, model admission capacity or worker unavailability
to 503, and execution failure to 500. Generation responses wait for scheduler
admission before sending HTTP 200 or initial SSE frames. A grouped completion
starts when its first choice is admitted so its bounded output queue can drain
while other choices wait. Requests with zero output tokens need no scheduler
admission. Failures observed before response headers use the corresponding HTTP
error status; failures after SSE headers terminate the stream with an error event
and no successful terminal marker. Chat and completion errors include `type`,
`message`, `param`, and a typed `code`; Responses uses `response.failed`, and
Messages uses its `error` event envelope. After frontend preparation, generation
responses carry an `x-request-id` header matching failure logs, which include a
GMT timestamp, request identifier, failure stage, and typed error code. Workers
retain the first failure, and a failed CUDA drain retains the graph and buffer owners.

### Native frontend dependencies

Helios uses Axum/Tokio for HTTP, Hugging Face `tokenizers` for tokenization,
and MiniJinja with Python-method compatibility for chat templates. Dependency
versions and features are specified in [Cargo.toml](../helios/Cargo.toml).
The [template renderer](../helios/src/frontend/mod.rs) supplies Python-compatible
JSON formatting, whitespace handling, and template exceptions. Rendering must
match the recipe's canonical template bytes and token IDs, including tool schemas
and image content. Recipes normally use the checkpoint template; Gemma 4 uses
Google's canonical templates across quantization publishers as specified in
[frontend metadata](model-development.md#sampling-and-frontend-metadata).

The [native image processor](../helios/src/frontend/images/mod.rs) is checked
against the checkpoint's Transformers processor. Compare resized and normalized
pixels, patch order, image grids, placeholder expansion, and positions before
admitting image requests. PNG, JPEG, and static WebP decoding preserve the
prepared profile's alpha and EXIF orientation semantics. JPEG and lossy WebP
RGB values use a per-channel tolerance of two against Pillow fixtures;
resize, normalization, and patch formation must
match exactly when given the same decoded pixels.

Images replay individually into a shared request feature buffer. Admission bounds
each image by the prepared patch graphs and their combined features by that
buffer's capacity. The serialized image-support count remains readable in existing
packages; runtime image counts follow feature storage rather than a four-image cap.

### Prepared graph bundle

The bundle exposes:

| Contract | Contents |
|---|---|
| Identity | ABI version, model fingerprint, device and CUDA context |
| Executables | Graph handles, operation kind, supported capacities |
| Buffers | Stable addresses, byte extents, types, strides, and roles |
| Resources | Page geometry, live-state and snapshot capacities |
| State operations | Initialize, restore, snapshot, and reset graph entries, each an indirect copy kernel driven by device-resident slot-index buffers |
| Execution ordering | Stream ownership and scratch-sharing constraints |
| Frontend | Fingerprinted tokenizer and chat template; typed EOS, template-token, image-preprocessing, and sampling settings; native output-parser selection |
| Lifetime | Strong owner reference and shutdown requirements |

The scheduler and transport operate on page IDs, state-slot IDs, capacities,
and graph entries. The model contract interprets the prepared buffer schema
and packs model-specific uploads. Preparation declares the state layout and
storage extents; the native contract validates how each operation uses them.

### Graph families

Prepare these operations:

- **Prefill:** metadata preparation, embedding, model forward, cache writes,
  state updates, and optional final-token sampling. Prompt penalty counts are
  initialized once at admission. The scheduler clips a chunk at a required
  message or prompt endpoint. GDN writes its final live recurrent state at
  that exact length; internal aligned checkpoint export is disabled.
- **Decode:** gather pending token from device request state, forward, sample,
  update request state including the penalty histogram and PLE n-gram history,
  retain the final hidden vector, and write completion records.
- **Resume sampling:** vocabulary projection and sampling from a checkpoint's
  saved final hidden vector.
- **Vision:** encode preprocessed patches and write fixed-capacity embedding
  storage.
- **State maintenance:** initialize or restore a live slot, or freeze its exact
  consumed state into an immutable snapshot. Semantic bindings carry the
  expected consumed count, source and destination state rows, and physical-slot
  target/draft page tables for snapshot and restore.

Padding changes capacity, never logical sequence length. Every padded row must
be excluded from KV writes, convolution updates, recurrent updates, selector
updates, and sampling.

Vision graphs use bounded patch capacities declared at startup. The Qwen catalog admits up to 16,384 patches per image, with capacity buckets of 256,
1,024, 4,096, and 16,384. Oversized processed images fail admission; the server
does not silently change processor settings.

### Native replay loop

Maintain persistent device storage indexed by request slot:

- Pending input token.
- Consumed-token count and position state.
- Generation count and completion flag.
- Page tables and state-slot references.
- Sampling parameters and RNG state.
- Penalty histogram over the vocabulary, about 1 MiB per slot.
- PLE n-gram history: the last two consumed tokens, gathered into packed rows
  each step because the b12x hash operator reads and never mutates it.

Rust uploads prompt chunks and scheduling changes through reusable pinned
buffers. Page tables remain resident; decode updates only changed entries.

The decode graph writes the sampled token back into the request's device state.
The following decode consumes it directly.

Ordinary decode uses two completion frames and permits at most two steps
in flight. Speculative generation permits one round in flight, so width
selection, copy proposals, and host constraints use its completed result. A copy
stream transfers compact token/status records after their production event. A
frame is reused only after its transfer completes. Input staging buffers are
double-buffered so that a frame's uploads never overwrite a pending transfer.

A grammar-constrained request needs a vocabulary mask for step k+1 derived from
the token sampled at step k. While any constrained request is in the decode
batch, the loop runs one step in flight: the Rust-owned host grammar engine
advances on the transferred token, writes each constrained lane's mask into the
staging buffer, and the sampling operation applies the mask before filtering.
Unconstrained ordinary-decode batches keep two steps in flight. Grammar compilation and
advancement do not call Python.

Device completion flags suppress further mutation for finished requests. Slot
retirement waits for every submitted operation that can reference the slot.

Prefill and vision execute on the same model stream as decode. This
allows b12x scratch and CSF buffers to remain shared safely. QSA's optional
selection stream and its readiness event are bound before capture; the captured
fork and join execute on replay.

### Sampling and RNG

The graph boundary includes sampling.

When tensor-parallel ranks maintain replicated sampling state, rank zero's
sample and validity determine every rank's update. Captured broadcasts precede
dependent model work and state commits. Learned draft proposals broadcast both
the chosen token and its exact FP32 proposal distribution, represented by at
most 64 token/probability pairs. Verification broadcasts the accepted prefix,
output tokens, validity, and RNG reservation before cache or request state changes.
Replicated floating-point arithmetic need not be bit-identical across ranks.

Temperature, top-k, top-p, min-p and penalty controls use one captured
probability-sampling path built from FlashInfer primitives. Captured operations
manage penalties, grammar masks and request-state updates. Temperature must be
positive; zero-temperature requests are rejected because greedy decoding
collapses the model's probability distribution.

Execution ABIs 2 and 3 retain `sampling_mode="general"` as their sole
supported manifest value. Bounded top-k optimization is deferred. Measure
sampler cost inside the complete graph before changing it; threshold ties
must never be truncated to a fixed number of candidates.

Repetition, presence, and frequency penalties read a per-slot vocabulary
histogram. A captured operation builds it from the full prompt token buffer at
admission exactly once, including on a prefix hit. Prefill does not increment
those prompt counts again. Each sampled output token increments its count once
after sampling, including samples from prefill and resume-sampling graphs;
consuming that pending token on the next decode does not increment it again.
The histogram is request state outside every checkpoint.

Keep explicit per-request seeds and counters in server-owned device state.
Pass device seed and offset tensors to FlashInfer's sampler. Each request uses
a one-row draw with Philox subsequence zero and word offset `4 * counter`, so
batch packing cannot change its random stream. Counters advance once for each
successful live sample and remain unchanged for
inactive, invalid or terminal requests.

Do not capture PyTorch generator-backed random operations in graphs launched
directly by Rust. PyTorch's own replay wrapper performs RNG bookkeeping before
launching its executable; a raw launch bypasses that bookkeeping.
[PyTorch replay implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDAGraph.cpp)

Sampling defaults are declared in each Ignition recipe after manual verification
against the model card for the default serving mode. `generation_config.json`
and `config.json` do not select sampling policy. Unspecified filters are disabled
and penalties are neutral. Complete defaults are stored in the package's
`DefaultSamplingParameters` protobuf message; missing or invalid defaults reject
loading. Optional inclusive ranges in that message bound request distribution
parameters; values outside those ranges clamp to the nearest endpoint. An absent
range fixes the parameter at its declared default, including for packages without
range metadata. Seed remains request-controlled. Qwen's default thinking-mode
recommendation is temperature 1.0, top-k 20, and top-p 0.95.

### Scheduling and output

Use a small scheduler:

- FIFO admission subject to resource reservations.
- A decode batch for runnable requests.
- At most one bounded prefill chunk between decode batches.
- Cancellation and completion processed at submission boundaries.
- Eviction of unused cached checkpoints before admitting additional work.

Reserve the private pages required for admitted prompt and output capacity.
Running requests therefore do not need emergency eviction of their own state.
Four requests with a 32,768-token output limit reserve about 3.2 GiB of output
pages at 24.75 KiB per token.

Rust token decoding and HTTP/SSE streaming consume native events asynchronously.
Bounded per-request output storage prevents an unresponsive consumer from
blocking unrelated requests. Native health reporting reflects worker readiness
and failure state without querying Python.

EOS and output-length enforcement belong to helios/device request state.
Token-sequence stopping also belongs to the core; presentation-level text
trimming must not mutate cached model history.

## 4. Prefix checkpoints and complete model state

### Immutable message, prompt, and terminal cache

The cache retains immutable, content-addressed checkpoints at the rendered
endpoints of complete messages, final prompt boundary, and terminal consumed boundary.
Equal endpoints share one checkpoint. Parent-linked complete-page chains,
ancestor pinning, and eviction preserve ownership across active branches.

The [Rust prefix cache](../helios/src/cache.rs) manages ownership, readiness,
pinning, and eviction. Helios allocates backing storage when loading the package;
[captured maintenance operations](../ignition/recipes/qwen38/maintenance.py)
reset, restore, and snapshot its contents.

Use one logical page allocation to identify the corresponding storage across all
sparse-attention layers:

- Main K pages.
- Main V pages.
- Compressed selector-key pages.

For 64 original tokens, QSA's compression ratio gives 16 selector rows. Live
raw selector rings are request-local; snapshots preserve the committed ring
entries needed to continue an incomplete four-token selector group.

### A checkpoint represents one exact consumed prefix

A Qwen checkpoint includes:

| State | Purpose |
|---|---|
| Immutable main KV pages | Attention history |
| Compressed selector-key pages | Sparse-attention selection history |
| Private partial main KV and selector pages | Exact endpoints within a page |
| Raw selector keys, logical tags, RoPE tags, and interval anchor | Open selector-group continuation |
| GDN recurrent matrices | Recurrent attention continuation |
| GDN convolution histories | Causal projection continuation |
| PLE convolution history | Dilated lexical-state continuation |
| PLE n-gram token history | Correct subsequent embedding lookup |
| Position continuation metadata | Text and multimodal RoPE continuity |
| Final hidden vector | Sampling after a complete prompt cache hit |
| Pre-mixer residual and its position, when speculation is prepared | Target feedback for learned MTP |
| Shifted draft pages and raw selector state, when learned MTP is prepared | Continuation at independently tracked draft length |

The PLE table and model weights are shared immutable model resources, not copied
into checkpoints.

For target consumed count `S`, snapshot filtering keeps raw entries with logical
positions in `[0,S)` and sets the restored interval anchor to `S−1`. Initialization
has zero consumed tokens, no valid raw entries, and anchor `−1`. Restore starts
with one accepted input; rejected speculative entries cannot become visible
through raw-ring tags.

Speculative target state reserves eight recurrent checkpoint columns per request.
A round writes the columns corresponding to its verification inputs.
Snapshot selects column `previous_accepted−1`, copies the corresponding three-token
GDN and nine-token PLE convolution windows, and zeroes the unused speculative
history tails. Restore installs this normalized state in the live column and
sets its accepted count to one. PLE token history and hidden/residual vectors
already describe the committed target endpoint.

Learned MTP has consumed `max(S−1,0)` feedback pairs. Its restored raw anchor is
`max(S−1,0)−1`; its validity mask uses the draft count, independently of target
raw state. Draft logical page `j` uses physical allocation `j+1`. Write
`S=64N+r`: save draft page `N−1` when `N>0`, and page `N` when `r≥2`. Thus up to
two draft pages need private restoration even though the target saves at most
one partial page. `S=1` has no valid draft page or raw entry.

Distinguish:

- **Consumed tokens:** incorporated into every model state.
- **Pending token:** sampled but not yet consumed.
- **Visible output:** tokens exposed after presentation rules.

Only consumed history can define a checkpoint.

### Exact endpoint capture

The shared frontend renders each message prefix without a generation header or
assistant continuation. It accepts every nonempty rendered token sequence that
is a strict prefix of the complete prompt, irrespective of message roles. A
prefix may contain a template-generated preamble or end after any system,
developer, user, assistant, or tool message. Templates can reject an incomplete
conversation after emitting a complete prefix; only emitted tokens matching the
validated full prompt are eligible. Context-dependent rendering that changes
previous tokens cannot create a checkpoint for that prefix. Equal endpoints are
deduplicated. Image placeholder expansion shifts endpoints to their positions in
the prepared token stream; no endpoint may split an atomic image span.

Endpoints are not rounded to token groups or KV pages. The scheduler obtains a
snapshot destination when it reaches an endpoint, using normal cache eviction.
If all snapshot slots are pinned, inference continues without publishing that
checkpoint. The number of messages does not enlarge the prepared snapshot pool.

At each required prefill endpoint, the scheduler drains submitted work, snapshots
the exact live state, and publishes only after the copy fence completes. The
published endpoint pins the request's shared parent and transfers its completed
private pages into that chain without changing physical page order. Further
prefill or generation resumes afterward. Terminal capture uses the final consumed
count and excludes a sampled pending token. No rolling aligned candidate rows,
internal GDN checkpoint exports, or boundary-label fields participate in this path.

The scheduler and prefix cache use the model's declared physical token capacity
per page. Qwen uses 64 tokens, GLM uses 256, and DeepSeek uses 128. Prefix identity uses fixed 64-token
hash blocks independently of physical storage. CPU lifecycle tests cover multiple
page geometries, exact message/prompt/terminal checkpoints, and branching restores
with private partial pages.

Complete target pages are immutable and shared. When `S % page_tokens != 0`, snapshot
storage owns the whole physical tail page and restore copies it into a private
allocation; at an aligned endpoint, maintenance never writes a shared full page.
The copied unused KV rows and incomplete selector-group values do not define
committed state. Valid token ranges, completed selector groups, and raw tags do.

The model state pool uses row zero for the null state. For runtime concurrency
`C` and maximum loaded verification width `W`, request slot `s` owns rows
`1 + s * W` through `1 + (s + 1) * W - 1`. Immutable snapshot `i` occupies
row `1 + C * W + i`. These partitions remain fixed while the model is loaded.
Metadata validation checks the exact source count,
epoch, transfer direction, needed page mappings, and cross-row aliases before
any state mutation.

Snapshot and restore preserve committed state exactly, protect immutable sources,
and allocate no storage during replay. Restored continuation uses the same exact-transfer and numerical-repeatability
requirements as direct continuation with the same partition and graph geometry. Comparing
uninterrupted prefill with split prefill also measures GDN partition sensitivity.

### Full prompt hits

Save the final hidden vector with each checkpoint: the combined stream after the
final hyperconnection mixer. Qwen4Exp sends that vector directly to `lm_head`;
there is no separate final normalization. The resume-sampling graph applies
vocabulary projection and sampling.

When the entire prompt matches a checkpoint, restore its state and run the
resume-sampling graph. This avoids replaying an already-consumed token or
rerunning the prefix merely to recover logits.

Sampling state starts from the new request's generation parameters and seed.
Prefix reuse does not inherit the previous request's RNG stream.

### Branching and lifetime

A branch pins its ancestor checkpoint, shares immutable pages, restores state
into a private live slot, and allocates private suffix pages including all
required target/draft boundary pages. A duplicate publication releases only its
unused snapshot reservation; it does not replace computed live pages with another
request's physical page chain. Admission reuse counts remain fixed even as the
request publishes additional shared ancestors.

Track logical references separately from outstanding GPU use. A page or slot
becomes reusable only after both have ended.

Under snapshot pressure, evict unpinned leaves first. If every leaf is pinned,
remove an ancestor checkpoint with exactly one child and no direct request pin:
transfer its ordered page segment and last-use fence to that child, bypass the
ancestor in the parent chain, and retire only the ancestor snapshot. Descendant
checkpoint IDs, snapshots, and complete page chains remain valid. The removed
prefix no longer matches lookups. This bounds snapshot consumption during long
appended conversations without discarding the longest reusable prefix. Branching
ancestors and directly pinned checkpoints retain their snapshots.

Maintain resource-specific padding sentinels in the prepared metadata contract.
Padding must never address another request's writable state.

### Multimodal identity

The chained prefix digest includes conditioning identities at the point where
those inputs enter the token stream.

Image identity includes the complete decoded representation consumed by its
preprocessing profile and the profile's settings. The prefix also identifies
the model and token layout. Appending another image preserves the digest of an
earlier unchanged prefix. Encoded-source keys are separate: they include the
declared encoding and source bytes and accelerate description lookup.

A sequence of image-placeholder token IDs alone is insufficient to identify
model state.

Canonical positions, token dependencies, and feature indices describe the whole
prompt. Admission derives a separate execution plan from the consumed endpoint
of the selected GPU or host checkpoint. Only inputs whose dependency spans
extend beyond that endpoint require preprocessing and vision replay. Selected
features occupy contiguous storage beginning at zero; input packing maps the
canonical indices into that storage without changing prompt identity.

CPU descriptions and prepared tensors use the host retention tiers beside
complete model checkpoints. Their eviction is independent: a checkpoint needs
no original image tensors to resume after an image span, and prepared tensors
can be reused when the surrounding text changes. Requests retain reconstructible
sources through preemption and restore failure, so readmission can select work
against a shallower prefix.

### Memory significance

Per-slot state for this model:

| State | Size per slot |
|---|---:|
| FP32 GDN recurrent matrices, 36 layers | 108 MiB |
| GDN convolution histories, 36 layers × 10,240 channels × 3 | 2.1 MiB |
| PLE convolution history, 10,240 channels × 9 | 0.2 MiB |
| PLE n-gram history, position metadata, hidden vector | under 8 KiB |
| Target raw selector state and partial pages | about 1.6 MiB |
| **Ordinary snapshot row** | **117,291,776 bytes (111.86 MiB)** |
| **Seven-draft state row** | **122,932,224 bytes (117.24 MiB)** |

The preparation API's 4 GiB checkpoint-state budget holds 36 ordinary or 34 seven-draft
snapshot rows, beyond the null/live rows and speculative recurrent-column
storage. The speculative layout includes extended convolution histories, target
feedback, and two shifted draft boundary pages. Preparation validates that the
configured snapshot budget can hold at least one row of the selected layout.

BF16 main KV across the 12 attention layers consumes **24 KiB per cached token**,
with compressed selector keys adding **0.75 KiB per token**. Main KV stays BF16
because the checkpoint declares no KV-cache quantization; QSA's FP8 E4M3
main-cache mode with per-layer descale halves the page cost and is a separately
qualified option.

Report page pressure and snapshot pressure separately. Eviction may be necessary
to recover either resource.

Host memory and disk retention extend this GPU cache through the proposed
[prefix-cache offload design](prefix-cache-offload.md). Offload copies only
published immutable checkpoints and restores complete page chains plus their
endpoint row. Its implementation and interference measurements are required
before enabling derived host-store defaults.

## 5. Implementation requirements and acceptance

### Integration and remaining work

Model integration follows the dependency order below. Each stage must preserve
the contracts established by the preceding stages.

1. **Reference and constructor wiring.** Use the declared checkpoint,
   quantization metadata, and canonical model semantics. Reference loaders must
   preserve CSF, NVFP4, MXFP8, PLE, and activation policy. A dequantized BF16
   model is a diagnostic reference for different arithmetic. Exercise the real
   preparation constructors on tensor metadata without CUDA initialization,
   using [the bootstrap construction test](../tests/preparation/test_bootstrap_construction.py).
2. **Prepared operators and complete graphs.** Compare components on identical
   operands, then validate metadata preparation, cache writes, sampling,
   arbitrary tails, maintenance, and full-prompt resume. Compile and capture
   every supported shape with explicit state and scratch lifetimes.
3. **Native package execution.** Export and load the actual package, validate
   request packing and completion interpretation, and compare Python and Rust
   replay under the same arithmetic policy. Test multi-rank agreement and
   coordinated shutdown.
4. **Agent request lifecycle.** Exercise repeated turns, tool-result append,
   shared-prefix branches, images, grammars, cancellation, and cache pressure
   through HTTP and SSE.
5. **Speculation.** Validate verification, commit, and vector completions with
   host-supplied copy proposals before depending on learned drafts. Add MTP
   feedback and history absorption, then compare fixed depths before adaptive
   width selection. The [MTP design](mtp-design.md) defines the transaction.
6. **Arithmetic and kernel efficiency.** Remove unjustified tuning constraints
   after calibrated component and continuation checks. Attribute graph cost to
   kernels and dependencies before changing fusion, precision, or overlap.

Structured output composes the answer grammar with the output protocol's
reasoning and tool delimiters. The grammar constrains the answer, while tool
payloads retain their normal parsing and validation. The native sampling mask
and speculative grammar advancement apply to the complete composed grammar.
`tool_choice=required` adds a prompt
instruction and validates that the model produced a completed call to a declared
tool. Generation uses ordinary sampling; an unsatisfied requirement fails the
response. Named tool selection filters the available definitions to the selected
tool without adding instructions or decoding constraints. Output validation
requires a completed call to that tool and rejects other tool names; presentation
omits ordinary answer text while preserving reasoning and tool arguments.
`tool_choice=none` removes tools from the template. Strict function schemas are
unsupported. See [the API contracts](api.md) for
protocol mappings and streamed errors.

Bounded top-k sampling and compact recurrent commit require separate component
and full-graph measurements. They cannot weaken threshold-tie handling,
request RNG, or checkpoint semantics. Host prefix-cache offload has its own
[implementation sequence](prefix-cache-offload.md#3-implementation-sequence).

### Correctness acceptance

Use the exact checkpoint throughout integration qualification.

Required cases:

- Cold prefill, chunked prefill, decode, and arbitrary prefill tails.
- Batch sizes 1–4, including padded capacity and changing slot assignments.
- Boundaries around 4-token QSA groups, 64-token pages, every captured prefill
  capacity and the configured chunk limit, including 4,096-token chunks.
- EOS, output limits, cancellation, slot reuse, and resource exhaustion.
- Prefix lengths immediately before, at, and after page boundaries.
- Full prompt hits through saved-hidden-vector sampling.
- Parent/child branches continued independently after restore.
- Identical text with different images, plus repeated identical image inputs.
- PLE EOS handling and n-gram continuity across chunks.
- QSA raw-ring and compressed-cache continuity across prefill/decode handoff.
- RNG progression across repeated replay and invariance under batch reordering.
- Shutdown while requests and asynchronous transfers are active.
- Restore into a QSA raw-state slot previously used by a request at the same
  position.
- Penalty histograms after admission from a cold prompt and after a prefix hit.
- Exact-once prompt and generated-token penalty counts across prefill, decode,
  and full-prompt resume sampling.
- Grammar-constrained decode with one step in flight batched with unconstrained
  requests.
- Native template bytes and tokenizer IDs against the locked Transformers
  reference for tools, tool results, reasoning controls, assistant continuation,
  Unicode, and image-bearing messages.
- Native image processing against the exact processor reference: resized and
  normalized pixels, patch ordering, grids, prompt expansion, and positions.
- HTTP/SSE disconnect cancellation, slow consumers, terminal-event delivery,
  worker failure health, and startup failure before socket binding.

Compare Rust execution against Python replay of the same prepared graphs and
initial state. This isolates runtime correctness from kernel numerical
differences. Require bitwise equality for state transfers and metadata. Numerical
replay comparisons account for the selected kernels' reduction-order variation;
cross-capacity cold-versus-restored comparisons also account for changes in
reduction association.

Test each component on identical operands against its Transformers or b12x
operator reference, with tolerances derived from the required precision and
reduction arithmetic. Measure repeated execution when a kernel's reduction
order can vary. Generation tests use at least 50 prompts spanning chat, tool
use, and long contexts, with explicit sampling settings and a 64-token output
budget.
Compare token agreement only for matched contexts and sampling, accounting
for measured repeatability. Reusable inputs and expected outputs are ordinary
fixtures.

Teacher-forced diagnostics retain full-vocabulary logits, relative L2 with the
reference norm as denominator, top-ten overlap, and FP32
`KL(reference || actual)`. Locate the first differing layer and test its
operators before changing recurrence rounding or normalization order.
Whole-model logit and encoder relative L2 are diagnostic; numerical acceptance
uses component-specific bounds.

### Graph and lifetime acceptance

After readiness:

- No Python request, preprocessing, model-step, grammar, or output callbacks.
- No per-step tensor construction or CUDA allocation.
- No kernel compilation or recapture.
- No eager path for short prompts, tails, or changing batch occupancy.
- No mutation from inactive lanes.
- No reuse of resources still referenced by queued work.
- No concurrent execution over aliased b12x scratch.
- No dependency on HTTP/SSE consumers for device token feedback.
- No caching-allocator activity across a replay window.
- No graph-private pooled output retained across graph boundaries; test
  alternating graph families with asynchronous completion transfers.
- No resource-owner release before admission stops, CUDA work drains, and all
  native workers join.

Exercise changed inputs and state assignments on repeated replays. Replaying an
unchanged warmup input is insufficient.

### Performance acceptance

Established serving baselines set the minimum performance bar. Compare the
same checkpoint, GPU, precision, prompt corpus, sampling settings, output
budgets, cache conditions, and concurrency. Flare must match or improve
end-to-end TTFT, inter-token latency, and completed-work throughput across the
required local workloads at concurrency 1, 2, and 4. Report individual workloads
and tail latency; an aggregate win does not erase a workload regression. Repeat
noisy measurements before declaring parity. Primary serving measurements use
temperature 1, top-k 20, and top-p 0.95. Correctness and request-completion gates apply to both engines.

Measure three paths with matching model, inputs, precision, and generation
behavior:

1. A serving baseline with the same workload and checkpoint.
2. The prepared Flare graphs under a minimal replay harness.
3. The complete Rust serving runtime.

At prefill limits 512 and 4,096, collect this minimum measurement set:

| Measurement | Conditions |
|---|---|
| Cold time to first token | 512-token and 4,096-token prompts |
| Inter-token latency, median and p99 | Concurrency 1, 2, 4; 256 generated tokens |
| Decode throughput | Concurrency 1, 2, 4; completed tokens per second |
| Prefill throughput | One 4,096-token prompt and one 64K-token prompt |
| Prefix-hit time to first token | 4,096-token cached prefix plus an 11-token tail |
| Startup | Cold tuning selection; warm selection and kernel caches |
| Host submission gap | Native executor event timeline per step |

Also report:

- Time to first token and inter-token latency.
- Prefill and decode throughput.
- GPU execution time versus host submission gaps.
- Native scheduling/submission latency.
- Cache-hit restoration and suffix-prefill cost.
- Snapshot traffic, memory footprint, and capture/startup cost.
- Median and tail behavior under concurrency 1, 2, and 4.
- Per-kernel GPU timelines at batch one and four, attributed to named operators
  and layers; kernel count per layer and graph; summed kernel duration against
  total graph time.

Measure GDN recurrence splits, QSA partitions, MoE reduction policy and
hyperconnection projection precision individually against the same inputs and
timeline. Keep explicit diagnostic configurations for controlled comparisons.
Fusion work is guided by measured operator time and kernel count. Host graph
launch traces alone do not establish a GPU execution breakdown.

The replay harness establishes the achievable floor for the captured computation.
The native core must sustain decode within 5% of that floor for the same graph
sequence, with frontend delivery measured separately. This host-overhead limit
is an additional requirement; efficient replay of slow graphs does not satisfy
the serving-performance floor.

The weight-traffic bandwidth bound excludes kernel latency and other execution
costs. At decode batch 1 this model reads about
6.9 GB per token:

| Weights read per decoded token | GB |
|---|---:|
| Routed experts, 10 of 512 per layer × 48 layers | 1.26 |
| GDN projections, MXFP8, 36 layers | 2.1 |
| Sparse attention and indexer projections, MXFP8, 12 layers | 0.62 |
| Shared experts, MXFP8, 48 layers | 0.24 |
| Hyperconnection low-rank projections, BF16, 48 layers | 1.26 |
| Vocabulary head, BF16 | 1.27 |
| Routers, BF16 | 0.13 |

At the target GPU's 1.79 TB/s this gives a bandwidth bound near 3.9 ms per token.
Serving performance must meet the measured baseline, and native replay must meet
the measured graph-overhead limit.

Keep raw measurements and the configurations needed to interpret them.
Correctness failures block performance qualification.
