# Model preparation and serving

## Runtime packages

Run commands from the repository root. Ignition prepares a source checkpoint
into a **runtime package** and publishes a directory entry in
`paths.package_cache` (default `~/.cache/flare/packages`). Its primary,
`model.safetensors`, contains the signed `flare.manifest`, unsigned
`flare.locations`, optional reconstruction hints, and frontend assets. Readably
named safetensors companions hold layer weights, tuning-dependent weights,
initial state, zstd-compressed cubins, and graph records. Cache companions hardlink
immutable payload objects whose identities hash the payload bytes.

The first canonical form packs the package into one `.safetensors` file; the
second is the directory form used by the cache. Export always publishes to that
cache. `ignition convert CACHE_ENTRY --output DIRECTORY` creates a standalone
directory; `ignition convert CACHE_ENTRY --form packed --output FILE` creates
one exact file path. `CACHE_ENTRY` is the path printed by export. Packing and
unpacking preserve signed payloads and require no signing key or GPU. Layer payloads are not grouped by a size threshold. Internal
logical-resource packages use envelope 6 with monolithic, numbered-shard, or
blob layouts. Legacy blob publication uses a fixed maximum of 64 GiB per file;
there is no configurable shard-size threshold.

`ignition export --no-weights` requires the exact stable and tuned weight
payloads in the cache. Initial state and execution resources may change with
the prepared capacities. Preparation and weight reads still occur. Successful
exports select their cache entry as the model's default. Recipes can declare
`export_repository` as the Hugging Face publishing target; prepared storage
uses the shared Flare cache. Multi-host export publishes the complete entry
in the coordinator's cache. See [package snapshots](cli-design.md#package-snapshots) for cache
management, conversion, and upload.

`ignition export --reuse-weights` uses the model's cached default package as
the weight source. It reuses that version's signed weight hashes and initializer
mappings, skips reading declared numeric weight storage during export, and
hardlinks the cached payloads into the resulting package without rehashing their
contents. Checkpoint loading
and model preparation still run. The checkpoint revision, TP size, architecture,
and declared weight layouts must match; missing or incompatible weights fail
export. Cached weight bytes are authoritative and are not compared with live
tensor values. Use ordinary export after changing weight values or packing.
Reuse includes Engram's file-backed embedding weights and scale planes, in
both shared and row-partitioned table layouts. Their geometry, quantization,
vocabulary map, and hashing parameters must match the cached version.
Graphs and initial state are always exported from the prepared model. See
[export flags](cli-design.md#31-ignition-command-line-flags) for the metadata requirements.

Each tensor-parallel deployment contains independently prepared rank metadata,
graphs, and kernels. Combining exported deployments can share or reconstruct
byte-identical weights; it cannot create an unexported TP size. Ignition creates
a per-user development signing key when `[publishing]` is absent. `flare run`
needs the corresponding publisher profile and first-use acceptance. See
[signing and development](cli-design.md#signing-and-local-development).
Unsigned rank staging is private to preparation and cannot load in Helios.

Flare loads the package files without Python. Helios is a Rust library;
`flare_heliopause` is a CPython extension imported by Ignition.

Recipes default to FP8 E4M3 KV storage, including learned draft-model caches. Qwen Flash retains BF16 selector keys;
GLM and DeepSeek retain their packed latent formats, including scale metadata and any BF16 rotary fields.
Recurrent state, compressor accumulators, and hidden activations retain their
model-specific precision. Cache writers and attention readers use matching
scales where required, and memory planning includes each cache's declared page
precision and partial-page snapshots.
Export a package to use these cache layouts; existing packages retain the
layouts captured when they were prepared.

## Environment setup

Set up the preparation environment and build the native CLI:

```sh
uv sync --locked --extra prepare --extra test
cargo build --release --manifest-path flare/Cargo.toml
```

Ignition applies three bundled preparation patches to the pinned PyTorch 2.14.1 and
Triton 3.8 preparation environment before importing preparation libraries in
`export`, `debug-serve`, `bench`, and `join`, including commands launched through `uv run`.
`ignition/patches/torch-inductor-triton-cache-filenames.patch` matches
Triton's 150-character physical cache filenames while retaining full CUDA symbol
names. `ignition/patches/torch-inductor-cooperative-static-launch.patch` preserves
cooperative vision reductions in portable static kernels through Triton's CUDA
launcher, including their cooperative launch attributes and frozen selections.
`ignition/patches/triton-platform-key-pointer-width.patch` derives Triton's helper
cache key from the interpreter's pointer width without launching `file`.
Nsight Systems injection can conflict with `file`'s seccomp sandbox and hang
startup ([NVIDIA known issues](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html#general-issues)).
The pointer-width key retains machine and operating-system identity and
invalidates compiled helper caches that used executable linkage metadata.
Patch application checks dependency versions and exact source context, accepts
already applied hunks, and replaces installation files atomically without
modifying uv's shared wheel cache. Each preparation host needs a writable,
dedicated Flare environment. Reinstalling Torch or Triton restores its distribution files;
the preparation command reapplies the patches.

`uv sync` installs dependencies without running the patches, and `uv build`
includes the patch files in Ignition's wheel and source distribution without
modifying its isolated build environment. For standalone Python tools and tests
that import Inductor or Triton before invoking Ignition preparation, apply the bundled
patches before starting their process:

```sh
uv run --locked --extra prepare python -m ignition.patches
```

`python scripts/setup_preparation.py` updates FlashInfer to the tip of its `main`
branch, syncs preparation/test dependencies, rebuilds FlashInfer and Heliopause,
and applies the bundled patches before exiting. It updates `uv.lock` when the
branch advances.

Build dependencies include a C/C++ compiler, CMake, Ninja, pkg-config, Protobuf's
`protoc`, OpenSSL headers, Python headers, and liburing headers. On Ubuntu:

```sh
sudo apt-get install build-essential cmake ninja-build pkg-config protobuf-compiler libssl-dev python3-dev liburing-dev
```

Windows native builds also require Perl for the bundled static OpenSSL build,
alongside the C/C++ toolchain. The resulting binaries do not require OpenSSL DLLs.

The JPEG decoder builds its bundled libjpeg-turbo source. Preparation requires
the CUDA toolkit, including `nvcc` and CUPTI. Native build libraries discover
the system toolkit; `[paths] cuda` overrides that discovery with an explicit
toolkit directory.

The `prepare` extra installs `cuda-tile[tileiras]`, including the `tileiras`
compiler toolchain required by cuTile kernels during model preparation.
cuTile discovers its matching compiler tools in the Python environment
independently of the toolkit used for native builds.

The DeepSeek Rust crates share a pinned revision of
[`local-inference-lab/deepseek-recipe`](https://github.com/local-inference-lab/deepseek-recipe)
in `helios/Cargo.toml`. Its `flare` branch exposes normalized-message byte
boundaries, advisory checkpoint hints, and numeric V4.1 reasoning effort.
To update these crates, integrate upstream changes
on that branch, validate the encoder, and update all three dependency revisions
together. Refresh and commit `helios/Cargo.lock`, `flare/Cargo.lock`, and
`heliopause/Cargo.lock`; run the Helios frontend/protocol tests and build Flare
and Heliopause against the pinned revision. Compare rendered prompts and token
IDs with upstream when changing encoder behavior.

The preparation dependency tracks the `main` branch of
`local-inference-lab/flashinfer`; `uv.lock` pins the resolved commit. The setup
script updates it automatically. To update the lock alone, run
`uv lock --upgrade-package flashinfer-python`, then sync the environment.
For local FlashInfer development, install the sibling
checkout explicitly with `uv pip install --python .venv/bin/python --no-deps -e ../flashinfer`;
`uv sync --locked --extra prepare --extra test` restores the Git dependency.
Inspect that checkout and its local changes before modifying it. After changing
Rust code in `heliopause/`, or Helios code used by the extension, rebuild the
installed extension explicitly:

```sh
uv pip install --python .venv/bin/python --no-deps ./heliopause
```

## GPU selection

Inspect GPU occupancy with `nvidia-smi` before preparation or loading. Select
available GPUs explicitly on shared hosts. `--gpus 2` means one rank on GPU 2;
`--gpus 0,1` means two ranks on GPUs 0 and 1. With `CUDA_VISIBLE_DEVICES` set,
indices refer to the exposed device list. Each tensor-parallel size requires a
separately prepared package; a TP2 package cannot be loaded as TP1. Qwen TP1 and TP2 and
GLM TP4 have been exercised end to end; Qwen TP4 still requires model validation.

## Preparation resources

`--autotune-effort off|default|audit` controls b12x candidate filtering.
`off` prepares heuristic choices without tuning races or tuned-selection cache
lookups. `default` races the candidates that pass each kernel family's
performance exclusions. `audit` lifts those exclusions and races every
candidate satisfying correctness, resource, dominance, and explicit
configuration constraints; it exists to check that the default search prunes
no winners and is meant for occasional sampled runs, not routine tuning. The
audit search includes every default candidate. Kernel families retain their
declared tuning axes.

`--collective-tuning static|runtime` selects when collective choices become fixed.
`static` uses ordinary preparation-time autotuning and emits no collective startup
tuners. Every recipe defaults to `static`. `[preparation] collective_tuning`
overrides the recipe default; the command-line option overrides configuration.
`runtime` requires `--collective-ladder PATH` and uses the file's final joint
kernel geometry for eligible plain BF16 direct PCIe reductions. Preparation
compiles that geometry without racing alternatives. A captured tuner selects
among the supplied submode and active-CTA profiles when the package loads,
including with autotune effort `off`; geometry stays fixed and no search runs
during requests. Other collective paths retain their preparation policy. The
[GB202 PCIe TP4 ladder](../ignition/preparation/ladders/README.md) covers four
SM120 GPUs with out-of-place BF16 reductions from 16 bytes through 8 MiB:

```sh
--collective-tuning runtime \
--collective-ladder ignition/preparation/ladders/gb202_pcie_tp4.json
```

`--lock-gpu-clocks` locks each participating GPU to its maximum supported
graphics frequency for the b12x tuning batch and actual Inductor autotuning
trials during compiled-forward preparation. Frozen warmup, capture, and serving
run without these locks. It applies to serving ranks and
extra tuning replicas, and does nothing with `--autotune-effort off`. The GPU's
UUID selects the device under `CUDA_VISIBLE_DEVICES`. Run with permission to
control clocks: Ignition uses `sudo -n nvidia-smi` when not root. Unsupported
clock control or missing permission fails preparation. Locks reset to driver
defaults after tuning or a Python exception; preexisting manual clock locks
are not restored. Memory clocks and tuning caches are unchanged.

Without extra tuning replicas, tensor-parallel ranks on identical GPUs divide
each kernel race's candidates and adopt the fastest measured configuration.
This applies within one host as well as across hosts. Fixed collective kernels
still prepare on every participating rank.

Set `[preparation] autotune_effort = "audit"` for a persistent setting; an
explicit flag wins. The plan carries the setting to every preparation rank
and tuning replica. Effort separates cached selections and prepared packages;
identical kernel programs share compilation artifacts across efforts.

Tuning checkpoints each completed measurement batch in a SQLite database beside
the kernel selection cache. Tensor-parallel ranks save independently, including
completed local races waiting for winner exchange. Restarting the same workload
restores measured candidates and the carried winner, then measures unfinished
candidates against that winner. The progress display distinguishes restored,
newly measured, and rejected candidates. An interruption loses the unfinished
measurement batch on each rank; compilation artifacts remain separately cached.

Partial checkpoints require the same candidate program identities, effort,
tuning shard count, and measurement settings. A kernel or predicate change
invalidates the affected race's checkpoint. Completed selection caches remain
usable independently of the tuning shard count. Resource-dependent cooperative
launch rejections are retried after restart; fatal CUDA errors still require
restarting preparation in fresh processes.

`--compile-workers N` sets a positive CPU compiler process budget per host for
b12x kernel compilation and complete-forward compilation. Available RAM and
device memory can reduce it; local preparation ranks share the budget. Set
`[preparation] compile_workers = N` for a persistent default; the CLI flag
overrides that setting. Omitting both retains automatic sizing, including
shared-memory limits on DGX Spark.

On integrated GPUs, kernel tuning is divided into memory-bounded jobs. The
terminal dashboard shows the active job number and total job count, with a
separate live display for each job. Redirecting output records plain kernel
progress milestones instead of the terminal dashboard.

`--tuning-gpus LIST` adds local GPU replicas for benchmarking b12x candidates.
The list begins with `--gpus` and continues in complete TP groups: for example,
`--gpus 0 --tuning-gpus 0-3 --compile-workers 24` prepares a TP1 package with
four tuning GPUs. Corresponding GPUs in each replica must match in model and
architecture and have enough free memory for the model and preparation storage.
Helpers exit and release their resources after tuning; serving TP size and capture
devices stay fixed. Serving ranks then compile and capture their forwards.
Both GPU lists use `CUDA_VISIBLE_DEVICES` ordinals. Kernel compilation runs in
CPU workers, independently of the number of tuning GPUs.

Multi-host `--ranks` preparation retains its existing sharding and transport;
`--tuning-gpus` is local-only and cannot be combined with it. An explicit
`--compile-workers` budget applies separately on each host.

## Exporting runtime packages

Export publishes a directory entry in the shared model cache and prints its
path. In the examples below, replace `CACHE_ENTRY` with that path before running
`ignition convert`. Conversion requires an unused destination and does not append
a suffix. Do not delete an existing package merely to reuse its name. `MODEL` can also be a
supported repository ID or a local source-checkpoint directory. Ignition
downloads missing source weights, loads and tunes kernels, compiles forwards,
warms and captures execution variants, writes the package, and exits.
Compilation is enabled by default; `--no-compile` is for development.
The examples omit default settings. Configuration-file overrides still apply;
inspect `ignition export MODEL --help` and the printed preparation plan for the
resolved capacities. Select idle GPUs explicitly. Native serving listens on
`127.0.0.1:8000` unless configuration or `--host` / `--port` overrides it.
Add `--verbose` for per-phase detail or `--lock-gpu-clocks` to lock clocks during
autotuning.

### Re-exporting with cached weights

After an ordinary export, use `--reuse-weights` when rebuilding graphs with
the same checkpoint, TP size, architecture, and compatible weight layouts.
For a DeepSeek V4.1 TP4 export:

```sh
.venv/bin/ignition export deepseek-v4.1-flash --gpus 0,1,2,3 --reuse-weights
flare/target/release/flare run CACHE_ENTRY --gpus 0,1,2,3
```

Replace `CACHE_ENTRY` with the directory printed by this export. The command
uses the model's cached default version as its weight source and publishes
the resulting package as the default. Checkpoint loading, preparation, and
capture still run; compatible tuning and compilation caches remain reusable.
Use ordinary export after changing weight values or packing, or when no
compatible cached default exists. `--no-weights` instead hashes the live
weights and requires their exact payloads to be cached; it cannot be combined
with `--reuse-weights`.

## Nemotron 3.5 Bolt 74B-A9B 24GB

`nemotron-bolt-24gb` prepares
`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` on one SM120/SM121 GPU.
The recipe includes the C-RADIO encoder for still images. Video is not prepared. LiLiCorr uses the matching
`NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` checkpoint and defaults to
seven draft tokens with adaptive verification. Both checkpoints use the standard
Hugging Face hub cache (`~/.cache/huggingface/hub`), or `[paths].hub_cache` when configured.

This example limits preparation to eight concurrent requests and 64 target rows:

```sh
.venv/bin/ignition export nemotron-bolt-24gb --gpus 0 \
  --max-concurrency 8 --speculative-token-capacity 64
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/nemotron-bolt-images-lilicorr-c8.safetensors

./flare/target/release/flare run \
  /data/flare-runtime/nemotron-bolt-images-lilicorr-c8.safetensors \
  --gpus 0
```

Use `--no-mtp` during preparation to omit the draft model. Mamba-2 prefill
uses b12x's packed selective scan; decode and speculative verification use
FlashInfer state updates. Continuation rows retain BF16 recurrent state,
convolution history, partial FP8 KV pages, and LiLiCorr context and rollback
records. Prefix reuse and speculative acceptance preserve that complete state
through the shared Helios cache contract. Image features enter causal prefill
and may cross chunk boundaries.

Shared experts overlap routing and routed experts for graph capacities through
64 rows. Larger capacities execute those branches serially. LiLiCorr's separately
scaled NVFP4 gate and up projections also overlap. Prepare with
`--no-stream-overlap` to place those projections and expert branches on the main stream.
Routing excludes padded target rows using the packed batch's valid-token count.
The 64-row speculative capacity covers eight requests with seven draft tokens each.
For concurrency 32 with seven draft tokens per request, use
`--speculative-token-capacity 256`.

## Qwen3.8 27B

`qwen3.8-27b` prepares `nvidia/Qwen3.8-27B-NVFP4` for text generation on one GPU.
It uses the canonical Qwen3.5 dense decoder with tensor-scaled FP8 attention
projections and NVFP4 MLP and vocabulary weights. FP8 KV pages and Gated DeltaNet
continuations reside in device memory. The pinned BF16
`incoai/Qwen3.8-27B-DFlash2` draft consumes residual features from target layers
5, 19, 33, 47, and 61 before final normalization, shares the target embedding and
vocabulary head, and proposes eight-token noncausal blocks. Speculation defaults
to seven draft tokens with adaptive packed verification capped at 128 physical
target rows and a 64-graph budget. Accepted GDN corrections commit into one
recurrent row per request. Five 2,048-token FP8 draft KV rings per continuation
track target consumption without lag and survive prefix snapshot/restore.
The recipe has no host-mapped tables or vision graphs. Prepare on the same GPU
architecture as the serving device:

```sh
.venv/bin/ignition export qwen3.8-27b --gpus 0
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/qwen38-27b-tp1.safetensors
```

Use `--mtp-draft-tokens N` to select the prepared draft depth. Width-one rounds
retain draft context so adaptive verification can resume wider rounds.
`--no-mtp` skips the draft download and omits its weights and state during preparation;
`flare run PACKAGE --mtp-draft-tokens 0` selects the package's ordinary graphs.

For complete GPU graph round latency without package export, use:

```sh
.venv/bin/ignition bench qwen3.8-27b --gpus 0 --output /tmp/qwen38-rounds.json
```

This prepares only the selected benchmark graph catalog. Defaults cover request
batches 1, 2, and 4, verification width eight, and consumed contexts 512, 2,048,
8,192, and 16,384 tokens. Timing uses real prefilled contexts and restores full
cache and speculative state before each sample, outside CUDA event timing.
The [benchmark CLI](cli-design.md#gpu-graph-latency-benchmark) defines workload
selection, repetitions, seeds, and the equal-weight latency score. Native
serving measurements still use an exported package with `flare run`.

## Qwen3.8 Flash

Enable hybrid history drafting at serving startup with
`flare run PACKAGE --history-drafts`; disable it with `--no-history-drafts`.
These options use the package's existing MTP graphs and are not export settings.

Export Qwen with up to five MTP draft tokens per round:

```sh
.venv/bin/ignition export qwen3.8-flash --gpus 2
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/qwen38-tp1-mtp5.safetensors

.venv/bin/ignition export qwen3.8-flash --gpus 0,1
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/qwen38-tp2-mtp5.safetensors
```

Qwen defaults to 32 concurrent requests, five MTP draft tokens, and 128 target-token
rows, with explicit flags and configuration taking precedence. Verification is
rectangular: the default catalog covers five drafts per request through batch 16
and three drafts at batch 32. Already exported packages retain their captured
capacities.

`ignition export qwen3.8-flash --gpus 0,1 --shard-hyperconnections` partitions
the 320-wide hyperconnection bottleneck across ranks, including target and MTP
final mixers. Each rank loads its down-projection rows and up-projection columns;
normalization and residual injection stay replicated. Partial gate logits round
to BF16 before transport, and the reduced BF16 logits feed sigmoid and the stream
mean. This changes reduction rounding relative to the replicated projection.
The option requires TP2 or TP4 and is recorded in the package's rebuild metadata.
Replicated execution is the default: the compute savings must outweigh the extra
10,240-element reduction per mixer and token. Changing the option requires a
fresh export; `--reuse-weights` requires matching prepared weight layouts.

Vision recipes accept `--image-concurrency N`, independently of
`--max-concurrency`. It defaults to `min(4, max_concurrency)` and can also be set
as `image_concurrency` in `[defaults]` or a model's configuration section.
Explicit flags override configuration. Text-only recipes ignore the configured
image default and reject an explicit image-concurrency flag.
Helios reserves a feature slot when a request needs vision computation. Requests
whose image spans are already covered by a reused prefix checkpoint need no
feature slot. Requests needing a slot queue when those slots are occupied;
eligible requests continue to admit. A reserved slot remains owned until
request retirement, including completion of outstanding GPU work after cancellation.
The limit does not change images per request or the prepared patch capacities.
Changing it requires exporting the package again.
Packages with vision require the prepared feature-slot capacity and binding.
Images replay one at a time into bounded windows in the request's feature buffer.
Chunked prefill consumes each window before the buffer is reused. Combined image
tokens may exceed the buffer's capacity; each image must fit one window and a
captured patch capacity. HTTP body and total context limits also apply.
PNG, JPEG, and static WebP data URLs are accepted.

For Qwen, each feature slot holds 16,384 BF16 vectors of width 2,560: 80 MiB per
rank. Four image slots therefore reserve 320 MiB even with 32 active request slots.

N-gram tables use indirect delivery exclusively: every graph consumes compact
GPU row buffers filled by Helios. Qwen PLE tables are stored once in the package.
Helios reads the 26.8 GiB table from disk by default, sharing the reader and
row cache across local GPUs;
`flare run --embedding-table-storage ram` keeps it in resident host memory.
The graph publishes GPU-produced lookup IDs before layer
0 and waits for row delivery at the layer-1 consumer. Disk and RAM delivery
use the same prepared model graphs; the flag selects table storage only.

## GLM 5.3 Flash

The `glm-5.3-flash` recipe uses `nvidia/GLM-5.3-Flash-FP4-Local` and supports
one, two, or four GPUs, including TP1 on DGX Station. Its logical context is
1,048,576 tokens; the default 4096-token prefill chunk controls graph execution
size independently of that context.

Projection formats follow the checkpoint's ModelOpt declarations. The target
decoder uses MXFP8 for KDA Q/K/V, dense MLPs, and the vocabulary head;
block-FP8 with FP32 scales for MLA projections and shared experts; and BF16 for
recurrent gates, indexer projections, and MLA key/value absorption. Compatible
projections are fused within each precision group. Normalization retains BF16
or FP32 checkpoint weights. Learned drafts use the target MXFP8 vocabulary head.

Image preparation includes the checkpoint's replicated vision tower and native
image processor. The tower merges four patches into each image feature, with
captured capacities from 64 through 32,768 patches. Each feature slot holds an
8192-feature window; the shared scheduler reuses that window as image prompts
advance. `--image-concurrency` defaults to the smaller of four and text
concurrency. `--language-model-only` omits image weights and graphs.

## DeepSeek V4.1 Flash

The `deepseek-v4.1-flash` alias selects `deepseek-ai/DeepSeek-V4.1-Flash-1017`
and supports TP3 and TP4 text generation, reasoning, DSML tool calls, and images
when the checkpoint contains its vision encoder.
TP3 partitions the 2304-wide routed and shared experts into 768-wide shards
without padding. Attention partitions whole output groups: padding eight groups
to nine gives each rank three groups and 24 heads. The last rank's extra group
has zero query and output weights; all checkpoint weights and scales remain
unchanged. Vocabulary shards contain 43136 rows each; the 128 extra gathered
logits are excluded before sampling. Engram's projection input columns divide its 24 heads evenly across three
ranks. Every host stores full Engram tables in checkpoint row order; local ranks
share table reads and bounded host caching. Graphs request canonical table rows,
and each rank selects its head columns for the partial projection before reduction.
Exports use the native DeepSeek V4.1 prompt encoder and include only the tokenizer
asset. Vision exports bind the atomic `<｜deepseek_image｜>` token as ID 129264,
which must agree with checkpoint `image_token_id`. Packages without explicit
prompt-encoder metadata continue to use their tokenizer and Jinja template.
Sampling defaults are temperature 1.0 and top-p 1.0, following the
[model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017#minimal-inference).
Target vocabulary projections keep the checkpoint's BF16 weights and produce
FP32 logits for generation, prompt scoring, and cached-prefix resume. Proposal
head precision is configured separately with `--draft-head-quantization`.
Preparation losslessly compresses backbone and DSpark MXFP4 expert scales with
CSF by default. Compressed scales stay resident, with shared
expansion buffers reused by serial expert layers. Planning conservatively estimates
3% less resident weight storage; loading measures the actual allocation.
`--expert-scale-compression native` preserves native scales. MLA arithmetic defaults
to BF16; `--mla-compute-mode auto` lets b12x select arithmetic while preserving
packed KV storage.
On SM120 and SM121, mHC's lagged pre and post/pre calls run through a TileLang
kernel pair. One CTA owns a hidden slice for every row, so the 24-wide projection
weights load once per call, and the projection runs on tensor cores as an FP16
high/low split. Mixing, collapse, and the Sinkhorn coefficients keep b12x's FP32
arithmetic, and the split reproduces the FP32 weights to 1e-7 relative. On GB10
one post/pre call costs 9.5 us at eight rows against b12x's tuned 14.7. The
packed planes add 2.5 MiB per mHC weight set and are shared by every prepared
shape that uses them.
The pair's grid is one CTA per hidden slice and each CTA loops row tiles
serially, so the slice width follows the part and the row count. A part with at
least 128 multiprocessors is already under-filled at 160 CTAs and keeps a
32-wide slice up to 256 rows; a part with 64 or fewer still fills at 80 CTAs, so
it keeps 32 up to 64 rows, leaves 128 to 1023 rows on b12x, and uses a 64-wide
slice from 1024 rows, where halving the split count halves the FP32 partials
and the finalize reduction. On GB10 that is 0.80-0.84 of b12x's cost at 7168
and 8192 rows, saving about 84-94 ms per 80-call prefill chunk, while a 32-wide
slice costs 33-49 ms more than b12x at those sizes. Calls below four rows, calls
without lagged coefficients, and calls with a row bound or an FP32 norm weight
keep b12x's fused launches.
DSpark is its learned draft method. Preparation defaults to seven draft tokens,
32 concurrent requests, a 4096-token prefill chunk, a 128-row physical target
capacity, and a budget of 64 packed round graphs. Positive-depth DSpark uses
packed GPU prefix selection automatically.
Each captured round contains proposal generation, GPU prefix allocation, target
verification, and accepted-state commit. Width-one graphs retain draft context
without generating proposals. Ordinary graphs remain available for native
`--mtp-draft-tokens 0`; `--no-mtp` omits draft weights during preparation.
Standalone proposal, verification, and joined wider speculative graphs are not
registered in the packed catalog. Image preprocessing preserves the checkpoint's
row-major patch layout and 1024-token image limit.

`--joint-tuning` enables joint selection of QKV projection, query normalization,
and query expansion plans. It covers every prepared text execution shape,
including prefill, ordinary decode, width-one continuations, packed target
verification, and DSpark proposals. Matching operator declarations share a
selection scored across their layer contexts; distinct declarations are tuned
independently. Batch capacity, physical token capacity, attention mode, and mHC
seams determine the measured contexts. The option defaults to off and requires
MXFP8 query normalization on the selected GPUs and `--autotune-effort default`
or `audit`. It adds preparation time without changing the serving graph catalog.

`--speculative-token-capacity` bounds physical target rows;
`--speculative-graph-budget` independently controls the number of packed rounds.
For each power-of-two request bucket `B`, the catalog retains capacities
`B + 1` and `min(token_capacity, B * (draft_tokens + 1))` when that interval is
nonempty. These endpoints preserve the smallest positive aggregate draft budget
and full-width verification wherever it fits. Width-one continuations cover
every request bucket independently and do not consume the packed graph budget.

Additional graphs refine the largest ratio between adjacent capacities across
all buckets. Each refinement inserts the integer nearest the geometric midpoint
that minimizes the larger resulting ratio. Equal ratios prefer smaller batches,
then smaller capacities. Comparing ratios gives each batch the same relative
resolution rather than assigning all graphs to the largest absolute row range.
At a fixed row cap, increasing the graph budget adds one distinct graph at a
time and preserves every graph already selected. Integer row capacities limit
the available resolution: larger budgets saturate at every valid capacity.
Budgets below the endpoint count fail before preparation; the default request
and token capacities require at least 12 packed graphs.

With seven drafts, 128 target rows, and a 64-graph budget, the catalog contains:

| Captured request capacity | Packed graphs | Minimum / maximum target rows |
|---|---:|---:|
| 1 | 7 | 2 / 8 |
| 2 | 10 | 3 / 16 |
| 4 | 12 | 5 / 32 |
| 8 | 12 | 9 / 64 |
| 16 | 14 | 17 / 128 |
| 32 | 9 | 33 / 128 |

These 64 combined packed rounds and six width-one continuations are separate
from ordinary decode, prefill, maintenance, and vision graphs. The persistent
preparation catalog prints every selected capacity. At batch 32, the 128-row
cap permits four target rows per request on average, while proposal generation
still produces seven drafts per request. Raising the row cap permits longer
aggregate verification prefixes; raising the graph budget improves resolution
within that cap and increases capture and retained graph resources.

The host owns admission, cohort membership, and cache reservations. The captured
round assigns prefixes from native proposal confidence and reports one terminal
completion. Native scheduling compares complete measured round cost and expected
emitted-token value across the packed and width-one candidates. Prefix limits
include output and context budgets, grammar constraints, and host-stop barriers.
Physical target padding executes at the captured row capacity.
Its 188.8 GiB of Engram tables are stored once in the package in checkpoint
row order. Every host needs the complete tables; local ranks share their
reader and host cache. Projection input columns are assigned round-robin by
hash head, so each TP4 rank computes two bigram, two trigram, and two four-gram
head contributions and reduces its partial projection before residual mixing.
Table rows and projection scales retain checkpoint precision.
Disk storage reads rows on demand through bounded direct-I/O staging;
`flare run --embedding-table-storage ram` loads the complete tables into
ordinary resident memory. Both modes copy compact rows into each rank's prepared
buffers and signal captured consumer events. Preparation appends the immutable
checkpoint spans without materializing the tables.
For disk storage, `--embedding-table-resident-scales` retains the original scale
bytes in RAM and removes scale reads on row-cache misses. This is enabled by
default on discrete GPUs and disabled when any selected GPU shares host memory.
Use `--no-embedding-table-resident-scales` to disable it explicitly. The scale
planes are shared across local ranks and add to the host's row-cache budget.
The loader reserves this memory before sizing KV storage.
Export with the recipe defaults on four available GPUs, then optionally create
a standalone packed file and serve it:

```sh
.venv/bin/ignition export deepseek-v4.1-flash --gpus 0,1,2,3
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/deepseek41-tp4.safetensors

./flare/target/release/flare run \
  /data/flare-runtime/deepseek41-tp4.safetensors \
  --gpus 0,1,2,3
```

For four DGX Sparks with one GPU per host, replace `--gpus 0,1,2,3` during
export with `--gpus 0 --ranks 4 --rendezvous 10.200.0.12:29630` and start an
`ignition join` process on each peer. Rank zero retains the checkpoint; peers
request their weight ranges during preparation. From a paired controller, use
`flare orbit run maxwell:/data/flare-runtime/deepseek41-tp4.safetensors --cluster sparks --gpus 0`
with a four-node `sparks` cluster. The prepared model uses RoCEnante collectives
across hosts. See [multi-host preparation and serving](multi-host.md) for
preparation authentication, orbit setup, and rank placement.

For three Sparks, use `--ranks 3` and join peer ranks 1 and 2. Launch the TP3
package with `flare orbit run NODE:PACKAGE --nodes NODE,PEER1,PEER2 --gpus 0`,
listing hosts in preparation rank order.

On GB10, a TP3 package with 32 concurrent requests, four image slots, seven
DSpark drafts, BF16 MLA, and CSF expert scales occupies about 104.7 GiB per
rank before physical cache backing. Use `--memory-limit=-1G` when launching
this configuration to retain a 1 GiB reserve. The integrated-memory planner
also preserves its separate 5% system allowance and 2 GiB host allowance.
A 4 GiB reserve leaves insufficient memory for continuation rows. Disk-backed
Engram tables retain the default 1 GiB row cache per host.

`--image-concurrency` defaults to the smaller of four and `--max-concurrency`.
For example, add `--max-concurrency 64` during export to
prepare 64 text request slots while retaining four independent image feature slots.
The native server can lower text concurrency at load time. Engram histories
exclude image positions, so n-grams cannot cross an image boundary.

DeepSeek V4.1's logical cache pages cover 128 tokens across compressed latent and
index regions with different byte sizes. Continuation rows retain sliding
windows, pending compression state, partial pages, DSpark context windows, and the
last hidden state. Both use runtime-sized virtual ranges; exported packages
contain their layouts and capacity bounds without cache payloads.
Engram indexing runs in the graph over committed history and candidate tokens.
The three preceding tokens live in continuation state, including reused prefixes.
DSpark proposals, target verification, and accepted-state commit use the shared
native speculative contract; Helios contains no DSpark model implementation.

The optional preparation flags `--draft-head-quantization nvfp4` and
`--markov-head-quantization nvfp4` select independent NVFP4 copies of the draft
vocabulary head and Markov output projection. Both use BF16 activations; the
Markov kernel's BF16 output is widened to FP32 before bias addition. Target
verification retains the checkpoint head. Omitting these flags preserves the
checkpoint precision. Each selection requires tuning, capture, and package export;
it cannot change an existing package at serving time. The same options apply to
the DeepSeek V4 Flash recipe. Quantization changes proposal probabilities and can
change acceptance and throughput; measure both separately and together.

DeepSeek V4.1 also accepts `--draft-head-quantization mxfp8` for an E4M3
proposal head with one power-of-two scale per 32 weights. The recommended
default is checkpoint precision: omit `--draft-head-quantization` to select it.
Use `--draft-head-quantization mxfp8` when reducing draft projection bandwidth
is useful while retaining BF16 activations, target checkpoint weight storage,
and the checkpoint Markov projection. It retains a separate quantized copy in
addition to the target weights, using more GPU memory. The initial weight
estimate excludes the extra copy;
preparation reports the actual allocation and sizes cache backing after loading.
It changes the proposal distribution and seeded text;
verification must use the actual proposal probabilities. Rebuild the package
when changing this preparation option.

```sh
.venv/bin/ignition export deepseek-v4.1-flash --gpus 0,1,2,3 \
  --draft-head-quantization mxfp8
```

DeepSeek V4.1 also accepts `--markov-head-quantization fp8`. It stores the draft
Markov output weights as E4M3 values with one FP32 reconstruction scale per
vocabulary row. The flag is independent of `--draft-head-quantization`; omit it
to retain checkpoint precision. Both the fused batch-one and wider prepared
Markov paths form FP32 transition scores, and the proposal draw and retained
rejection distribution use those same scores. Target vocabulary weights remain
BF16 and target logits remain FP32. Row-scaled FP8 is lossy draft arithmetic,
so seeded text and acceptance may change. To prepare NVFP4 draft vocabulary
weights together with FP8 Markov weights while retaining the recipe's other
defaults, run from the repository root:

```sh
.venv/bin/ignition export deepseek-v4.1-flash --gpus 0,1,2,3 \
  --draft-head-quantization nvfp4 --markov-head-quantization fp8
```

Use an ordinary export when first changing either head's weight format.
`--reuse-weights` trusts the previous package's weight bytes, while
`--no-weights` requires the exact changed payloads to be cached already.

`--vocabulary-parallel-sampling` keeps DeepSeek V4.1 proposal and verification
probabilities on their vocabulary-owning GPUs. Only normalization statistics,
selected-token probabilities, and residual winners cross ranks. The default is
disabled. DeepSeek V4.1 requires TP4 for this option; its TP3 gather is unavailable.
The option requires positive learned draft depth and supports BF16 or
row-scaled FP8 Markov weights; NVFP4 Markov weights are unsupported. Target
vocabulary weights and logits retain BF16 and FP32 precision. Full softmax and
greedy targets are supported; top-k, top-p, and min-p filtering must remain
disabled, as in the recipe's default policy. Raw generation logprobs and grammar
constraints remain available. Block rejection changes seeded output relative to
the full-vocabulary serial sampler.

For a TP4, single-request configuration with seven drafts, run:

```sh
.venv/bin/python -m ignition export deepseek-v4.1-flash --gpus 0,1,2,3 \
  --max-concurrency 1 --mtp-draft-tokens 7 --speculative-token-capacity 8 \
  --prefill-chunk-size 4096 --expert-scale-compression native --mla-compute-mode bf16 \
  --draft-head-quantization nvfp4 --markov-head-quantization fp8 \
  --vocabulary-parallel-sampling --joint-tuning --collective-tuning runtime \
  --collective-ladder ignition/preparation/ladders/gb202_pcie_tp4.json
```

Use a fresh weight export when first enabling this option: the Markov head's
stored vocabulary partition changes. Complete-model serving has been exercised
at TP4 with one concurrent request; wider serving configurations still need
model validation.

## DeepSeek V4 Flash

DeepSeek V4 Flash uses `local-inference-lab/DeepSeek-V4-Flash-Vision-Exp-lossless-CSF`, including its
vision encoder and checkpoint-integrated DSpark model. The `deepseek-v4-flash`
alias prepares five draft tokens by default, with adaptive verification. It
accepts images within its image-token budget and has no offloaded embedding tables.
Exports use the native DeepSeek V4 prompt encoder, carry only the tokenizer asset,
and bind the atomic `<｜deepseek_image｜>` placeholder as token 129264.
Both native DeepSeek encoders default to thinking enabled and high effort, accept
`low`, `high`, `xhigh`, and `max`, and reject `drop_thinking`, assistant
continuation, and `add_generation_prompt=false`. DeepSeek V4 rejects numeric
effort. DeepSeek V4.1 also accepts integer scores from 1 through 100 in
`reasoning_effort` or `chat_template_kwargs.reasoning_effort`; these are prompt
hints rather than token budgets. See [reasoning controls](api.md#reasoning-sampling-and-usage)
for request examples and named-level mappings. System-prefix caching
derives reusable token prefixes from the official encoder's rendered conversations.
Previously exported Jinja packages retain their
packaged rendering behavior and require both tokenizer and template assets.
Use two or four available GPUs for preparation and serving:

```sh
.venv/bin/ignition export deepseek-v4-flash --gpus 0,1,2,3
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/deepseek4-tp4.safetensors

./flare/target/release/flare run \
  /data/flare-runtime/deepseek4-tp4.safetensors \
  --gpus 0,1,2,3
```

V4 uses 256-token logical pages containing 64 C4 latent records with an FP8
retrieval index, or two C128 latent records with dense history attention. Each latent page
stores FP8/BF16 payloads followed by its own scale footer. Continuation rows
preserve the complete 128-token compression interval, C4 overlap, sliding
windows, partial pages, and DSpark context. Only accepted tokens update the
continuation state. Bidirectional image spans stay within one prefill chunk;
the scheduler shortens a preceding text chunk when necessary. Image preprocessing,
feature delivery, text execution, and output parsing all run natively in Helios.
Packages with 128-token V4 pages are incompatible and must be re-exported.

## MiMo V2.6 Pro MOPD

MiMo V2.6 Pro MOPD uses `XiaomiMiMo/MiMo-V2.6-Pro-MOPD` with checkpoint
MXFP4 experts, block-FP8 dense projections, and BF16 draft and vision weights.
The `mimo-v2.6-pro-mopd` alias prepares text, images,
and seven DFlash draft tokens with adaptive verification. Audio is not prepared.
Select eight available GPUs:

```sh
.venv/bin/ignition export mimo-v2.6-pro-mopd --gpus 0-7
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/mimo-v26-pro-mopd-tp8.safetensors

./flare/target/release/flare run \
  /data/flare-runtime/mimo-v26-pro-mopd-tp8.safetensors \
  --gpus 0,1,2,3,4,5,6,7
```

Global attention uses paged KV storage; continuation rows retain sliding-window
KV, partial pages, and accepted DFlash context. DFlash consumes selected target
layer features and predicts an eight-position block containing the pending
token and seven proposals. Helios uses the shared speculative execution contract;
draft modeling and state updates are captured by Ignition. The vision encoder
publishes compact image features for causal text prefill. Loading with
`--mtp-draft-tokens 0` selects ordinary execution without draft-state updates.

## Step-5 Preview

Step-5 Preview uses `stepfun-ai/Step-5-Preview-NVFP4`. The
`step5-preview-nvfp4` alias prepares text, images,
and three causal MTP draft tokens with adaptive verification. Checkpoint NVFP4
routed experts use W4A4. Routed experts declared BF16 in checkpoint metadata
are converted to NVFP4 during loading and use b12x W4A16 with BF16 activations;
the converted weights are retained in captured graphs and exported packages.
Shared, dense, and MTP MLP gate/up weights are packed and converted to MXFP8
during loading. Target-model projections select BF16 or quantized activations
through autotuning. MTP projections retain BF16 activations to bound rounding
error through the predictor feedback chain. Down and attention projections remain
BF16, with FP32 residual accumulation. Paged and sliding-window
K/V, including MTP windows, use FP8 E4M3 with unit descales;
sliding attention consumes live candidates in BF16 before committing them to FP8 history;
Q/K normalization and rotary computation retain their BF16 rounding before the
FP8 store. With CSF scales and shared scratch, the concurrency-16/MTP3 graph
catalog, including vision, fits on four RTX PRO 6000 Blackwell GPUs. This
memory-constrained example limits speculative target rows to 16 and selects
fixed-width serving, with width-one execution at batches that cannot fit wider
verification. The recipe defaults are concurrency 32, 128 target rows, and
adaptive verification:

```sh
.venv/bin/ignition export step5-preview-nvfp4 --gpus 0,1,2,3 \
  --max-concurrency 16 --no-mtp-adaptive-verification \
  --speculative-token-capacity 16 --expert-scale-compression csf
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/step5-tp4-mtp3-c16-fp8kv.safetensors

./flare/target/release/flare run \
  /data/flare-runtime/step5-tp4-mtp3-c16-fp8kv.safetensors \
  --gpus 0,1,2,3
```

`--expert-scale-compression csf` compresses all Step-5 routed-expert block
scales losslessly during loading. Both checkpoint W4A4 experts and converted
W4A16 experts use CSF, retaining their activation modes. Serialized layers share
one pair of expansion buffers, whose contents are zero-initialized on export.
One transient loading pool reuses conversion scratch across layers and retires
after weight loading, retaining live weights for graph export. The separate
graph-capture pool remains alive. Vision, prefill, decode, and draft activations
reuse typed graph buffers. MTP feedback, sampling, metadata, and resume scratch
reuse a separate buffer pool across graph variants, with disjoint buffers within
each graph. Sequential predictors share their norm-concatenation output. Kernel
workspaces use the shared operator arena, with separate slices for overlapping
expert branches. Compiler-created intermediates use one shared CUDA graph pool.
The memory plan estimates 6% lower resident weight storage; actual memory depends
on encoded exceptions and is reported after loading. Larger concurrency requires
separate preparation and memory validation.

Sliding attention keeps a 512-token continuation window. Sparse attention uses
128-token logical pages containing FP8 keys and values and an FP8 retrieval
index for eight-token regions. Continuation rows also preserve partial index
regions and accepted MTP state. Accepted target tokens, hidden states, and embeddings
remain in continuation storage until the following round consumes them in three
packed predictor passes. Candidate KV never enters committed predictor windows.
Region scoring, selection, and attention use
b12x kernels. Attention output gating is fused into the direct output or split
merge store, preserving BF16 rounding of attention and sigmoid before their
product. SwiGLU clamps the gate after SiLU. Helios manages these resources
through the shared cache and speculative execution contracts. The perception encoder
prepares 504-pixel local crops and a 728-pixel overview. Serving defaults follow
the model card: temperature 1, top-p 0.95, no top-k filter, and neutral penalties.

One-shot TP reductions within the channel's byte capacity fuse the residual
update and zero-centered RMSNorm into the collective. A bounded resident grid
retains row ownership across ranks while processing multiple rows per block.
The collective preserves FP32 residual storage and BF16 normalized outputs.
Attention and dense MLP projections enter the fused
collective directly as BF16; routed MLP updates enter as FP32. Attention and dense
MLP reductions round to BF16 before residual addition; routed MLP reductions retain
FP32. MLP reductions consume the following layer's input-normalization weight.
The final target and predictor MLPs
retain their separate final residual update. DMA prefill and hierarchical
collectives use separate reduction and normalization kernels.

## Kimi K3

Kimi K3 uses the local `/data/models/Kimi-K3-MXFP4-CSF` checkpoint. The
`kimi-k3-mxfp4-csf` alias prepares text generation, reasoning, XTML tool calls,
and images on sixteen GPUs. It preserves MXFP4 experts
with compressed scales, BF16 projections, and FP32 KDA recurrent state.
Its 93-layer decoder combines KDA, unrotated latent attention, attention
residuals, and latent SiTU experts. Red Hat DSpark uses
`RedHatAI/Kimi-K3-speculator.dspark` revision
`38a88101e0d46bb22134b9da340f381b954d40d4`, downloaded to the Hugging Face hub
cache (or `[paths].hub_cache`), and defaults to seven draft tokens per round.
A target checkpoint may instead carry that draft in its `dspark/` directory;
`kimi-k3-dspark` selects this layout at `/data/kimi-k3-dspark`.

For Kimi K3, `--online-mxfp8` converts KDA input, forget-gate, and output
projections; MLA latent/gate, query, and output projections; shared
expert gate/up and down projections; and both tensor-parallel layouts of the
MoE latent down projection and its latent up projection. Activations remain
BF16. MLA's batched key/value projections, routers, the first dense MLP,
embeddings, vocabulary head, and vision encoder retain checkpoint
precision. DSpark converts its context projection, attention outputs, and MLPs.
Shared and routed experts keep disjoint
kernel scratch when their execution overlaps.

This example limits concurrency to four and enables online MXFP8 conversion;
image concurrency follows the four-request limit:

```sh
.venv/bin/ignition export kimi-k3-mxfp4-csf --gpus 0-15 \
  --max-concurrency 4 --online-mxfp8
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/kimi-k3-mxfp4-csf-tp16-c4-images-redhat-dspark-7-mxfp8.safetensors

./flare/target/release/flare run \
  /data/flare-runtime/kimi-k3-mxfp4-csf-tp16-c4-images-redhat-dspark-7-mxfp8.safetensors \
  --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

The draft consumes AttnRes mixtures entering target layers 24, 48, 72, 88,
and 92, before input normalization. Its five causal Qwen3 layers use
2,048-token BF16 key/value windows. An anchor followed by mask tokens produces
seven proposals; each preceding proposal conditions the Markov bias.
Ignition uses greedy proposals and adaptive target verification widths,
without the draft's confidence head. The
[Red Hat model card](https://huggingface.co/RedHatAI/Kimi-K3-speculator.dspark)
reports eight-token probabilistic proposals with block rejection sampling,
which differs from this execution configuration. Only accepted target features
update draft windows; KDA verification commits the accepted recurrent prefix.
Loading with `--mtp-draft-tokens 0` selects ordinary execution; exporting with
`--no-mtp` excludes draft weights and state.

Target FP8 MLA latent records occupy 128-token pages.
With context parallelism across sixteen ranks, concurrency four, and a
4096-token prefill chunk, preparation backs ten logical KV pages and nine
continuation rows: one null row, four live rows, and four snapshot destinations.
Retaining CUDA graph handles for export does not expand this working set.
Native package loading sizes serving caches from available device memory;
in-process diagnostic serving limits admission to physically backed storage.
Continuation rows preserve KDA
convolution and recurrent state, DSpark key/value windows, partial latent pages, and the normalized
last hidden state. The MoonViT encoder publishes hidden-column-sharded features
for causal text prefill, with storage for 65,536 patches per active feature
window. Conversations can consume multiple windows through chunked prefill.
XTML encoding distinguishes protocol tokens from literal marker text.
Serving defaults follow the model card's agentic sampling recommendation:
temperature 1, top-p 1, no top-k filter, and neutral penalties. Reasoning is
always enabled; requests to disable thinking are ignored without an error.
Effort defaults to `max`; requests can select `low`, `high`, or `max` and
override sampling defaults. Preserve assistant reasoning in conversation history.

## MiniMax M3.1 Flash

The `minimax-m3-1-flash` alias selects
[`MiniMaxAI/MiniMax-M3.1-Flash`](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)
at revision `9e7ebacac350497ad4a1272a2e817639bf11561c`. Preparation uses four
SM120/SM121 GPUs, MXFP8 projections, NVFP4 routed experts, and FP8 target,
indexer, and DSpark caches. Sampling defaults are temperature 1.0 and top-p
0.95, following the model card.

DSpark defaults to 32 concurrent requests, five draft tokens,
a 128-row physical target capacity, a 64-graph packed budget, and GPU causal
prefix allocation. Each packed graph generates proposals, allocates prefixes,
verifies the admitted tokens, and commits accepted state. Width-one graphs
preserve draft context. `--speculative-token-capacity` and
`--speculative-graph-budget` control the same shared catalog geometry described
under DeepSeek V4.1.

Ignition applies `ignition/recipes/minimax_m3/assets/chat_template.patch.json`
to the downloaded checkpoint template. The annotated result preserves the pinned upstream template byte for byte after removing message
marks. It retains the upstream adaptive-thinking behavior and XML tool syntax.
Vision preparation includes temporal patch duplication and spatial patch
merging; `--language-model-only` omits the vision encoder. Optional
`--expert-scale-compression csf` losslessly compresses expert scales and shares
their expansion scratch across routed layers. Planning retains the native
weight budget for this encoding; loading reports its actual allocation.

```sh
.venv/bin/ignition export minimax-m3-1-flash --gpus 0-3
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/minimax-m3.1-flash-tp4.safetensors
flare/target/release/flare run /data/flare-runtime/minimax-m3.1-flash-tp4.safetensors --gpus 0,1,2,3
```

## Muse Spark Open

The `muse-spark-open-nvfp4` alias selects
[`meta-models/Muse-Spark-Open-816B-A42B-NVFP4`](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4)
at revision `4239e4db19376904f2e82c167e78f70abb5455fe`. Text preparation uses
eight SM120 GPUs, checkpoint NVFP4 experts, FP8 E4M3 KV storage, a 262144-token
context, four concurrent requests, and 4096-token prefill chunks. Target and draft
queries and attention outputs use BF16. Image, audio, and video are not prepared.

DFlash2 drafting uses
[`nvidia/Muse-Spark-Open-816B-A42B-DFlash2`](https://huggingface.co/nvidia/Muse-Spark-Open-816B-A42B-DFlash2)
at revision `fbec887a79866e9ecff6c57adfd9fd02aa3cadea`. Preparation downloads
its configuration and weights into the configured Hugging Face hub cache. The
five-layer BF16 draft uses eight-token blocks with seven proposed tokens, target
features from layers 3, 19, 31, 47, and 59, and 2048-token sliding attention.
Target and draft histories use unit-scale FP8 E4M3 storage. The default adaptive
verification catalog uses width-one and width-eight graphs with per-request
prefixes of one through eight tokens and 32 packed target-token rows.
`--mtp-draft-tokens N` selects zero through seven proposals;
`--no-mtp-adaptive-verification` fixes the verification width. `--no-mtp` skips
the draft download, weights, state, and speculative graphs.

At tensor parallelism eight, draft weights occupy 1,596,065,024 bytes per rank,
including replicated convolution and candidate-selection tensors and fused context
projections. Each continuation row occupies 27,525,120 bytes with draft history
or 24,903,680 bytes with `--no-mtp`; each 128-token paged KV block occupies
524,288 bytes. These cache sizes exclude transient graph scratch.

Sampling follows the checkpoint model card: temperature 1.0, top-p 1.0, and
top-k disabled. The checkpoint generation configuration's top-k 64 is not the
recommended inference default. Reasoning effort accepts `low`, `medium`,
`high`, and `xhigh`, with `xhigh` as the template default. Set
`reasoning_effort` or `chat_template_kwargs.reasoning_effort` explicitly to
override it.

Muse Spark preparation requires FlashInfer/b12x operators for unweighted NVFP4
routed-expert outputs, independent FC1 half gains, and head-dimension-64 paged
decode with BF16 queries and FP8 E4M3 caches, including quantized sliding-cache
verification and rollback. The FlashInfer Git dependency pinned in
`pyproject.toml` lacks these operators. Complete the standard environment setup and install the local
checkout's build and runtime requirements into `.venv` first. Build requirements
include `setuptools>=77`, `packaging>=24`, and `apache-tvm-ffi>=0.1.11,<0.2`;
the CUDA 13 checkout's NCCL-EP dependency requires `nvidia-nccl-cu13>=2.30.7`.
Install the checkout containing the required operators with:

```sh
FLASHINFER_BUILD_NO_PIP=1 uv pip install --python .venv/bin/python \
  --no-deps --no-build-isolation -e /path/to/flashinfer
```

`FLASHINFER_BUILD_NO_PIP=1` disables nested dependency installation and NCCL
upgrades in the FlashInfer build hook. `--no-build-isolation` uses the build
requirements already installed in `.venv`.

Running `uv sync --locked --extra prepare --extra test` restores the pinned Git
dependency, so repeat the editable installation afterward when preparing Muse
Spark. The local checkout must provide both the `flashinfer` and `b12x` imports.

Ignition applies `ignition/recipes/muse_spark/assets/chat_template.patch.json`
to the downloaded checkpoint template. The annotated result preserves the canonical checkpoint bytes when message marks are removed.
Native recipient headers route `self` messages to reasoning, `commentary` and
`user` messages to visible content, and tool recipients through ATEM function
calls. `<|eom|>` separates messages within a generated turn; `<|eot|>` and
`<|end_of_text|>` end generation. Tool arguments in request history are normalized
to JSON objects before rendering. Caller content and header attributes retain
the canonical control-sequence escaping.

```sh
.venv/bin/ignition export muse-spark-open-nvfp4 --gpus 0-7 --prompt-logprobs
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/muse-spark-tp8.safetensors
flare/target/release/flare run /data/flare-runtime/muse-spark-tp8.safetensors --gpus 0,1,2,3,4,5,6,7
```

## Storage and graph capacities

Gemma 4 and Muse Spark default to four concurrent requests. Gemma 4's speculative
target-token capacity is the larger of padded concurrency and draft count plus
one, giving six rows at its default settings. Muse Spark declares 32 packed
target-token rows for seven DFlash2 drafts at concurrency four. Other recipes
default to 32 concurrent requests, with speculative recipes declaring 128 target-token rows independently
of concurrency.
Explicit flags and configuration override these defaults; existing packages retain their
captured capacities. Memory planning rejects capacities that do not fit the
selected GPUs.

| Recipe | Draft tokens | Verification layout | Speculative graphs at default concurrency | Maximum drafts at batch 32 |
|---|---:|---|---:|---:|
| Qwen3.8 Flash, Qwen3.8 27B, GLM 5.3 | 5 | Rectangular | 34 | 3 |
| Step-5 | 3 | Rectangular | 24 | 3 |
| DeepSeek V4, MiniMax M3.1 | 5 | Packed | 64 packed + 6 width-one | Up to 5, sharing 128 rows across requests |
| DeepSeek V4.1, MiMo V2.6 Pro, Kimi K3, Nemotron Bolt | 7 | Packed | 64 packed + 6 width-one | Up to 7, sharing 128 rows across requests |
| Gemma 4 | 5 | Rectangular | 10 at concurrency 4 | Not prepared at default concurrency |
| Muse Spark | 7 | Packed | 49 packed + 3 width-one at concurrency 4 | Not prepared above concurrency 4 |

The 128-row rectangular catalogs cover full draft depth through batch 16.
Gemma 4's six-row default covers five drafts at batch 1 and two at batch 2,
with width-one execution at batch 4. The counts include
width-one speculative graphs and exclude ordinary decode, prefill, maintenance,
and vision graphs. Capturing full rectangular depth at batch 32 requires 192 rows
for five drafts or 256 rows for seven drafts, set with
`--speculative-token-capacity`. Every packed recipe's round graph count is bounded
independently by `--speculative-graph-budget`.

Every model export prints a VRAM inventory for each rank after final publication,
including exports without `--verbose`. Heliopause measures device usage and package
allocator mappings; Ignition attributes retained ranges to checkpoint tensors,
scratch pools, graph temporaries, I/O and persistent state, communication buffers,
kernel globals, and allocator slack. Owner fields and defining source locations
identify the largest consumers in each category. Aliased ranges count once;
shared scratch displays representative tensor owners.

Preparation-only storage is separate from fixed package storage. Cache figures
count physically backed pages, not their virtual serving capacity. Mapped host
storage and imported peer mappings are listed outside local VRAM totals. PyTorch
allocated and reserved bytes overlap the attributed storage. Cubin data sizes are
not resident code sizes: graph executable overhead, kernel code, other driver
allocations, and other processes remain in the unattributed device-usage remainder.
The returned `RuntimePackage.memory_inventory` contains every attributed source
group; console output shows the largest five per category. Source locations name
owning classes or closures, not allocation stack traces.

Ignition commands accept `--config PATH`; the default configuration is
`~/.config/flare/config.toml`. Storage locations are configured with
`[paths] hub_cache`, `kernel_cache`, and `package_cache`; `[memory] reserve`
leaves GPU memory available for other processes during preparation and serving.
Ignition additionally budgets 1 GiB for warm-up and capture, plus 128 KiB per
effective prefill token above 4096, including mixed prompt/decode rows.
Preparation KV backing covers a prefill chunk plus boundary pages per captured request row, unless an
explicit preparation page count is supplied programmatically. Continuation
backing covers the null row, live request and verification rows, and snapshot
destinations used during preparation. Planned prefix-cache slots remain virtual;
they do not require physical backing during export. The memory report separates
the declared continuation extent from `preparation_state_rows` and
`preparation_state_bytes` where those extents differ. Both caches retain their
full virtual extents in the package. Helios
sizes its caches after uploading the selected graphs, keeping a separate 64 MiB
driver margin.
At package load time, `flare run --kv-cache-limit context` bounds the shared
KV pool by one prepared context; `concurrency` uses one context per selected
active request, and the default `memory` fills the available budget. Runtime
`--memory-limit` accepts an occupied size (`30G`), free reserve (`-30G`), or
fraction of total device memory (`0.8`). It replaces the exported serving
reserve and includes model resources as well as caches. Both flags apply at
package load time without another export. See
[runtime memory limits](cli-design.md#runtime-memory-limits) for rounding,
minimum capacities, and integrated-memory accounting. Preparation retains its
own existing memory policy.

Before compilation and capture, preparation prints a persistent graph catalog
with counts, batch and prefill capacities, logical widths, physical target rows,
and draft depths, including when `--verbose` is omitted.
Ignition captures power-of-two
batch capacities through the concurrency limit
and width one at every bucket. Rectangular speculative and verification shapes
require `batch * width <= speculative_token_capacity`; the preparation flag
`--speculative-token-capacity` defaults to 128 physical target rows except for
Gemma 4's derived capacity. Packed recipes use the shared capacity selection
described under DeepSeek V4.1 rather than rectangular `batch * width` shapes.
Previously exported separate-proposal
catalogs cover every bucket and load-time draft depth independently of that limit.
Flare can reduce concurrency at load time. Rectangular catalogs permit smaller
positive draft limits; packed catalogs require their captured positive depth or
zero. Prefill capacity stays prepared.
With `--mixed-prefill`, Qwen also captures mixed prompt/decode graphs: every
prompt bucket at width one and one balanced prompt bucket at each wider
verification depth, for each batch capacity of at least two. The ordinary and
draft-state-maintaining catalogs remain distinct. Mixed capture is disabled by
default and independent of the MTP flags.
See [mixed graph coverage](scheduling-runtime-design.md).

## Serving runtime packages

Load an exported `.safetensors` package directly with the native CLI:

```sh
./flare/target/release/flare run /data/flare-runtime/qwen38-tp2-mtp5.safetensors \
  --gpus 0,1
curl --fail http://127.0.0.1:8000/health
```

Select a subset of a package's captured execution capacities:

```sh
./flare/target/release/flare run /data/flare-runtime/qwen38-tp1-mtp5.safetensors \
  --gpus 2 --max-concurrency 2 --mtp-draft-tokens 3 \
  --no-mtp-adaptive-verification
```

`--mtp-draft-tokens N` limits each round to N drafts plus its pending input.
`--mtp-draft-tokens 0` selects ordinary prefill, decode, and prefix-cache graphs
without draft execution or draft-state updates. Every speculative package also
contains these ordinary graphs. Adaptive width-one rounds retain draft state so
later rounds can resume speculation. Ignition's `--no-mtp` excludes the draft model
during preparation. `--mtp-adaptive-verification` selects among the loaded
widths; `--no-mtp-adaptive-verification` selects the fixed width where captured,
with width-one execution available at larger batch capacities. Omitted flags
use the package's serving defaults. Limits cannot exceed captured capacity;
missing required batch/width combinations fail before GPU initialization.
Concurrency three requires the batch-four bucket. Only selected graphs and
required kernel functions are loaded. Weights, MTP weights, and transport buffers
retain their prepared layouts. Continuation rows and KV pages occupy separate
fixed virtual ranges with declared maximum extents. Helios backs those ranges
according to selected concurrency, verification width, and available GPU memory;
reducing execution state leaves more physical memory for KV pages.

The selected GPU count must match the package, and each GPU must match its
rank's architecture and memory requirements. Flare serves all ranks in one
process with one execution thread per rank. Its dashboard runs on an
interactive terminal. Use `--api-key-file PATH` to supply a serving API key;
exported packages do not contain the key. Stop Flare with Ctrl-C or SIGTERM
so native replay drains before CUDA resources are released.

## DGX Spark

DGX Spark's GB10 is SM121 and shares physical DRAM with the CPU. Prepare an
SM121 package on Spark; an SM120 package does not satisfy the architecture
check. GPU weights, KV, registered host buffers, resident embedding tables,
compiler workers, and the OS compete for that memory. Set `[memory] reserve`
for host activity as well as driver headroom; `reserve = "4G"` leaves four GiB
outside the model's cache budget. RAM embedding tables load before cache
sizing. Disk embedding tables use bounded staging and can retain their scale
planes in RAM with `--embedding-table-resident-scales`. The loader accounts for
resident scales and the optional row cache before sizing KV, for both PLE and
Engram packages. Disk tables default to a 1 GiB host-RAM row cache per Spark;
`--embedding-table-cache-size` overrides it, and `0` disables it.

Spark's NVML interface does not report VRAM capacity. Ignition identifies an
integrated device through CUDA and uses Linux `MemAvailable`, capped by CUDA's
device total, for planning. Qwen preparation and Helios use the same available
RAM budget when allocating weights and caches; swap is not GPU capacity.
CUDA free-memory queries exclude reclaimable page cache on GB10. Compiler worker
budgets account for both CPU and GPU allocations in shared RAM. Helios sizes
its caches independently after loading.
VMM allocations and registered host mappings remain distinct CUDA
memory types despite sharing DRAM; do not dereference a device allocation on
the CPU or replace system-scope synchronization with volatile accesses. See
[NVIDIA's Spark CUDA guidance](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html).

A model too large for one Spark runs tensor-parallel across two or four Sparks,
one rank per host, reducing through b12x RoCEnante over their ConnectX-7 ports:
`ignition export --ranks N --rendezvous ADDR:PORT` on rank zero with
`ignition join` on each peer writes one package on rank zero, and
`flare orbit run NODE:PACKAGE --cluster NAME` serves it from any machine,
starting every rank through its host's `flare agent`. See
[serving across DGX Sparks](multi-host.md) for agent installation, pairing,
and failures.

## Package and development serving

Export a package, then launch the native server:

```sh
.venv/bin/ignition export qwen3.8-flash --gpus 0,1
.venv/bin/ignition convert CACHE_ENTRY --form packed --output qwen.safetensors
flare/target/release/flare run qwen.safetensors --gpus 0,1
```

To test a serving-only Rust change, rebuild Flare and load an existing compatible
package. Re-export when changing prepared kernels, capacities, or the package
format. Export destinations must not already exist.

Use `.venv/bin/ignition export --help` for preparation flags and
`flare/target/release/flare run --help` for serving flags. See [the runtime package design](runtime-package-design.md)
for resource ownership and [the CLI design](cli-design.md) for configuration.

For fault diagnosis, `ignition debug-serve` hands prepared graphs directly to
Helios without exporting a package. Its native path requires Heliopause's
`qualification` Cargo feature. All debug modes, including `--direct` and
`--eager`, are unsuitable for actual serving or apples-to-apples performance
comparisons against other runtimes; use an exported package with `flare run`.
Select idle devices explicitly; see [the debug launch example](cli-design.md#development-without-package-export)
for the extension build and diagnostic commands.
