# Model preparation and serving

## Runtime packages

Run commands from the repository root. Ignition prepares a source checkpoint
into a **runtime package**: a metadata-only primary `.safetensors` file and
immutable sibling payload blobs named by their SHA-256 hashes. The primary's
`flare.metadata` contains the signed Protobuf envelope; `flare.layout` maps its
logical resources to the blobs. `flare.payload` is empty. Weights, embedding
tables, graphs, kernels, and frontend assets all reside in payload blobs.
Adjacent weight resources owned by one rank share shards up to the file-size
limit. Graphs, kernels, and supporting resources use separate shards so changing
them does not rewrite unchanged weight shards. Rank ownership and logical file
boundaries remain separate for selective distribution. Export, conversion, and
signing use the same grouping rule.

The published payload-file limit is 64 GB (64,000,000,000 bytes), including the
header. The reader also accepts existing payloads up to 64 GiB. `--shard-size SIZE`
overrides `[publishing] shard_size` with a smaller positive limit. Resources
remain contiguous; export rejects a resource that exceeds the limit. Existing
monolithic and numbered-shard packages remain loadable.

Exports hardlink unchanged blobs from `paths.package_cache`. `ignition export
--no-weights` requires exact cached stored-data payloads and creates only missing
execution-resource blobs and metadata. Preparation and weight reads still occur.
Recipes can declare `export_repository`; without `--output`, exports use a unique
snapshot directory under that repository's local cache namespace. See
[package snapshots](cli-design.md#package-snapshots) for conversion and upload.

Tensor-parallel packages contain one resource bundle per rank.
Final packages use a format-4 signed envelope and encrypted cubins and graph
records. Ignition creates a per-user development signing key when `[publishing]`
is absent. `flare run` needs the corresponding publisher profile and first-use
acceptance. See [signing and development](cli-design.md#signing-and-local-development).
Unsigned rank staging is private to preparation and cannot load in Helios.

Flare loads the package files without Python. Helios is a Rust library;
`flare_heliopause` is a CPython extension imported by Ignition.

Every recipe defaults to FP8 E4M3 KV storage, including learned draft-model
caches. Qwen Flash retains BF16 selector keys; GLM and DeepSeek retain their
packed latent formats, including scale metadata and any BF16 rotary fields.
Recurrent state, compressor accumulators, and hidden activations retain their
model-specific precision. Cache writers and attention readers use matching
scales, and memory planning includes FP8 pages and partial-page snapshots.
Export a package to use these cache layouts; existing packages retain the
layouts captured when they were prepared.

## Environment setup

Set up the preparation environment and build the native CLI:

```sh
uv sync --locked --extra prepare --extra test
cargo build --release --manifest-path flare/Cargo.toml
```

Build dependencies include a C/C++ compiler, CMake, Ninja, pkg-config, Protobuf's
`protoc`, OpenSSL headers, Python headers, and liburing headers. On Ubuntu:

```sh
sudo apt-get install build-essential cmake ninja-build pkg-config protobuf-compiler libssl-dev python3-dev liburing-dev
```

Windows native builds also require Perl for the bundled static OpenSSL build,
alongside the C/C++ toolchain. The resulting binaries do not require OpenSSL DLLs.

The JPEG decoder builds its bundled libjpeg-turbo source. Preparation requires
the CUDA toolkit, including `nvcc` and CUPTI; configure `[paths] cuda` when it
is not supplied by the Python environment.

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

The preparation dependency pins a Git revision of `local-inference-lab/flashinfer`
in `pyproject.toml`; `uv.lock` records the same commit. To change it, update the
revision, run `uv lock --upgrade-package flashinfer-python`, then sync the environment.
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

`--autotune-effort off|low|medium|high|max` controls b12x candidate filtering.
`off` prepares heuristic choices without tuning races or tuned-selection cache
lookups. `low` applies the broadest performance exclusions; `medium` is the
default search policy; `high` keeps only conservative performance exclusions;
`max` enumerates every declared candidate satisfying correctness, resource,
and explicit configuration constraints. Each searched level includes the
lower levels' candidates. Kernel families retain their declared tuning axes.

Without extra tuning replicas, tensor-parallel ranks on identical GPUs divide
each kernel race's candidates and adopt the fastest measured configuration.
This applies within one host as well as across hosts. Fixed collective kernels
still prepare on every participating rank.

Set `[preparation] autotune_effort = "high"` for a persistent default; an
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

Choose an unused output filename; export appends `.safetensors` if needed and
refuses to overwrite one. Do not
delete an existing package merely to reuse its name. `MODEL` can also be a
supported repository ID or a local source-checkpoint directory. Ignition
downloads missing source weights, loads and tunes kernels, compiles forwards,
warms and captures execution variants, writes the package, and exits.
Compilation is enabled by default; `--no-compile` is for development.

## Nemotron 3.5 Bolt 74B-A9B 24GB

`nemotron-bolt-24gb` prepares
`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` on one SM120/SM121 GPU.
The recipe includes the C-RADIO encoder for still images. Video is not prepared. LiLiCorr uses the matching
`NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` checkpoint and defaults to
seven draft tokens with adaptive verification. Both checkpoints use the standard
Hugging Face hub cache (`~/.cache/huggingface/hub`), or `[paths].hub_cache` when configured.

```sh
.venv/bin/python -m ignition export nemotron-bolt-24gb --gpus 0 \
  --max-concurrency 8 --prefill-chunk-size 4096 --mtp-draft-tokens 7 \
  --speculative-token-capacity 64 \
  --output /data/flare-runtime/nemotron-bolt-images-lilicorr-c8.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/nemotron-bolt-images-lilicorr-c8.safetensors \
  --gpus 0 --bind 127.0.0.1:8001
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
`--max-concurrency 32 --speculative-token-capacity 256`.

## Qwen3.8 27B

`qwen3.8-27b` prepares `nvidia/Qwen3.8-27B-NVFP4` for text generation on one GPU.
It uses the canonical Qwen3.5 dense decoder with tensor-scaled FP8 attention
projections and NVFP4 MLP and vocabulary weights. FP8 KV pages and Gated DeltaNet
continuations reside in device memory. The checkpoint's BF16 MTP layer shares
the target embedding and vocabulary head. Speculation defaults to five draft
tokens, supports up to seven, and commits accepted GDN corrections into one
recurrent row per request. Draft KV trails target consumption by one token;
prefix snapshots preserve its partial page separately. The recipe has no
host-mapped tables or vision graphs. Prepare on the same GPU architecture as
the serving device:

```sh
.venv/bin/ignition export qwen3.8-27b --gpus 0 --max-concurrency 4 \
  --output /data/flare-runtime/qwen38-27b-tp1-c4.safetensors --verbose
```

Use `--mtp-draft-tokens N` to select the prepared draft depth. Width-one rounds
retain draft context so adaptive verification can resume wider rounds.
`--no-mtp` omits the draft weights and state during preparation;
`flare run PACKAGE --mtp-draft-tokens 0` selects the package's ordinary graphs.

## Qwen3.8 Flash

Enable hybrid history drafting at serving startup with
`flare run PACKAGE --history-drafts`; disable it with `--no-history-drafts`.
These options use the package's existing MTP graphs and are not export settings.

Export Qwen with up to five MTP draft tokens per round:

```sh
.venv/bin/ignition export qwen3.8-flash --gpus 2 \
  --mtp-draft-tokens 5 --max-concurrency 32 \
  --output /data/flare-runtime/qwen38-tp1-mtp5.safetensors --verbose

.venv/bin/ignition export qwen3.8-flash --gpus 0,1 \
  --mtp-draft-tokens 5 --max-concurrency 32 \
  --output /data/flare-runtime/qwen38-tp2-mtp5.safetensors --verbose
```

Qwen defaults to 32 concurrent requests, five MTP draft tokens, and 128 target-token
rows, with explicit flags and configuration taking precedence. Verification is
rectangular: the default catalog covers five drafts per request through batch 16
and three drafts at batch 32. Already exported packages retain their captured
capacities.

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
Helios reads the 26.8 GiB table from disk by default on every GPU;
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
logits are excluded before sampling. Engram's 24 heads divide evenly across
three ranks.
Exports use the native DeepSeek V4.1 prompt encoder and include only the tokenizer
asset. Vision exports bind the atomic `<｜deepseek_image｜>` token as ID 129264,
which must agree with checkpoint `image_token_id`. Packages without explicit
prompt-encoder metadata continue to use their tokenizer and Jinja template.
Sampling defaults are temperature 1.0 and top-p 1.0, following the
[model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017#minimal-inference).
Preparation losslessly compresses backbone and DSpark MXFP4 expert scales with
CSF by default. Compressed scales stay resident, with shared
expansion buffers reused by serial expert layers. Planning conservatively estimates
3% less resident weight storage; loading measures the actual allocation.
`--expert-scale-compression native` preserves native scales. MLA arithmetic defaults
to BF16; `--mla-compute-mode auto` lets b12x select arithmetic while preserving
packed KV storage.
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
Its 188.8 GiB of Engram tables are stored once in the package, partitioned by
hash head across TP ranks. Heads are assigned round-robin, so each TP4 rank
owns two bigram, two trigram, and two four-gram heads, with their projection
input columns. Table rows and projection scales retain checkpoint precision.
Each rank reads its table rows, computes a partial
projection, and reduces the result before residual mixing. Multi-host package
distribution transfers only the table partitions owned by the receiving rank.
Disk storage reads rows on demand through bounded direct-I/O staging;
`flare run --embedding-table-storage ram` loads the assigned partitions into
ordinary resident memory. Both modes copy compact rows into each rank's prepared
buffers and signal captured consumer events. Preparation appends the immutable
checkpoint slices without materializing the tables. Packages containing shared
table planes retain their prepared replicated computation and shared row delivery.
For disk storage, `--embedding-table-resident-scales` retains the original scale
bytes in RAM and removes scale reads on row-cache misses. DeepSeek V4.1 Flash
TP4 uses about 1.4 GiB per rank for these scales, in addition to any row-cache
budget. The loader reserves this memory before sizing KV storage.
Select four available GPUs explicitly:

```sh
.venv/bin/python -m ignition export deepseek-v4.1-flash --gpus 0,4,6,8 \
  --max-concurrency 32 --mtp-draft-tokens 5 --prefill-chunk-size 4096 \
  --speculative-token-capacity 128 --speculative-graph-budget 64 --expert-scale-compression csf \
  --output /data/flare-runtime/deepseek-v41-flash-1017-csf-tp4-c32-dspark5.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/deepseek-v41-flash-1017-csf-tp4-c32-dspark5.safetensors \
  --gpus 0,4,6,8 --host 127.0.0.1 --port 8001
```

For four DGX Sparks with one GPU per host, replace `--gpus 0,4,6,8` during
export with `--gpus 0 --ranks 4 --rendezvous 10.200.0.12:29630` and start an
`ignition join` process on each peer. Rank zero retains the checkpoint; peers
request their weight ranges during preparation. From a paired controller, use
`flare orbit run maxwell:/data/flare-runtime/deepseek-v41-flash-1017-csf-tp4-c32-dspark5.safetensors --cluster sparks --gpus 0`
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
For example, add `--max-concurrency 64 --image-concurrency 4` during export to
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
.venv/bin/python -m ignition export deepseek-v4-flash --gpus 3,4,5,6 \
  --max-concurrency 4 --mtp-draft-tokens 5 --prefill-chunk-size 4096 \
  --output /data/flare-runtime/deepseek-v4-flash-vision-tp4-c4-dspark5.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/deepseek-v4-flash-vision-tp4-c4-dspark5.safetensors \
  --gpus 3,4,5,6 --host 127.0.0.1 --port 8001
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
.venv/bin/ignition export mimo-v2.6-pro-mopd --gpus 0,1,2,3,4,5,6,7 \
  --max-concurrency 4 --mtp-draft-tokens 7 --prefill-chunk-size 4096 \
  --output /data/flare-runtime/mimo-v26-pro-mopd-tp8-c4-dflash7.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/mimo-v26-pro-mopd-tp8-c4-dflash7.safetensors \
  --gpus 0,1,2,3,4,5,6,7 --host 127.0.0.1 --port 8001
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
catalog, including vision, fits on four RTX PRO 6000 Blackwell GPUs:

```sh
.venv/bin/ignition export step5-preview-nvfp4 --gpus 0,1,2,3 \
  --max-concurrency 16 --mtp-draft-tokens 3 --no-mtp-adaptive-verification \
  --speculative-token-capacity 16 --prefill-chunk-size 4096 --expert-scale-compression \
  --output /data/flare-runtime/step5-tp4-mtp3-c16-fp8kv.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/step5-tp4-mtp3-c16-fp8kv.safetensors \
  --gpus 0,1,2,3 --host 127.0.0.1 --port 8001
```

Add `--expert-scale-compression csf` to compress all Step-5 routed-expert block
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

```sh
.venv/bin/python -m ignition export /data/models/Kimi-K3-MXFP4-CSF --gpus 0-15 \
  --max-concurrency 4 --image-concurrency 4 --mtp-draft-tokens 7 \
  --prefill-chunk-size 4096 --compile-workers 16 --online-mxfp8 \
  --output /data/flare-runtime/kimi-k3-mxfp4-csf-tp16-c4-images-redhat-dspark-7-mxfp8.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/kimi-k3-mxfp4-csf-tp16-c4-images-redhat-dspark-7-mxfp8.safetensors \
  --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --host 127.0.0.1 --port 8001
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

The annotated template in `ignition/recipes/minimax_m3/assets/chat_template.jinja`
preserves the pinned upstream template byte for byte after removing message
marks. It retains the upstream adaptive-thinking behavior and XML tool syntax.
Vision preparation includes temporal patch duplication and spatial patch
merging; `--language-model-only` omits the vision encoder. Optional
`--expert-scale-compression csf` losslessly compresses expert scales and shares
their expansion scratch across routed layers. Planning retains the native
weight budget for this encoding; loading reports its actual allocation.

```sh
ignition export minimax-m3-1-flash --gpus 0-3 --language-model-only \
  --expert-scale-compression csf --output /data/flare-runtime/minimax-m3.1-flash-tp4.safetensors
flare/target/release/flare run /data/flare-runtime/minimax-m3.1-flash-tp4.safetensors --gpus 0-3
```

## Storage and graph capacities

Gemma 4 defaults to four concurrent requests; its speculative target-token capacity
is the larger of padded concurrency and draft count plus one, giving six rows at
its default settings. Other recipes default to 32 concurrent requests, with
speculative recipes declaring 128 target-token rows independently of concurrency.
Explicit flags and configuration override these defaults; existing packages retain their
captured capacities. Memory planning rejects capacities that do not fit the
selected GPUs.

| Recipe | Draft tokens | Verification layout | Speculative graphs at default concurrency | Maximum drafts at batch 32 |
|---|---:|---|---:|---:|
| Qwen3.8 Flash, Qwen3.8 27B, GLM 5.3, DeepSeek V4 | 5 | Rectangular | 34 | 3 |
| Step-5 | 3 | Rectangular | 24 | 3 |
| MiMo V2.6 Pro, Kimi K3, Nemotron Bolt | 7 | Rectangular | 44 | 3 |
| DeepSeek V4.1 | 7 | Packed | 64 packed + 6 width-one | Up to 7, sharing 128 rows across requests |

Rectangular catalogs cover full draft depth through batch 16. The counts include
width-one speculative graphs and exclude ordinary decode, prefill, maintenance,
and vision graphs. Capturing full rectangular depth at batch 32 requires 192 rows
for five drafts or 256 rows for seven drafts, set with
`--speculative-token-capacity`. DeepSeek V4.1's packed graph count is bounded
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
Ignition additionally budgets 1 GiB for warm-up and capture. Preparation KV backing
covers a prefill chunk plus boundary pages per captured request row, unless an
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
`--speculative-token-capacity` defaults to 128 physical target rows for every
speculative recipe. DeepSeek V4.1's packed capacities follow its packed catalog table
above rather than rectangular `batch * width` shapes. Previously exported separate-proposal
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
  --gpus 0,1 --host 127.0.0.1 --port 8001
curl --fail http://127.0.0.1:8001/health
```

Select a subset of a package's captured execution capacities:

```sh
./flare/target/release/flare run /data/flare-runtime/qwen38-tp1-c4-mtp5.safetensors \
  --gpus 2 --max-concurrency 2 --mtp-draft-tokens 3 \
  --no-mtp-adaptive-verification --host 127.0.0.1 --port 8001
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
.venv/bin/ignition export qwen3.8-flash --gpus 0,1 --output qwen.safetensors --verbose
flare/target/release/flare run qwen.safetensors --gpus 0,1 --port 8001
```

To test a serving-only Rust change, rebuild Flare and load an existing compatible
package. Re-export when changing prepared kernels, capacities, or the package
format. Export destinations must not already exist.

Use `.venv/bin/ignition export --help` for preparation flags and
`flare/target/release/flare run --help` for serving flags. See [the runtime package design](runtime-package-design.md)
for resource ownership and [the CLI design](cli-design.md) for configuration.

For model development, `ignition serve` hands prepared graphs directly to Helios
without exporting a package. It requires Heliopause's `qualification` Cargo
feature. Select idle devices explicitly; see [the development launch example](cli-design.md#development-without-package-export)
for the extension build and serving commands.
