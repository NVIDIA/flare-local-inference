# Ignition preparation and Flare serving commands

For environment setup and model-specific launch examples, see
[model preparation and serving](model-preparation.md).

Flare serves the model architectures its recipes support on NVIDIA SM120 and
SM121 GPUs. Recipes declare default checkpoints and supported preparation
capacities. The command line selects models, GPUs, concurrency, storage, and
listeners. The engine derives memory and execution geometry from those choices;
advanced preparation flags expose the explicit overrides listed below.

See [native inference APIs](api.md) for Chat Completions, raw Completions,
Anthropic Messages, stateless Responses, authentication, and client configuration.

## 1. Principles

- **Intent, not mechanism.** A parameter names a decision only the user can
  make: which GPUs, how many simultaneous requests, where to listen.
- **The engine is the recipe.** Per-model knowledge lives in `ignition/recipes/`
  as qualified code. A recipe declares its checkpoints, their pinned revisions,
  supported tensor-parallel sizes, concurrency range, memory cost model, and
  template defaults. The command line never needs to know which model it runs.
- **Derive execution geometry.** Context length, page count, GPU prefix-cache
  size, snapshot slots, graph capacities, worker counts, and body limits follow
  the recipe, host, and selected capacities. Advanced flags override the recipe's
  prefill chunk and speculation settings.
- **Memory follows the model and the host.** Weights, graphs and
  scratch, live state, and a prefix cache sized from `--max-concurrency` come
  first. KV pages take everything else: the GPU's free memory at startup minus
  engine headroom. The page pool holds active requests and every cached prefix,
  so memory beyond what active requests need keeps more history resident.
  The configuration's GPU reserve leaves memory for processes started later.
  Runtime `--memory-limit` replaces that serving reserve with an occupied-memory
  cap, free-memory reserve, or fraction of total memory. `--kv-cache-limit`
  optionally bounds the shared page pool by one context or context times concurrency.
- **Estimate roughly, allocate from the device.** Before weights load, Flare
  estimates memory from the recipe's declared sizes, only to choose GPUs and to
  refuse a model that cannot fit. Allocation follows the memory the device has
  free during preparation, and `flare run` prints what it allocated. An estimate that
  does not fit fails before any socket opens, with the numbers and the settings
  that would make it fit.
- **No environment variables.** Flare reads none, with one exception:
  `CUDA_VISIBLE_DEVICES` restricts which GPUs the caller exposed, which changes
  the GPU each index names, so Flare inspects it to interpret `--gpus`.
  Variables that underlying libraries read are set by Flare from its own
  configuration, and b12x's from the recipe's declared settings, before those
  libraries are imported.
- **Two levels of precedence.** Flags override the configuration file, which
  overrides the recipe. Nothing else exists.

## 2. Commands

The Python distribution and source package are named `ignition`. Install the
preparation environment with `uv sync --locked --extra prepare --extra test`,
then use `.venv/bin/ignition` or `.venv/bin/python -m ignition`.

```
ignition export  [MODEL] [--output FILE] [flags]  prepare a native model package
ignition convert FILE [--output FILE] [flags]   convert a package to shared blobs
```

Every command has exactly one name. `MODEL` is a recipe alias, a Hugging Face
repository id with an optional `@revision`, or a local checkpoint directory;
with no `MODEL`, `export` and `serve` use `defaults.model` from the configuration file and
fail with a usage error if it is unset.

`ignition --help` and preparation-command help list every declared model alias,
supported tensor-parallel size, and default/maximum draft count. Put an alias or
declared checkpoint repository before `--help` for its defaults, limits, and
optional features, for example `ignition export step5-preview-nvfp4 --help`.
Help reads recipe declarations without loading model dependencies, probing GPUs,
or downloading checkpoint files. Unknown model aliases suggest a close match
when one exists.

Ignition exposes preparation and development launch commands. It prints
preparation progress without a banner. The Rust Flare executable displays the
Flare banner on a terminal and reports model loading through Helios.

`export` imports the `flare_heliopause` CPython extension in Ignition's process.
Ignition performs weight loading, tuning, compilation, warmup, and capture,
then hands graph handles, metadata, and retained resource owners to Heliopause.
The native exporter writes compiled CUDA modules, allocation contents, graph
descriptions, and frontend assets to content-addressed payload blobs with a
metadata-only `.safetensors` primary. Native stderr phases report scanning,
hashing, copying, cache reuse, encryption, and publication progress. Every
destination must be unused. Ignition exits without opening a listener after
publication. Launch the exported package
with `flare run PACKAGE`. Qwen, GLM, and DeepSeek provide native package storage support.
Tensor-parallel packages contain one resource bundle per selected GPU and load
in one process with one execution thread per rank. API keys are excluded from
the package.

Build the public serving executable, which links the Helios library:

```sh
cargo build --release --manifest-path flare/Cargo.toml
flare/target/release/flare run PACKAGE --gpus 0 --host 127.0.0.1 --port 8000
```

```
flare run [PACKAGE] [flags]               serve locally, join a rank, or launch peer agents
flare download REPOSITORY [flags]         download a prepared package
flare inspect PACKAGE [--memory-map] [--graphs] [--frontend] [--all] [--json | --plain]
flare agent [--listen IP:PORT]            run this host's rank agent
flare agent install [--join-code CODE | --accept-first] [--listen IP:PORT]
flare agent pair HOST... [--code CODE]    pair for --hosts or Orbit launches
flare agent reset                        forget authorized controllers and pairing codes
flare orbit pair HOST... [--code CODE]    pair this controller with hosts
flare orbit controllers add NAME KEY | remove KEY
flare orbit status [--cluster NAME]
flare orbit run [PACKAGE] [--cluster NAME | --nodes NODE,...] [--name GROUP] [--persistent] [--executable PATH] [--dry-run] [serving flags] [--detach]
flare orbit attach [GROUP] [--cluster NAME]
flare orbit stop [GROUP] [--cluster NAME]
flare orbit rm GROUP                     remove a stopped definition or retained launch
flare orbit forget NODE                   remove a node from the orbit file
flare orbit push PACKAGE --to NODE,...    stage package data without starting ranks
flare orbit packages [--cluster NAME]     what each node's cache holds
flare orbit ls | dashboard [--cluster NAME]  groups on the cluster's nodes; their live metrics
flare trust revoke FINGERPRINT
```

`flare run PACKAGE --rank 0 --rendezvous HOST:PORT` serves rank zero of a
multi-host package. Each peer can run `flare run --rank N --rendezvous HOST:PORT`
without a local package: serving bootstrap receives its required ranges over
CX7, verifies the package and publisher trust, then joins capacity agreement.
A supplied local peer package must have the same export identity as rank zero.

`flare run PACKAGE --hosts PEER1,PEER2,...` runs rank zero locally and starts
explicitly ordered peer agents. `--hosts` excludes rank zero. Pair the peers
with `flare agent pair` or reuse Orbit pairings. The head node needs no agent
or cluster definition. The shared registry and key default to
`~/.config/flare/orbit.toml` and `~/.config/flare/orbit/controller.key`;
`--agents-file` and `--controller-key` override them. Agents use their installed
serving executable. The rendezvous defaults to port 29800 on the head node's
address toward its first peer; an explicit `--rendezvous` overrides it.

Serving package distribution uses 1..64 streams per peer, selected by rank
zero's `--transfer-streams` (default 8). Direct and switched fabrics route to
the peer; `--roce-ring` uses intermediate ranks. `--package-cache` sets the
receiving rank's cache root (default `~/.cache/flare`). Rank zero's
`--stage-only` distributes and verifies packages without GPU loading, and exits
all ranks. Manual ranks need no agents, route signatures, or routing keys.
All launch modes use the same native distribution and serving bootstrap.

`flare orbit` commands take `--orbit-file PATH` and `--key-file PATH`,
defaulting to `~/.config/flare/orbit.toml` and
`~/.config/flare/orbit/controller.key`. `flare orbit run` takes the serving
flags of `flare run` and applies them to every rank; `--host`, `--port`,
`--served-model-name`, `--scheduler-policy`, and `--api-key-file` apply to rank
zero. `--api-key-file` names a file rank zero must be able to read; Orbit does
not upload it. Package inputs are Hugging Face repositories
`ORG/NAME[@REVISION]`, controller-local files, or `NODE:/absolute/path` on a
paired node. Orbit downloads repository packages on a candidate node and
verifies their publisher on the controller. Download flags apply to that
node's cache; `--token-file` is read on the controller.

Ranks use each node's installed agent executable by default. An explicit
`--executable PATH`, or a saved group's override, uploads the chosen build.
`--persistent --name GROUP` saves a launch for manual reuse;
`--detach` returns when processes start, before HTTP readiness is guaranteed.
See [Orbit](orbit.md) and the [multi-host guide](multi-host.md). Orbit commands
are implemented in Unix builds; Windows builds reject them.

For MSVC cross-compilation, native Windows builds, and GPU compatibility
requirements, see [Windows serving](windows.md).

The native device ordinal follows CUDA's visible device enumeration. Optional
`--api-key-file PATH` supplies authentication at launch. Loading requires
the package's GPU architecture and sufficient GPU and mapped host memory.
Recorded addresses must be available; an address collision fails startup.
The package records prepared concurrency, the available prefill and speculation
graphs, and maximum virtual extents for cache storage. `--max-concurrency N`
and `--mtp-draft-tokens N` select supported reductions at load time;
`--mtp-adaptive-verification` and `--no-mtp-adaptive-verification` select the
verification policy. `--history-drafts` enables host history tracking and feeds
continuations into existing graph inputs; it does not change the captured graphs.
`--no-history-drafts` disables those proposals. Packages exported by Ignition
record no history preference. Existing packages retain their recorded defaults
unless explicitly overridden, including copy-only packages whose prepared
graphs remain loadable. Omitted capacity flags use package defaults. Only selected
graphs and required kernels load. Continuation rows and KV pages receive physical
storage according to those selections and available GPU memory.

## 3. Parameters

### 3.1 Ignition command-line flags

`export` and `serve` accept the preparation flags below. `export` also accepts
`--output FILE` and defaults to a local snapshot. Listener flags apply only to
`serve`; native package serving uses `flare run` flags. Development `serve` also
accepts the runtime `--history-drafts` / `--no-history-drafts` options for native
serving. `export` accepts neither option, and `--direct` rejects history drafting.

| Flag | Meaning | Default |
|---|---|---|
| `MODEL` | Alias such as `qwen3.8-flash`, Hugging Face repository id such as `org/name` or `org/name@revision`, or checkpoint directory | `defaults.model` |
| `--gpus LIST` | GPU indices as NVML enumerates them: `0`, `0,1`, `0-3`. The count is the tensor-parallel size and must be one the recipe supports. | First GPUs that fit |
| `--max-concurrency N` | Maximum simultaneously active requests, 1 through the recipe's limit | `defaults.max_concurrency`, else the recipe default |
| `--host ADDR` | Listening address. `serve` only. | `server.host`, else `127.0.0.1` |
| `--port N` | Listening port. `serve` only. | `server.port`, else `8000` |
| `--verbose` | Per-phase detail on stderr | off |
| `--config PATH` | Configuration file | `~/.config/flare/config.toml` |
| `--shard-size SIZE` | Hard payload-file limit for export/sign/convert, at most 64 GiB | `[publishing] shard_size`, else `64G` |
| `--no-weights` | Export only when exact stored-data blobs are cached | Off |
| `--repo-id NAMESPACE/REPOSITORY` | Export repository namespace for the default local snapshot path | Recipe `export_repository` |
| `--official-source URL` | Published repository URL under the configured publisher namespace | `[publishing] official_source` |

Preparation flags are grouped by capacity/compression, speculative decoding,
and advanced options in `--help`. Recipe values apply when flags are omitted;
explicit flags take precedence.

| Flag | Meaning | Default |
|---|---|---|
| `--prefill-chunk-size N` | Most prompt tokens processed in one prefill step, a multiple of 64. A larger chunk reaches a long prompt's first token sooner; while it runs, other active requests wait for their next token. Command line only. | The recipe's chunk |
| `--mixed-prefill` | Capture mixed prompt/decode graphs alongside separate graphs. Supported by Qwen with concurrency of at least two, with or without speculation. Command line only; existing packages retain their captured catalog. | Disabled |
| `--expert-scale-compression [MODE]` | Losslessly compress routed-expert scales: DeepSeek V4.1 backbone and DSpark MXFP4 scales, or Step-5 NVFP4 scales including BF16 experts converted to W4A16. `csf` is a compressed scale encoding expanded during execution. Estimated resident GPU weight savings are 3% for DeepSeek V4.1 and 6% for Step-5; actual savings depend on encoded scales. A bare flag selects `csf`; `native` disables compression. Retain compressed scales and share expansion scratch across serial layers. | Recipe default: `csf` for DeepSeek V4.1, native otherwise |
| `--no-mtp` | Disable multi-token speculation: load no draft model, keep no verification rows, and decode one token per step. | Speculation on |
| `--mtp-draft-tokens N` | Most draft tokens verified in one round; see `MODEL --help` for its limit. Verification capacities cover counts from 0 through `N` within the speculative token capacity. Retained and packed proposal producers execute the full configured block; verification adapts using measured cost and historical acceptance or native proposal confidence. Zero retains draft state with width-one execution; `--no-mtp` excludes draft weights and state. | Recipe declaration |
| `--speculative-token-capacity N` | Maximum physical target-token rows in wider speculative graphs; width one covers every request bucket. Applies to every speculative recipe. | 128 |
| `--speculative-graph-budget N` | Maximum packed round graphs to capture, independently of physical target-token rows. Excludes mandatory width-one and ordinary graphs; requires packed adaptive learned speculation at positive draft depth. Must cover both physical-row endpoints for every eligible request bucket. | 64 for DeepSeek V4.1 |
| `--no-mtp-adaptive-verification` | Default to fixed-width serving. Capture includes every smaller width within the speculative token capacity. Positive-depth DeepSeek V4.1 DSpark requires adaptive verification. | Adaptive |
| `--mla-compute-mode {auto,bf16}` | Constrain DeepSeek V4.1 MLA arithmetic while preserving packed KV storage and autotuning among compatible kernels. | Recipe default: `bf16` for DeepSeek V4.1, `auto` otherwise |
| `--no-stream-overlap` | Run every concurrent branch (shared experts, projection chains, query chains) serially on the main stream with the same kernels, to diagnose scheduling-dependent faults. Slower. Recipes that run no branches concurrently reject it. | Overlap |
| `--autotune-effort off\|low\|medium\|high\|max` | Kernel search effort. Off uses heuristics; low applies broad exclusions; medium balances search cost; high retains only conservative exclusions; max enumerates every declared legal candidate. Overrides `[preparation] autotune_effort`. | Medium |
| `--compile-workers N` | CPU compiler process budget per host, divided among local preparation ranks and bounded by available RAM and device memory. Applies to b12x kernel compilation and complete-forward compilation. Overrides `[preparation] compile_workers`. | Automatic |
| `--tuning-gpus LIST` | Local kernel-tuning GPUs. Start with the serving GPU list, then append complete replicas of that TP group on matching hardware. Serving TP size stays fixed. Command line only; cannot be combined with multi-host `--ranks`. | Serving GPUs |
| `--no-compile` | Development only: capture graphs without compiling forwards. A recipe never sets it. | off |
| `--online-mxfp8` | Store the recipe's eligible BF16 projections as MXFP8 while loading (Kimi K3: KDA projections, MLA latent/gate, query and output projections, shared experts, MoE latent down/up projections, and DSpark draft context, outputs and MLPs), trading checkpoint precision for decode bandwidth. MLA's batched key/value projections retain BF16. Recipes that name no such projections reject it. | Checkpoint precision |
| `--language-model-only` | Prepare no vision encoder and serve text only, as vLLM's flag of the same name; requests with images are rejected and the encoder's weights, scratch, and graphs become KV pages. Recipes without a separable encoder reject it. | The recipe's image support |

Speculative token capacity and packed graph budget also accept configuration-file
defaults and model overrides. The remaining speculation settings are command-line
only. `--no-mtp` excludes explicit speculation flags and model-specific packed
graph budgets. The `Select GPUs.` startup line states the settings in effect: `up to
N draft tokens` while verification adapts, `exactly N draft tokens` when it is
fixed, and `branches serialized` under `--no-stream-overlap`. That flag is also
recorded in the plan and the exported graph catalog; it changes graph scheduling,
not kernel selection, so its packages share tuned kernels with overlapped ones.

Gemma 4 defaults to four concurrent requests and six speculative target-token
rows. Other recipes default to 32 concurrent requests, with speculative recipes
declaring 128 target-token rows. Their rectangular catalogs cover the recipe's
full draft depth through batch 16 and up to three drafts at batch 32; Step-5's three-draft catalog
covers its full depth at batch 32. Each recipe selects its proposal and
prefix-allocation mode.
DeepSeek V4.1 defaults to seven draft tokens, concurrency 32, a 4096-token
prefill chunk, 128 physical target rows, and at most 64 packed round graphs,
with packed GPU prefix selection.
Its catalog contains combined proposal/allocation/verification rounds and
width-one graphs; ordinary graphs remain available for `--mtp-draft-tokens 0`.
The [model preparation guide](model-preparation.md#deepseek-v41-flash) defines
the packed capacity selection. The row cap bounds target work and storage;
raising it permits longer aggregate prefixes at larger batches. The independent
graph budget controls catalog density between each batch's smallest wider round
and its maximum physical extent. Width-one and ordinary graphs do not consume
that budget. Budgets below the required endpoint count fail during planning;
budgets above the number of distinct valid capacities capture every capacity.
For eight requests with five drafts each, this example instead limits physical
target capacity to 24 rows while retaining six-token per-request prefixes:

```sh
.venv/bin/ignition export deepseek-v4.1-flash --gpus 0,1,2,3 \
  --max-concurrency 8 --mtp-draft-tokens 5 \
  --speculative-token-capacity 24 --prefill-chunk-size 4096 \
  --output /data/flare-runtime/deepseek41-tp4-c8-packed24.safetensors --verbose
```

Packed packages require adaptive execution at their captured draft depth, or
`--mtp-draft-tokens 0` for ordinary execution. Native serving uses all captured
packed capacities by default. Prefer `--mtp-max-verification-token-budget N` to
retain every captured capacity at or below a ceiling; for example:

```sh
./flare/target/release/flare run \
  /data/flare-runtime/deepseek41-tp4-c8-packed24.safetensors \
  --gpus 0,1,2,3 --mtp-max-verification-token-budget 24 \
  --host 127.0.0.1 --port 8001
```

For a controlled comparison, replace the maximum flag with
`--mtp-verification-token-budget 24` to select one exact captured capacity per
batch, bounded by that batch's full prefix extent. The exact and maximum flags
are mutually exclusive and require a packed adaptive catalog.

A catalog containing only combined packed rounds and width-one continuations
allows a runtime row budget below the selected padded batch capacity. Batches
without a retained combined round use their width-one graphs. For the default
DeepSeek V4.1 catalog, concurrency 32 with an exact budget of eight retains
batches 1, 2, and 4 at eight rows each; batches
8, 16, and 32 retain width-one execution. A maximum budget of eight retains every
captured combined capacity at or below eight instead. At least one combined
round must remain, and an uncaptured exact capacity fails selection.

Catalogs containing separate proposal and verification graphs require the
budget to cover the selected padded batch capacity. A maximum budget must cover
the captured per-request verification width for either catalog. Missing required
graphs fail before GPU initialization.

DeepSeek V4.1 defaults to BF16 MLA arithmetic. `--mla-compute-mode auto`
lets b12x choose arithmetic. Packed KV storage remains unchanged, and compatible
kernels still autotune. The pinned
FlashInfer dependency supplies the required preparation API. For example:

```sh
.venv/bin/ignition export deepseek-v4.1-flash --gpus 0,1,2,3 \
  --max-concurrency 8 --mtp-draft-tokens 5 \
  --speculative-token-capacity 24 --prefill-chunk-size 4096 \
  --mla-compute-mode bf16 \
  --output /data/flare-runtime/deepseek41-tp4-c8-packed24-bf16.safetensors --verbose
```

The arithmetic choice applies to target, prefill, and draft attention and is
fixed in the exported graphs. Changing it requires another export.

CPU compilation and GPU tuning have independent controls. For a TP1 package,
`--gpus 0 --tuning-gpus 0-3 --compile-workers 24` uses four GPUs to benchmark
disjoint candidate subsets and at most six CPU compiler processes per tuning
replica. Each replica loads the model and its preparation storage; only the
serving GPU captures graphs. Helpers exit after tuning, allowing forward
compilation to use the full host CPU budget, bounded by memory. Compiler
processes dispatch b12x compilation without opening CUDA contexts; complete
forward workers also use the owning rank's GPU for Inductor preparation.

For TP2, `--gpus 0,1 --tuning-gpus 0-5` creates three independent TP2 groups.
Corresponding ranks share candidate results while collectives stay within each
replica. The resulting package still contains two ranks. Additional GPUs must
match the corresponding serving GPU model and have enough free memory for a
replica. Both GPU lists use `CUDA_VISIBLE_DEVICES` ordinals.

Without these flags, preparation retains automatic worker sizing and its
existing local or multi-host sharding. On integrated GPUs such as DGX Spark,
CPU and GPU allocations share one memory budget. Multi-host preparation keeps
its rank coordination, candidate sharing, and compiled-forward distribution;
`--compile-workers` sets a separate budget on each host. Local tuning replicas
are selected only by `--tuning-gpus`.

Both Ignition commands accept `--config`.

Multi-host `ignition export` and `ignition serve` transfer checkpoint tensor
ranges and prepared export payloads over the discovered CX7 fabric interfaces.
Direct cables between two ranks and switched fabrics use direct peer TCP
connections. `--roce-ring` selects forwarding through rank neighbors for a
physical ring. `--transfer-streams N` selects 1 through 64 concurrent payload
streams per peer (default 8), split across the available rails in either topology.
Each rank owns its transfer helper; preparation requires no Orbit agents,
controller files, pairing, or route signatures. The existing shared preparation
token authenticates checkpoint and export frames end to end without encryption.
Metadata, endpoint exchange, and coordination use the rank rendezvous.
See [checkpoint preparation](multi-host.md#prepare-the-package) for credentials,
buffer bounds, and rendezvous traffic.

#### Development without package export

For development, `ignition serve` prepares graphs and hands them directly to
Helios in the same process. It requires a cached checkpoint and a Heliopause
build with the `qualification` feature. Requests execute in Rust without Python
callbacks. This command skips package export and reload; weight loading, kernel
tuning, compilation of missing forwards, and graph capture still run.
Diagnostics retain the bounded cache backing allocated during preparation.
Use an exported package for long-context capacity tests; native package loading
sizes the serving cache from the memory available after loading the model.

Select idle serving GPUs explicitly. Tensor-parallel preparation distributes
forward compilation across the ranks within their memory budgets. Compiler
workers release their CUDA contexts before cache storage is materialized.
Build the extension once, then start a TP2 DeepSeek V4 development server:

```sh
uv pip install --python .venv/bin/python --no-deps ./heliopause \
  --config-settings 'build-args=--features=qualification'
.venv/bin/ignition serve deepseek-v4-flash --gpus 0,1 \
  --max-concurrency 1 --mtp-draft-tokens 3 --no-mtp-adaptive-verification \
  --prefill-chunk-size 4096 --host 127.0.0.1 --port 8001 --verbose
```

Rebuild the extension after changing its Rust code. Python recipe and component
changes require preparation again but no Rust rebuild. Validate the chosen
configuration through an exported package before publishing native-serving
performance results.

The `deepseek-v4-flash` alias selects the lossless MXFP4-CSF checkpoint
`local-inference-lab/DeepSeek-V4-Flash-Vision-Exp-lossless-CSF`.
The `deepseek-v4.1-flash` alias selects `deepseek-ai/DeepSeek-V4.1-Flash-1017`,
with CSF compression of its backbone and DSpark MXFP4 expert scales by default.
`--expert-scale-compression native` preserves native scales during preparation.
Compressed routed-expert scales remain in GPU memory. Serialized layers reuse scale buffers by expert geometry;
b12x prepares the scales needed by its selected MoE kernel before execution.
Kernel autotuning uses the selected effort across the supported execution plans.

DeepSeek V4.1 development serving keeps its Engram table partitions in RAM. Allow
about 189 GiB of host RAM across the four ranks, in addition to preparation and
serving memory. Each rank loads the checkpoint spans for its six hash heads into
immutable resident storage. Requests gather rows from RAM without reading the
checkpoint files or calling Python. Each rank owns its delivery staging and CUDA
events. This development path supports ranks on one host and uses RAM only.

With four serving GPUs, run:

```sh
.venv/bin/ignition serve deepseek-v4.1-flash --gpus 0,1,2,3 \
  --max-concurrency 32 --mtp-draft-tokens 5 --prefill-chunk-size 4096 \
  --speculative-token-capacity 128 --host 0.0.0.0 --port 8001 --verbose
```

The command prepares and captures graphs, loads the RAM tables once, and starts
Helios directly. It does not write or reload a runtime package. The extension
build above is required once; model edits reuse cached kernel and forward
artifacts where their prepared computation is unchanged. Use an exported package
for final native performance validation: development ranks run in separate
processes with separate delivery staging, while the native CLI shares staging
between its rank threads.

DeepSeek V4.1 exports combined packed rounds with GPU prefix allocation by
default. This command uses the recipe's concurrency 32, seven draft tokens,
4096-token prefill chunk, and 128 physical target rows:

```sh
.venv/bin/ignition export deepseek-v4.1-flash --gpus 0,1,2,3 \
  --output /data/flare-runtime/ds41-tp4-c32-d7-p128.safetensors
```

The default catalog contains 64 packed rounds with logical width eight and six
proposal-free width-one rounds. Packed execution generates the seven-draft block,
then allocates actual prefixes on the GPU within each request's immutable limit
and the captured aggregate row capacity. It reports one terminal completion.
Ordinary prefill, decode, and maintenance graphs remain available when native
serving selects `--mtp-draft-tokens 0`.

### 3.2 Native serving flags

`flare run --scheduler-policy balanced` is the default and balances token
throughput with output cadence. For throughput-oriented serving, use
`flare run model.safetensors --gpus 0,1,2,3 --scheduler-policy throughput`.
This preference retains token-service fairness and can produce longer gaps
between larger output bursts. It uses the package's existing graphs.

`flare run PACKAGE` loads an exported file without the Ignition configuration
reader. `PACKAGE` can also be a Hugging Face repository ID with an optional
`@revision`; Flare downloads its prepared package before loading it. A
[serving container](container.md) packages Flare and Helios without
Python or the CUDA toolkit. `flare --help` lists commands; `flare run --help`,
`flare run -h`, and `flare help run` describe every serving option, its defaults,
and accepted values.
Help also works after the package path and other options, without opening the
package or initializing CUDA. Options may precede or follow `PACKAGE`;
value-taking options accept both `--option VALUE` and `--option=VALUE`.
Help exits with status 0; usage errors go to stderr with status 2. Its options are:

| Flag | Meaning | Default |
|---|---|---|
| `--publisher-keys PATH` | Additional JSON verification profiles; importing a key does not approve execution or verify identity | Embedded profiles, remembered keys, and the default development profile for development packages |
| `--dangerously-approve-key FINGERPRINT` | Explicitly approve this full lowercase SHA-256 key fingerprint; requires the matching publisher flag | Interactive first use or remembered trust |
| `--dangerously-accept-publisher NAMESPACE` | Accept this exact publisher namespace; requires the matching key flag | Interactive first use or remembered trust |
| `--trust-store PATH` | Persistent key/namespace acceptance file | `~/.config/flare/trust.toml` |
| `--verify-weights` | Verify the stored allocation and table ranges of the ranks served before loading model GPU resources | off |
| `--gpus N[,N...]` | Selected GPUs in rank order; indices follow an inherited `CUDA_VISIBLE_DEVICES` list when present | First N visible GPUs for the package's N local ranks; one GPU per host for multi-host packages |
| `--host ADDR` | Listener IP address, IPv4 or IPv6 | `127.0.0.1` |
| `--port N` | Listener port | `8000` |
| `--served-model-name ALIAS` | Replace the package's serving aliases with this nonblank alias | Package aliases |
| `--api-key-file PATH` | File containing the serving API key | No authentication |
| `--max-concurrency N` | Reduce concurrency within captured batch capacities | Package default |
| `--kv-cache-limit concurrency\|context\|memory` | Bound shared KV history by prepared context times selected concurrency, one prepared context, or available memory | `memory` |
| `--memory-limit LIMIT` | Per-GPU occupied size (`30G`), free-memory reserve (`-30G`), or fraction of total memory (`0.8`) | Package serving reserve |
| `--mtp-draft-tokens N` | Reduce the maximum draft count; zero selects ordinary execution without draft generation, absorption, or draft-state maintenance | Package default |
| `--mtp-adaptive-verification` / `--no-mtp-adaptive-verification` | Adaptive or fixed selection among captured widths | Package default |
| `--history-drafts` / `--no-history-drafts` | Augment learned drafts with shared decoded-history continuations; requires enabled speculation and captured history inputs, as Qwen Flash provides | Off; existing packages retain recorded defaults unless overridden |
| `--mtp-adaptive-verification-alpha ALPHA` | Positive finite exponent on split-proposal prefix-survival scheduling scores; below 1 favors longer verification prefixes | `1.0` |
| `--mtp-verification-token-budget N` | Select one exact captured packed verification capacity per batch | All captured capacities |
| `--mtp-max-verification-token-budget N` | Retain all captured packed verification capacities at or below a total-row ceiling | All captured capacities |
| `--embedding-table-storage disk\|ram` | PLE and Engram table storage; both modes deliver compact GPU rows | `disk` |
| `--embedding-table-cache-size SIZE` | Per-host RAM budget for cached disk-backed PLE/Engram rows, including the index, usage counters and miss workspace; accepts sizes such as `1G` or `2GiB`; `0` disables | `4GiB` on discrete GPUs; `1GiB` on unified-memory GPUs such as DGX Spark |
| `--embedding-table-resident-scales` | Keep original embedding scale bytes in host RAM while weights remain on disk; adds to the row-cache budget | Disabled |

`--mtp-adaptive-verification-alpha 0.5` scores each joint prefix-survival
probability `p` as `sqrt(p)` when choosing split-proposal verification widths
and packed budgets. `1.0` uses the probabilities directly; values above `1.0`
favor shorter prefixes. Confidence observations and token acceptance remain
unchanged. This serving option uses existing package graphs and does not enable
adaptive verification when fixed-width selection is configured.

`--prefix-cache-manager SOCKET` selects an external shared checkpoint cache over
one Linux Unix stream socket per replica. It replaces local checkpoint retention:
nonzero `--prefix-cache-memory` and any `--prefix-cache-disk` are rejected.
Multi-host packages are rejected before staging allocation. The option is
forwarded to every rank, so the same refusal applies to distributed launches.
`--prefix-cache-tenant LABEL` selects the manager namespace shared by replicas;
the label is 1 to 64 bytes of `[A-Za-z0-9._-]` and defaults to empty, which
shares records only with other unlabelled replicas. It is forwarded to every
rank. Without the option, local stores and disabled serving follow their
existing paths.

The manager and replica must share effective uid/gid; the manager records the
connecting process from Unix peer credentials and opens a pidfd for window
reclamation. Containers sharing the socket must share the relevant PID
namespace. See the
[pooled checkpoint client](prefix-cache-offload.md#pooled-checkpoint-client) for
staging, reconnection, cancellation, durability and validation limits.

`--embedding-table-workers 1|2|4|8` controls Windows disk-reader workers.
Workers share 128 I/O slots and 8 MiB of direct-read staging per embedding table.
Windows defaults to four workers; other platforms retain one. Explicit worker
counts override the default. Large gathers use scoped helper threads, whose startup
cost is included in gather latency. Small gathers stay on the delivery thread.
Worker phase counters sum overlapping worker intervals and must not be subtracted
from elapsed gather time. RAM storage rejects this disk-only option.

Disk-backed embedding tables default to a 4 GiB host-RAM row cache per host,
or 1 GiB when any selected GPU shares host memory, as on DGX Spark. Explicit
cache sizes, including `0` to disable caching, override the hardware default.
RAM table storage and packages without embedding tables allocate no row cache
by default. An explicit nonzero cache requires disk storage and an embedding
table catalog. Each host shares its budget across embedding tables and local
tensor-parallel ranks. The loader commits cache memory before
sizing KV and continuation pools, so the cache uses part of the integrated GPU's
existing memory budget. The cache remains fixed in size while serving and warms
from requests each time the server starts.

`--embedding-table-resident-scales` loads the original scale plane into immutable
host RAM before KV sizing. It applies to every table in the package's embedding
catalog, including Qwen PLE and DeepSeek Engram. The loader uses the declared
weight and scale row sizes; it does not select behavior by model identity.
Weight rows remain on disk, and a row-cache miss reads
only the weight plane. The resident scale allocation is additional to
`--embedding-table-cache-size` and is included in the integrated GPU memory
budget. This option works with existing packages and requires disk table storage;
RAM table storage already retains both planes. The metric
`flare_engram_resident_scale_bytes` reports the scale allocation separately from
the row cache.

Ignition exports each recipe's canonical model name as its default serving
alias, such as `DeepSeek-V4.1-Flash`. Checkpoint selectors such as
`deepseek-v4.1-flash` select preparation inputs and configuration sections;
they do not set the advertised API name. Existing packages retain their
embedded aliases.

`--served-model-name` changes the name advertised by `/v1/models` and accepted
for requests. It replaces the package aliases; the canonical model ID remains
accepted and appears as the model card's `root`. The override does not modify
the package or change its identity. `flare run` accepts numeric IPv4 and IPv6 addresses through
`--host`; use an unbracketed IPv6 address such as `--host ::1 --port 8001`.
The combined `--bind` flag has been removed.

Selected concurrency and draft depth cannot exceed prepared capacity.
Concurrency three requires the batch-four bucket. Fixed verification loads
width `N+1` where the token envelope permits it, plus width-one execution at every
batch; adaptive verification loads the permitted
widths. Missing variants fail before GPU allocation. Continuation rows and
KV pages receive physical backing after graph upload, within fixed virtual
extents. Reducing execution state can leave more memory for KV pages.

### Runtime memory limits

`--kv-cache-limit memory` uses the remaining memory budget for KV, as before.
`context` caps useful KV history at one prepared context; `concurrency` caps it
at one prepared context per selected active request. Concurrency three uses
three contexts even though its graphs use a batch-four bucket. Context means
prompt plus generated tokens. These modes leave `--max-concurrency` unchanged:
multiple shorter requests share the pool in `context` mode.

History caps round each context up to logical cache pages and add the selected
graphs' required private draft-page storage. They use the prepared context
ceiling before fitting the served context to physical memory. All modes remain
bounded by available memory and prepared virtual extents. The capped modes may
start with a smaller pool than their ceiling; their minimum is one history page
plus required draft storage per active request. The default memory mode retains
its 8,192-token-per-request startup floor. A pool below its minimum fails startup.
Prefix checkpoints continue borrowing the shared KV backing and are evicted
under the existing shared policy.

`--memory-limit 30g` budgets at most 30 GiB for Flare on each selected GPU,
including loaded weights, graph and kernel setup, scratch, execution state,
caches, and runtime headroom. `--memory-limit -30g` leaves at least 30 GiB free
at startup, in addition to mandatory engine headroom. `--memory-limit 0.8`
uses 80% of CUDA-visible total memory as Flare's occupied budget; it is not
80% of the memory currently free. Other processes still constrain the actual
available supply. Lowercase and uppercase size suffixes use binary units,
including `G`, `GB`, and `GiB`; occupied and reserve sizes require a suffix.
Bare numbers must be greater than zero and at most one. Zero, negative
fractions, nonfinite values, and overflowing sizes are rejected. Both
`--memory-limit -30g` and `--memory-limit=-30g` work.

An explicit memory flag replaces the package's serving reserve; the two are
never subtracted together. Omission preserves the exported `[memory] reserve`.
Integrated GPUs also count resident host tables, embedding row caches, and
future prefix-cache host pools in the same physical budget. Shared host memory
is charged once per host. The existing driver margin, host allowance, and
system headroom remain in force. An occupied cap smaller than known fixed
resources fails before loading those resources. The loader reconciles observed
startup consumption after graph upload and before allocating cache backing.
Discrete occupied/fraction limits use NVIDIA NVML to observe memory before
CUDA context creation; Linux integrated devices use `MemAvailable` instead.

These are startup sizing budgets, not OS-enforced quotas. Pools stay fixed
during serving; Flare does not resize them as other processes allocate memory.
Startup reports the requested policies, loaded footprint, per-rank cache
budgets, allocated KV tokens, and whether memory, the selected KV ceiling, or
prepared virtual extents limited the pool. TP ranks agree on a capacity each
can back. Runtime memory flags reuse the same prepared package and do not
alter preparation or captured graphs. They are absent
from `ignition export` and the diagnostic `ignition serve` commands.

### Downloading published packages

```sh
flare download ORG/MODEL --gpus 0,1,2,3
flare run ORG/MODEL --gpus 0,1,2,3
flare download ORG/MODEL --revision gb10
flare run ORG/MODEL@gb202 --filename model.safetensors --gpus 0,1,2,3
```

These examples use `ORG/MODEL` as a repository placeholder. Repositories contain
prepared Flare packages, not source Transformers checkpoints. Hardware variants
are separate Hugging Face revisions named `gb10` and `gb202`. Automatic selection
queries the selected GPUs' PCI device identifiers through NVML without initializing
CUDA. GB10 selects `gb10`; known GB202 devices, including RTX PRO
6000 Blackwell, select `gb202`. All selected GPUs must belong to the same family.
An unknown identifier or a driver unable to report it requires an explicit
revision. GPU selection follows `CUDA_VISIBLE_DEVICES` for both commands.
Revision detection runs in a short-lived process so NVML's worker threads cannot
overlap CUDA visibility changes. Without `--gpus`, it uses the first visible GPU.
After reading the package, `flare run` selects its required number of
local GPUs and narrows CUDA visibility before initializing the serving driver.
An explicit GPU list must match the package's local rank count.

An explicit `--revision` overrides `@revision` and hardware detection. It accepts
a branch, tag, or commit, and permits downloading on a machine without NVIDIA
hardware. Online resolution obtains the revision's commit and downloads from
that exact commit. Flare excludes SHA-256 blob filenames and
`*-data-NN-of-NN.safetensors` companions when selecting the primary; multiple
primaries require `--filename`. Referenced companions resolve from the primary's
directory at that same commit. Offline resolution requires every referenced file.
Hardware selection does not infer tensor-parallel size or graph capacities from
filenames. The selected package must match the serving GPU count, topology, and
capacities; the native loader validates its contract.

`flare download` prints only the cached package path on stdout; resolution and
terminal download progress go to stderr. It does not load or execute the package,
or grant publisher trust. `flare run` applies the existing signature checks and
publisher-approval flow after resolution. An existing path, an absolute path, a
path starting with `./` or `../`, or a `.safetensors` path selects local serving.
Download options are rejected for local package paths.

Both commands accept these download options:

| Flag | Meaning | Default |
|---|---|---|
| `--revision REVISION` | Explicit Hub branch, tag, or commit | Detected `gb10` or `gb202` |
| `--filename FILE` | Repository-relative prepared package filename | Sole primary `.safetensors` file |
| `--cache-dir PATH` | Hugging Face hub cache directory | `~/.cache/huggingface/hub` from the OS user account |
| `--token-file PATH` | File containing a Hub token for gated or private repositories | No authentication |
| `--offline` | Resolve only from cached refs and snapshots and skip the publisher identity check; make no network requests. Accepted with a local package path | Off |

The native client uses Hugging Face's `refs`, `snapshots`, and `blobs` cache
layout. It loads regular snapshot files directly. Multi-file snapshots containing
symlinks, including those created by Python's Hub client, use regular hardlinks in
the cache's `flare-packages` directory. These links share the cached file's bytes
and preserve companion paths when the primary path is canonicalized. Filesystems
without hardlink support report an error instead of copying payloads. Local
package companions may also be symbolic links to regular files on other storage
devices; the loader validates and retains the opened targets.

When a file is absent from a snapshot, Flare checks its Hub ETag and size against
the repository's blob cache. An existing blob is hardlinked into the snapshot
without downloading its body. Identical files therefore share disk storage across
revisions of the same repository. Concurrent Flare downloads serialize the blob
lookup and transfer to preserve that sharing. Reuse operates on complete files;
downloads use resumable HTTP transfers through `hf-hub`, without Xet chunk
reconstruction. A cached blob with an unexpected size or file type reports an
error. Package signature verification still occurs when serving.

Offline automatic filename selection considers the files present in that cached
snapshot; use `--filename` to select one explicitly. Cache paths, endpoint, and
authentication are configured explicitly; native commands do not interpret
`HF_HOME`, `HF_HUB_CACHE`, `HF_TOKEN`, or `HF_ENDPOINT`.

### Package snapshots

`ignition export MODEL` writes a unique local snapshot without requiring
`--output`. A recipe's `export_repository` supplies its Hugging Face target;
`--repo-id NAMESPACE/REPOSITORY` overrides it. Recipes without a target use their
local recipe name as the cache namespace. For Qwen Flash:

```sh
ignition export qwen3.8-flash
ignition export qwen3.8-flash --no-weights
```

With the default package cache, the layout is:

```text
~/.cache/flare/packages/
  blobs/<sha256>.safetensors
  models--qwen38/
    snapshots/<snapshot-id>/
      model.safetensors
      <sha256>.safetensors
```

Each payload filename hashes the complete file. Snapshot payloads are read-only
hardlinks to `blobs/`, so identical bytes occupy disk space once. Every primary
contains only the signed model envelope and the storage layout. All model payloads,
including graphs and kernels, follow in separate files no larger than 64 GB
(64,000,000,000 bytes). Size suffixes are binary; packing also applies this
publication cap when the configured limit is `64G`.
`--output FILE` selects another primary location on the cache's filesystem;
a different filesystem requires a package cache on that filesystem. Configure
`paths.package_cache` accordingly.

Weight resources are packed into shards by rank and logical file, separately
from executable and supporting resources. `--no-weights` requires both the
captured allocations and the resulting stored-data shards to be cached.
A missing or changed payload fails export before publishing
the primary. This includes initialized persistent buffers as well as weights.
The flag avoids temporary allocation files and duplicate weight writes. It still
loads and prepares the checkpoint and reads captured storage to establish exact
content identity. Changed graph captures and encrypted executable resources may
produce additional blobs. Regular exports also reuse matching cached blobs.

Multi-host preparation keeps a blob cache on each host. Peers send manifests;
rank zero requests only files absent from its own cache. Native serving distributes
only the payload blobs needed by each rank and reuses blobs across snapshots.

Convert an existing package without a GPU or private signing key:

```sh
ignition convert /models/existing.safetensors \
  --publisher-keys /keys/publisher.json
```

Conversion accepts monolithic, numbered-shard, and snapshot packages, verifies
signed resources and stored data, and preserves the exact signed envelope. Supply
the public publisher profile for a production package; the local development
profile is loaded automatically. `--output FILE` selects a destination. Conversion
leaves its source intact and needs space for payload bytes absent from the cache.
It reads the complete package to verify its contents. Identical conversions reuse
blobs without rewriting them. A resource larger than the selected limit is
rejected because native resources must remain contiguous.

The printed snapshot directory is self-contained. Upload that directory to the
recipe's repository using the Hugging Face upload client; hardlinks appear as
ordinary files. Export and conversion do not upload or change publisher trust.
Keep every referenced payload beside its primary when copying a snapshot. Removing
a snapshot removes its hardlinks; cached blobs remain available for other exports.
There is no automatic blob garbage collection.

### Signing and local development

Final exports always carry signatures and encrypted executable resources.
Configure published exports with all three values:

```toml
[publishing]
signing_key = "/private/flare-signing-key.pem"
namespace = "example-publisher"
official_source = "https://huggingface.co/example-publisher/MODEL"
```

The key must be an owner-only Ed25519 PKCS#8 PEM file. Encrypted keys prompt
for their passphrase in an interactive terminal; unencrypted keys are also
accepted. Runtime publisher profiles contain only the verification key.
`--official-source` overrides the configured repository URL. Setting any signing
identity field requires all three identity fields. `[publishing] shard_size` may
be configured independently and does not select a production identity.

`shard_size = "64G"` is the hard payload-file limit; export/sign/convert
accept `--shard-size SIZE` up to 64 GiB, including each file's header. Zero and
limits at or below 4096 bytes are invalid. A resource larger than the limit fails
export. The primary contains metadata only; all resources reside in immutable
content-addressed blobs. See [package snapshots](#package-snapshots).

`ignition keygen` creates an encrypted production key and public profile.
`ignition key-password` changes the key's passphrase while preserving its identity.
`ignition sign` verifies and promotes an existing development package without
model preparation or GPU access. See [production signing](signing.md) for key
custody, commands, and package promotion.

With no publishing identity configured, Ignition creates and reuses
`development-signing-key.pem` beside its configuration file, with owner-only
permissions. It also writes `development-publisher.json` containing the public
verification profile. These packages use the explicit namespace and source
`local-development`, with no claim to be an official published checkpoint.
When `--publisher-keys` is omitted for a development package, Flare looks for
`~/.config/flare/development-publisher.json`. Explicit profile paths override
this lookup, and `--trust-store` does not relocate it. A missing default profile
leaves verification to embedded or remembered keys; an invalid profile is an
error. The key must match the package's signer and pass signature verification.

There is no unsigned runtime mode and no blanket approval for development keys.

Publisher identity, key, source, license, and verification disclosures appear
only on first acceptance of a signing-key fingerprint and publisher namespace.
On a terminal, the banner animation finishes before those disclosures and the
acceptance prompt. Remembered acceptance suppresses the disclosures, while
every load still verifies signatures and executable resources.

Every signing key is verified the same way. On every load that is not
`--offline`, Flare fetches `publisher.json` from the root of the package's
signed official source on huggingface.co, the profile `ignition keygen` writes.
A file that lists the signing key under the package's namespace verifies the
identity. A file that exists but omits the key rejects the package with no
prompt and no override, remembered or not: the package was not signed by its
claimed publisher, or the key was revoked by removing it from the file. When
the file is absent, the repository is unreachable, or `--offline` is set, the
identity is unverified unless an earlier load confirmed it; loading then needs
the key from an embedded profile, `--publisher-keys`, or remembered acceptance,
and the disclosure names the reason. Embedded profiles and `--publisher-keys`
verify signatures only; they never confer a verified identity. Accepting a key
remembers its verification key and how its identity was established, so later
loads need no profile. A package downloaded from a repository other than its
signed official source prints a warning naming both on every load.

The disclosure names the model license the exporter recorded from the
checkpoint's model card, with a link to the license text in the repository.
Typing the publisher namespace agrees to the terms at that link and trusts the
signing key; the typed string is the key material Helios derives executable
decryption keys from, so acceptance is what decrypts the package's kernels.

To load a locally exported development package using the default public profile:

```sh
flare run model.safetensors --gpus 0
```

For a profile beside a custom Ignition configuration file or from another host,
pass its path with `--publisher-keys PATH`.

First interactive use requires typing `local-development`. For unattended
loading, supply both dangerous-approval flags with the exact fingerprint and
namespace. `flare trust revoke FINGERPRINT --trust-store PATH` removes acceptance;
omitting the path uses the default trust store. No official release key is
required for independent preparation and signing: a publisher is verified by
the `publisher.json` at its official source, as [production signing](signing.md)
describes. To test a production-signed package before publishing its key, pass
the `publisher.json` beside the signing key to `flare run --publisher-keys PATH`.

### 3.3 Configuration file

`~/.config/flare/config.toml`. All keys optional. Written on first export with
the resolved values present as comments.
`[server]` configures `ignition serve`; pass native serving options to `flare run`.

```toml
[server]
host = "127.0.0.1"
port = 8000
api_key = ""                      # Empty: no authentication.

[defaults]
model = "qwen3.8-flash"
max_concurrency = 4

[preparation]
# compile_workers = 24           # Optional CPU process budget per host.
# autotune_effort = "medium"      # off, low, medium, high, max.

[memory]
reserve = "0"                     # GPU memory left for processes started later.

[models."qwen3.8-flash"]          # Per-model overrides of [defaults].
max_concurrency = 2

[paths]
hub_cache = "~/.cache/huggingface/hub"
kernel_cache = "~/.cache/flare/kernels"
package_cache = "~/.cache/flare/packages"
export_staging = ""               # Rank files an export assembles from; empty: beside the package.
cuda = "/opt/cuda"                # Used when installed packages provide no toolkit.
```

`[models.<alias>]` overrides `[defaults]`; flags override both.
Both sections accept `speculative_token_capacity`, overridden during preparation
by `--speculative-token-capacity`. Omitting it uses the recipe declaration when
supplied: every speculative recipe declares 128 target rows.
Both sections also accept positive integer `speculative_graph_budget`, overridden
by `--speculative-graph-budget`. DeepSeek V4.1 declares 64 packed round graphs.
The `[defaults]` graph budget applies only to active packed adaptive learned
speculation; unsupported recipes, `--no-mtp`, and zero draft depth ignore it.
An explicit flag or model-specific graph budget rejects those incompatible modes.
Neither capacity setting changes the graphs in an already exported package;
native exact and maximum verification budgets select only captured capacities.

Default request parameters, meaning the values a request receives for fields
it does not set, such as thinking, reasoning effort, sampling, and output
limit, are a separate mechanism that this specification does not yet define.
Requests receive the checkpoint's template defaults, the recipe's manually
verified model-card sampling defaults, and the engine's output limit.
Sampling defaults are explicit package metadata; config-file sampling values
do not select them. Sampling requests are clamped to the package's inclusive
model-author ranges; parameters without ranges are fixed at their defaults.
Seed remains request-controlled.

`memory.reserve` exists for one case: a process that starts after Flare and
needs GPU memory. Processes already running when Flare starts need no
setting, because supply is free memory rather than card size. It lives in the configuration file and is exported as the package's serving
reserve.

The GPU prefix cache is derived. It holds model states at instruction,
prompt, and answer boundaries so that later requests repeating a prefix skip
recomputing it. A finished exchange can retain its prompt and answer states, so the
engine exposes up to eight snapshot slots per concurrent request, about four
recent exchanges each. Snapshot slots share physical backing with KV pages and
acquire it only when retaining a checkpoint. Unpinned checkpoints are evicted
to admit requests. If active requests or transfers own all suitable backing,
the engine skips optional checkpoint retention. There is no dedicated snapshot
memory fraction or minimum resident checkpoint count.

### 3.4 Configuration boundaries

Configuration assigns each decision to one owner:

| Decision | Owner or interface |
|---|---|
| Checkpoint and revision | `MODEL`, including a local directory or `org/name@revision` |
| GPUs and TP size | `--gpus`; rank count follows the selected list |
| Concurrent requests | `--max-concurrency` |
| Active image requests and feature slots | `--image-concurrency`, then model/default configuration, otherwise `min(4, max_concurrency)` for vision recipes |
| Listener | Ignition and native Flare `--host` and `--port` |
| Context limit | Checkpoint maximum; admission still requires sufficient shared cache capacity |
| KV pages and GPU prefix-cache rows | Derived from available memory, graph cost, and concurrency |
| KV history ceiling | Runtime `--kv-cache-limit` |
| Serving memory budget | Runtime `--memory-limit`, otherwise exported `[memory] reserve` |
| Preparation headroom | 1 GiB for compiled warmup and capture |
| Serving driver margin | 64 MiB after graph upload |
| Prefill chunk | Recipe default, overridden by `--prefill-chunk-size` |
| Speculation | Recipe default, overridden by `--no-mtp` and the advanced speculation flags |
| Branch overlap | Recipe declaration, serialized by `--no-stream-overlap` |
| Weight format, KV precision, page geometry, kernel plans | Checkpoint metadata and recipe preparation |
| Queue, frontend workers, event capacity, body limit | Concurrency and recipe image limits |
| Compiler workers | Automatic memory-bounded sizing; `[preparation] compile_workers` or `--compile-workers` overrides |
| Kernel search effort | `--autotune-effort`, then `[preparation] autotune_effort`, otherwise `medium` |
| Local tuning replicas | `--tuning-gpus`; independent of serving TP size |
| CUDA toolkit | Installed packages, then `[paths] cuda` |
| Served names | Recipe alias and repository ID, exposed through `/v1/models` |
| API key | Ignition configuration or native `--api-key-file`; excluded from packages |
| Sampling distribution controls | Recipe defaults and optional model-author ranges; request values clamp to ranges, or to defaults when no range exists |
| Seed, output limit, and stops | Request fields, with declared defaults for missing values |

Explicit request values take precedence within the declared sampling bounds. Speculation's benefit
depends on native confidence or measured acceptance and execution cost. Reducing the draft limit at
load time retains draft weights; freeing those weights requires preparation
with `--no-mtp`.

Host checkpoint and prepared-input retention share the memory and disk settings
in the [host retention design](prefix-cache-offload.md). Both tiers remain
disabled unless configured. The memory setting includes registered checkpoint
slabs and ordinary retained CPU inputs; bounded transfer, preprocessing, and
decoded-source windows are accounted for separately at startup.

## 4. Derived settings

| Setting | Derived from |
|---|---|
| Tensor-parallel size | Count of `--gpus`, or the smallest recipe-supported size whose estimate fits the first GPUs with enough free memory |
| Context limit | The smaller of the prepared context ceiling and loaded KV capacity after per-request draft-page storage; admission also checks available shared capacity |
| KV pages | Remaining supply after the runtime memory limit or package reserve, engine headroom, weights, graphs, scratch, and execution rows; bounded by `--kv-cache-limit` and prepared virtual extents. Retained snapshot rows borrow page backing; the load display reports total KV capacity |
| Prefix cache | Up to eight snapshot slots per `--max-concurrency` request, acquired on demand from shared KV backing and evicted under pressure |
| Live state | One null row plus the recipe's execution rows per active request slot; GLM verification commits into one live row |
| Decode graph capacities | 1 through `--max-concurrency`, rounded to the recipe's captured set |
| Image feature arena | Prepared image concurrency times per-request feature capacity and embedding width; slots remain owned through request retirement |
| Prefill chunk | The recipe's chunk, 4,096 tokens for Qwen, GLM, and DeepSeek; `--prefill-chunk-size` overrides |
| Speculative catalog and draft counts | Recipe, overridden by `--no-mtp` and the `--mtp-*` flags; the draft count adapts from measured rounds unless fixed |
| Kernel selections | Autotuned during preparation; cached by checkpoint revision, recipe version, GPU, and settings |
| HTTP queue, workers, event capacity, body limit | `--max-concurrency` and the recipe's image limits |
| Compile workers | Automatic or explicit host process budget, divided among local ranks and bounded by host RAM and device memory |
| Sampling defaults | Complete Ignition recipe declaration, manually verified against the model card and exported in typed Protobuf fields |
| Template defaults | The checkpoint template's own defaults, pending the same mechanism |
| Served names | Recipe alias and repository id |

## 5. Command behavior

### 5.1 `ignition export`

Resolve `MODEL`, discover GPUs through NVML, and derive the preparation plan.
Download missing checkpoint weights, then prepare and write a runtime package
in a unique local snapshot under the recipe's export repository.
`--output FILE` selects another destination, appending `.safetensors` if needed;
the destination must not exist. `--no-weights` requires exact cached stored-data
blobs. Export closes the prepared model and exits without starting a listener.
Preparation failures leave no published package.

SIGINT and SIGTERM end preparation immediately. Preparation exit codes are
0 for success, 2 for usage, 3 for fetch, 4 for capacity, and 5 for preparation.

### 5.2 `flare run`

Animate the Flare banner during loading on a terminal. Validate the package,
initialize GPUs, allocate model memory, load weights and compiled kernels,
instantiate graphs, allocate caches, load frontend assets, and start HTTP.
Loading performs no kernel compilation or graph capture. Helios reports each stage with a ten-column bracketed
progress bar, a result, and elapsed time.
Terminal bars use cyan completed segments, dark gray remaining segments, and
half-cell boundaries. Stages without a known total animate a cyan-to-gray
gradient across all ten cells.
The `Verify package` stage times signed metadata authentication before model
loading. `--verify-weights` adds a timed `Verify stored data` stage after publisher
acceptance. Executable resources are verified as part of kernel and graph
preparation.
Memory loading advances by bytes; kernel and execution preparation
advance by resource counts. Stages with no known total show a cycling bar.
Completed stages show `[    OK    ]`; a failed stage shows `[  FAILED  ]`.
A narrow terminal truncates the description while preserving status and timing.
Redirected output contains plain start and completion lines without animation.
After startup, a usable terminal switches to the live metrics dashboard.
Shutdown restores the terminal after draining active execution.

### 5.3 `flare inspect`

Print a compact model summary: identity and publisher verification, architecture,
served names, context and execution capacities, cache history kinds, speculation,
sampling defaults and request ranges, and per-rank fixed memory and graph counts.

The command verifies the signed index using embedded publisher profiles, profiles
provided with `--publisher-keys` (development packages default to
`~/.config/flare/development-publisher.json`), and remembered trust-store keys.
It opens referenced companion files to check their layout, but reads no stored
weights, decrypts no executables, initializes no GPU, and requests no license
acceptance. Verification uses no network access. `--rank N` selects one rank of a
multi-host package, including a copy holding only that rank's payloads.

Detail flags are additive:

| Flag | Additional sections |
| --- | --- |
| `--memory-map` | Recorded allocation addresses, fixed and virtual byte extents, stored payload locations, runtime sizing formulas, address reservations, and execution buffer shapes and strides, for each selected rank |
| `--graphs` | Captured graph capacities, timing estimates, and node and edge counts for each selected rank |
| `--frontend` | Tokenizer, tool and reasoning parsers, markers, vision, frontend assets, and auxiliary embedding table layouts |
| `--all` | All detail sections, including package, model, cache, and speculation settings; JSON also includes the complete serving contract |

`--json` changes the format of the selected sections. With no detail flags it
prints the summary, excluding graph catalogs, allocation maps, frontend details,
and the full contract. Byte counts and addresses remain exact integers; absent
optional values remain null, and explicit zeroes are preserved. Packages contain
prepared serving metadata rather than the checkpoint's raw model configuration.

Text uses aligned values and tables inside light single-line boxes, with cyan
section labels and muted field labels on a terminal. A compact five-line outlined
block rendering of the first served name uses a gold-to-orange gradient and
spaced letters, with the exact name in the Model box's `name` field. Long names
wrap at separators, splitting individual words only when necessary. Names
containing unsupported characters appear only as ordinary text in the Model box.
The recorded model ID is used when no served name is present; its literal value is
also available in JSON and `--all`. Values wrap to the available width between
64 and 120 columns.
Redirected output and `--plain` disable styling; redirected output uses 80 columns.
`TERM=dumb` also disables styling. Tables wrap identifiers rather than truncating
them. The memory map distinguishes virtual reservations and cache capacity limits
from physical backing, which serving selects at load time.

`--unverified` reads only the bounded primary-file metadata, without requiring a
publisher key, verifying signatures, or opening companion files. Output explicitly
labels this mode unverified. It accepts the same section flags, JSON output, and
rank selection. `--publisher-keys` and `--trust-store` cannot be combined with it.
Inspection supports model index versions 4 and 5 without re-export.

## 6. Capacity errors

An estimate that does not fit fails `ignition export` before loading weights with the
requirement, what is available, and the options that satisfy it:

```
error: needs 68.3 GiB weights + 7.0 GiB graphs + 5.2 GiB live state + 0.9 GiB prefix cache
       + 0.8 GiB minimum pages (4 × 8,192 tokens) = 82.2 GiB;
       has 95.6 GiB free − 1.0 GiB headroom − 21.0 GiB reserve = 73.6 GiB.
       Options: [memory] reserve = "12G".
```

The requirement is the smallest estimate the engine accepts: the prefix cache
at its floor and 8,192 tokens of pages per active request. Every listed option
satisfies it on its own; the candidates are a smaller `reserve`, a lower
`--max-concurrency`, `--no-mtp`, and more GPUs. The error cannot occur without
a `reserve`, another process on the GPU, or a model that does not fit at all,
because pages and the prefix cache otherwise shrink to supply.

The estimate is rough. A model near the limit can pass it and exhaust GPU
memory during preparation; `ignition export` reports that as a capacity error too, quoting
the failed allocation.

The same form applies to GPU selection, "needs 2 GPUs with 95 GiB; 1
selected; use --gpus 0-1", and to requests rejected at admission over HTTP,
whose response body names the exceeded capacity and the setting that raises
it.

## 7. Caches and the first run

Compiled kernels and tuning selections live under `[paths].kernel_cache`,
keyed by checkpoint revision, recipe version, GPU, and settings. Repeating
`ignition export` reuses compatible tuning selections and compilation artifacts. Changing
`--max-concurrency`, `--gpus`, or the speculation flags can add graph shapes
and incur partial tuning.

FlashInfer's JIT build files also live under `paths.kernel_cache` (default
`~/.cache/flare/kernels`), in a separate directory for each resolved FlashInfer
installation. Editable checkouts can therefore compile independently even when
they share a package version and GPU architecture. The first use of an installation
can compile its sampling kernels; later launches reuse that installation's files.
Set `kernel_cache = "/data/flare-kernels"` in the configuration's `[paths]` table
to move preparation caches together. Keep the default unless another storage
location is needed.

`ignition export` prints checkpoint resolution, selected GPUs, preparation settings, and
each phase's counts and elapsed time. The phases include weight loading,
`Prepare embedding tables`, kernel tuning, forward compilation, KV allocation,
graph capture, and package export.

After export, launch `flare run PACKAGE` to display the banner and native
loading stages without preparation.

Python-owned preparation phases come from `ignition.progress.phase`; on a
terminal they show an activity bar while running. The enclosing export phase
prints initial and final summaries while native subphase rows animate, with
its Python activity bar paused. `Select GPUs.` names the GPUs by index, followed
by `of CUDA_VISIBLE_DEVICES=...` when the caller restricted them, and reads
`speculation off` under `--no-mtp`. `Allocate KV cache.` reports the pages
allocated. A console too narrow for a row shortens its detail text and keeps
the status and elapsed time. `--verbose` adds each phase's detail lines.

## 8. Model resolution

A recipe declares each qualified checkpoint as an alias, a source, a revision,
and an architecture. Hub sources use repository IDs and pinned revisions;
local sources use absolute checkpoint directories and the revision `local`.
`MODEL` resolves in this order:

1. An existing directory is a checkpoint directory.
2. A string containing `/` is a Hugging Face repository id, optionally
   followed by `@revision`. Repository IDs match declared sources without
   regard to casing. A declared repository at its pinned revision, or with no
   revision, is the qualified checkpoint.
3. Anything else is an alias.

Any other directory or repository is served by the recipe whose architecture
its `config.json` names. For a repository, Flare reads `config.json` and the
safetensors index from the hub without downloading tensors; without
`@revision` it uses the repository's current commit and reports it. The
estimate sizes weights from that index and the recipe's placement. The
recipe's alias remains a served name. An
architecture no recipe supports, or an alias no recipe declares, is a usage
error listing the supported aliases; a repository the hub cannot provide is a
fetch error.

The hub cache follows the Hugging Face hub library's convention unless
`[paths].hub_cache` is set. A hub alias reuses a complete snapshot of its pinned
revision, including a repository cache directory whose name differs only in
casing. A complete exact-case snapshot takes precedence. An incomplete snapshot
does not hide a complete case-variant snapshot; other revisions are never
substituted. If no complete snapshot exists, the alias downloads its pinned
revision. Explicit local directories retain their filesystem casing. A local
alias resolves its directory directly and fails if it is absent.
`kimi-k3-mxfp4-csf` names `/data/models/Kimi-K3-MXFP4-CSF`.
External draft checkpoints use the same hub cache as target checkpoints.
Kimi's Red Hat DSpark checkpoint, `RedHatAI/Kimi-K3-speculator.dspark` at revision
`38a88101e0d46bb22134b9da340f381b954d40d4`, is downloaded there during preparation
unless the target checkpoint supplies it in `dspark/`.
Seven draft tokens and adaptive
verification are the defaults. `--no-mtp` prepares ordinary execution without
downloading or loading the draft checkpoint.

## 9. Recipe obligations

For the command line to stay this small, every recipe declares:

- aliases, repository ids, revisions, architectures;
- each revision's stored tensor bytes by placement, device or auxiliary table
  storage, so the estimate needs no cached snapshot; another checkpoint's placement
  scales from these by its index's total size;
- supported tensor-parallel sizes and the concurrency range;
- a memory cost model: bytes per cached token, bytes per snapshot, bytes of
  live state per active request, and a graph-and-scratch estimate for a cold
  host;
- a prefill chunk and the decode capacities;
- its speculation settings, whether drafts can be copied from shared decoded history,
  the draft model's share of weights and pages, and its costs with speculation
  disabled;
- image limits;
- per checkpoint, optionally, the bytes of weights one decode step reads for
  one request across every rank, dense weights plus the experts one token
  activates at their stored precision, which the serving contract records as
  `decode_weight_bytes` and the dashboards multiply by verification steps per
  second to show weight bandwidth.

Model-specific validation lives here, not in the entrypoint. A recipe that
cannot honor the settings reports a capacity error in the section 6 form.

### Experimental RoCE ring configuration

`ignition export` and `ignition serve` accept `--roce-ring` or `--roce-chain` with `--ranks N`;
`flare run` and `flare orbit run` accept it with a multi-host package. Rank
order defines the ring, including the cluster's node order when agents launch
ranks. The transport accepts 2..16
ranks, while recipe validation retains each model's supported TP sizes. This
option is recorded in the preparation plan, sent to preparation peers, and
forwarded to serving peers. It changes native transport routing, not the
captured collective operation contracts. The same routing choice applies to
checkpoint and prepared export TCP transfers; manual preparation needs only the
rank rendezvous and the shared preparation token. `--transfer-streams N` selects
1..64 payload streams per peer (default 8) with direct or ring routing. Peers
connect using `ignition join`. Rank zero receives missing export files through
the authenticated streams before assembling and signing the package.

`flare orbit run` starts native serving ranks, which distribute package bytes
through CX7. `flare orbit push --to NODE,...` prestages through agents when the
target list contains one node per package rank. Ring prestaging requires that
exact list in physical ring order. Both Orbit commands accept
`--transfer-streams N` (1..64, default 8). Named groups retain the stream count;
an explicit value overrides it. Agent control and prestaging use paired
identities, encrypted sessions, and signed grants without additional
configuration. Serving distribution uses plain TCP and normal package trust.
See [multi-host serving](multi-host.md) and [RoCE ring transport](roce-ring.md).

`--prompt-logprobs` is an opt-in preparation flag for `export` and
development `serve`. It captures separate scored prefill graphs; default exports
contain no prompt scoring storage or graphs. See [the logprob contract](logprobs-design.md#prompt-scoring).

## 10. Non-goals

Topology files, model families, launch override blocks, per-host environment
maps, container orchestration, multiple models in one process,
and serving an architecture no recipe supports.
Cross-node tensor parallelism across supported hosts is in scope. The
`flare orbit` commands and `flare agent` own host discovery, pairing, package
distribution, and rank process management. `flare run` executes its assigned
rank. See [Orbit](orbit.md) and [multi-host serving](multi-host.md).
