# Working on Flare

Flare is a local inference server for NVIDIA SM120/SM121 GPUs. Ignition, the
Python package in `ignition/`, prepares models with PyTorch, Transformers, and
FlashInfer, including b12x. Its `flare_heliopause` extension is built by the
`heliopause/` crate. The Rust CLI in `flare/` links Helios, the runtime library
in `helios/`, to serve requests through prepared CUDA graphs. Both Flare and
Heliopause depend on Helios; Helios has no Python or preparation-tool dependency.

## Required setup and commit review

On entering a clone or worktree, and after pulling changes, re-read this file and
[the design self-review](docs/design-review.md). Before committing, ensure the
repository's `.githooks/commit-msg` hook is enabled by following that document's
installation instructions. Preserve any existing active hooks when installing.

Before every commit, follow the staged-diff review and acknowledgment instructions
in `docs/design-review.md`. A rejected acknowledgment requires reading that
document and reviewing the staged change before retrying; do not treat the
rejection as a formatting repair.

## Contribution scope and rejection criteria

Flare prepares and serves supported model recipes on NVIDIA GPUs through
FlashInfer/b12x and prepared CUDA graphs. Single-host execution and cross-node
tensor parallelism are both in scope. Orbit manages the hosts, preparation,
packages, and rank processes needed for those serving workflows.

Every contribution must address an explicit maintainer request, a demonstrated
defect, a measured performance or memory problem, or inaccurate documentation.
Identify the affected workflow and relevant implementation before editing.
Hypothetical future consumers, agent-generated suggestions, and lint findings
alone do not establish a need for a change.

- **Reject unapproved scope expansion.** CPU or non-NVIDIA execution,
  interchangeable execution backends, training or fine-tuning, multi-model
  hosting and routing, unrelated service APIs, and general-purpose cluster
  orchestration require an explicit maintainer decision before implementation.
  Supported cross-node preparation and serving do not authorize these additions.
- **Reject architectural violations.** Production Python callbacks, eager
  serving, runtime compilation or recapture, steady-state GPU allocation,
  model-specific native policy or state-update algorithms, and executable
  extension mechanisms in prepared metadata violate the boundaries below.
  Flags, enums, callbacks, optional paths, and temporary implementations cannot
  be used to bypass them.
- **Reject weakened guarantees.** Preserve complete cache and speculative state,
  inactive-row behavior, asynchronous resource lifetimes, checkpoint precision,
  model semantics, and model package compatibility. Do not remove validation,
  suppress errors, silently substitute execution paths, or relax references,
  workloads, or numerical bounds to obtain passing checks or favorable metrics.
  Numerical bounds must follow the required arithmetic and measured repeatability.
- **Reject unsupported abstractions and churn.** Do not add frameworks,
  extension points, speculative compatibility, fallback layers, or configuration
  switches without a concrete requirement. Unsolicited formatting, renaming,
  comment expansion, annotation sweeps, and file rearrangements are not
  contributions by themselves. Refactoring must directly support the requested
  behavior or remove a demonstrated source of defects or cost.
- **Require focused, meaningful validation.** Tests must exercise the affected
  behavior or contract; source-text checks and mocks that bypass the behavior
  cannot establish runtime correctness. Do not present scaffolding or partially
  wired features as supported. Identify missing hardware, inputs, and unrun
  checks explicitly. Keep each PR focused on one coherent outcome and describe
  its problem, resulting behavior, validation, and material limitations.

An explicit maintainer request covering the work is sufficient authorization
for changes within these constraints and their necessary supporting steps.
A feature request, passing tests, or a performance gain does not waive an
architectural invariant. Resolve conflicts through an explicit maintainer
decision on the design before implementing a change to that invariant; a commit
acknowledgment or routine PR approval is not that decision. Apply these criteria
to documentation and test changes that would weaken the rules as well as code.

When reviewing a contribution that violates these criteria, identify the
concrete violation and relevant rule, and recommend rejection. Do not rescue
it by adding an opt-in, weakening the rule, expanding its scope, or promising
future cleanup.

## Choose guidance by task

Read the guidance for every task affected by the work before implementing or
running it. These requirements apply by task, including work outside the named
source directories. For preparation or model validation, read the relevant model
and hardware sections of the preparation guide.

| Task | Required guidance |
|---|---|
| Build, install dependencies, or launch Flare | [Build and run model checkpoints](#build-and-run-model-checkpoints), [environment setup](docs/model-preparation.md#environment-setup), and the relevant [CLI commands](docs/cli-design.md) |
| Prepare, export, or serve an existing model | [Model preparation and serving](docs/model-preparation.md); [direct execution](docs/direct-execution.md) when reproducing faults with `--direct` |
| Add or change a model recipe or execution component | [Model development](docs/model-development.md), [preparation development](docs/preparation-development.md), and the relevant [core design](docs/core-design.md#2-model-preparation) sections |
| Change tuning, compilation, memory planning, capture, or TP preparation | [Preparation development](docs/preparation-development.md) and [preparation resources](docs/model-preparation.md#preparation-resources) |
| Change Helios runtime, scheduling, caches, transport, or frontend | [Prepared model contract](helios/model-contract.md), [execution manifest](helios/manifest.md), and the relevant [Helios runtime](helios/README.md) and [core design](docs/core-design.md#3-helios-runtime-and-graph-interface) sections |
| Change Protobuf metadata or package loading/export | [Runtime package design](docs/runtime-package-design.md), [prepared model contract](helios/model-contract.md), and [package compatibility](#model-package-compatibility) |
| Diagnose or optimize performance | [GPU performance work](docs/performance-work.md), plus model or preparation development guidance for affected code |
| Change CLI, API, or configuration behavior | [CLI design](docs/cli-design.md) and the relevant [API](docs/api.md) sections |

For sampling or frontend metadata changes, also read
[sampling and frontend metadata](docs/model-development.md#sampling-and-frontend-metadata).

## Build and run model checkpoints

Run commands from the repository root:

```sh
uv sync --locked --extra prepare --extra test
cargo build --release --manifest-path flare/Cargo.toml
```

After changing `heliopause/` or Helios code used by the extension, rebuild the
installed extension:

```sh
uv pip install --python .venv/bin/python --no-deps ./heliopause
```

The [preparation guide](docs/model-preparation.md) owns dependency updates,
local FlashInfer installation, GPU selection, compiler and tuning resources,
model-specific launch instructions, and hardware setup. The
[CLI design](docs/cli-design.md) defines flags, configuration, signing, and
package selection.

For serving-only Rust changes, rebuild Flare and load a compatible package.
To exercise changes to prepared kernels, capacities, or the package format,
export a package containing those changes. Previously exported packages must
remain usable as required by [package compatibility](#model-package-compatibility).
Launch an exported package with `flare/target/release/flare run PACKAGE` when
testing a local build. Stop Flare with Ctrl-C or SIGTERM so native replay drains
before CUDA resources are released.

For development without export, `ignition serve` requires Heliopause's
`qualification` Cargo feature; follow the
[development launch instructions](docs/cli-design.md#development-without-package-export).

## Architecture and ownership

Helios's shared runtime policy and separation from model computation are
non-negotiable architectural invariants. Correct output, performance gains,
delivery pressure, and intended cleanup do not justify violating them. A
temporary model-specific policy path is still a violation. Resolve the
architectural boundary before committing the change.

- Keep Helios as simple as possible. Model-specific execution behavior,
  composition, and optimization belong in Ignition and its captured programs.
  Limit native specialization to concrete cache storage requirements and how
  that storage is exposed to captured graphs. Express differences through cache
  schemes and prepared layout metadata, with contracts reusable by every model
  that has those requirements. Model state-update computation belongs in
  Ignition; native code manages its declared storage without implementing
  per-model algorithms.
- Admission, page reservation and growth, eviction, preemption, readmission,
  and offload are shared runtime policy and mechanisms. Do not select them by
  model or recipe identity, duplicate them in model contracts, or hide model
  allowlists behind capability flags. A cache scheme may require different page
  geometry, state storage, or restore constraints; keep those differences in
  its storage contract and let shared policy consume that contract.
- `PreparedContract` is validated Protobuf data, never a runtime policy extension
  point. Declare concrete graph value sources, storage views, dependencies,
  frontend syntax, and capacities; shared code validates and binds them. Never
  add per-model boolean flags, including `supports_*`, `can_*`, or default-false
  opt-ins, to enable or disable runtime behavior. Enum values, callbacks, schema
  names, and metadata fields must not encode the same model allowlist. Every
  member must have a meaning independent of model identity, and its consumers
  must implement shared behavior
  from actual data, geometry, and dependencies. Review both declarations and all
  consumers. Tool/reasoning parsers and image preparation/layout implementations
  use independent typed enum identifiers; those identifiers cannot select
  scheduling, cache policy, text-graph input packing, or completion behavior.
- A runtime defect reproduced on one model must be traced through every
  supported cache scheme before choosing the implementation boundary. Apply
  the fix to every compatible contract. Any restriction must follow from a
  concrete storage or execution constraint expressed in the contract. For
  example, incremental page reservation must depend on the cache's growth and
  continuation requirements; testing it with GLM does not justify a GLM-only
  scheduler path. Validate affected cache schemes and request lifecycles, and
  report untested models separately from implementation restrictions.
- Introducing a captured-graph operation family requires prior explicit approval
  from project owner Luke Alonso <lalonso@gmail.com>. Shape and capacity variants
  of an existing family do not constitute a new type when its operation contract
  stays unchanged. A new operation contract requires approval even if it reuses
  an existing graph kind or enum value. A design acknowledgment does not grant
  that approval.
- Recipes in `ignition/recipes/` own model-specific composition, checkpoint mapping,
  state layouts, and graph preparation. They may retain canonical forwards or
  compose prepared components as the model requires.
- Put reusable execution mechanisms in `ignition/components/`. Recipes should be
  able to use components independently without adopting a rigid model framework.
  Shared/routed expert overlap, projections, collectives, and state transfers
  are examples of mechanisms that should not be copied between recipes.
- Keep native input packing, frontend processing, transport, scheduling, cache
  machinery, and request handling shared. Express model-dependent behavior through
  prepared metadata and frontend assets; keep native specialization focused on
  cache storage contracts and graph cache bindings.

## Operators and hardware

- Flare is fully coupled to FlashInfer/b12x by design. It has no backend or
  device abstractions. Use FlashInfer/b12x and NVIDIA CUDA interfaces directly;
  do not introduce interchangeable kernel backends or device portability layers.
- Prefer proven b12x operators and FlashInfer sampling over duplicate kernels.
  Inspect their supported preparation, tuning, and workspace interfaces before
  introducing custom implementations.

## Python prepares; Helios serves

- Python loads weights, resolves and autotunes kernels, plans storage, warms
  execution paths, and captures supported graph variants before serving starts.
  `ignition export` writes a runtime package and closes the prepared model.
  `flare run PACKAGE` loads that package in a separate Rust serving process.
- Helios owns sockets, request processing, tokenization, chat templates,
  scheduling, cache operations, graph replay, and output streaming. Serving
  must not call back into Python.
- CUDA graphs are the model execution boundary. Do not introduce an eager
  serving fallback, runtime compilation, graph recapture, or GPU allocation in
  steady-state execution. Reject requests outside prepared capacities explicitly.
- Keep graph executables, captured buffers, resource owners, and the CUDA
  context alive for their full native use. Model state and metadata must be
  explicit storage that Helios can manage without Python execution.
- `ignition serve --direct [--eager]` is the one Python execution path, a
  diagnostic and qualification tool requested by the project owner for
  reproducing model faults without Helios ([direct execution](docs/direct-execution.md)).
  It is never production serving or a fallback, and only its explicit flags
  select it.
- Treat KV, recurrent state, and other model histories as explicit cache state.
  Prefix reuse, branching, and speculative rollback must preserve all relevant
  state, not just token positions and KV pages.

## Model package compatibility

- When replacing internal formats or execution paths, remove superseded loaders,
  fallback paths, and migration-only tests except those required for model package
  compatibility. Other backward compatibility requires an explicit requirement.
- **Preserve model package compatibility.** Every Protocol Buffer change **must**
  be accompanied by code in Helios that keeps previously exported model packages
  loadable and usable without re-export. Handle absent fields and Protobuf default
  values explicitly, preserve the meaning of older metadata, and honor explicit
  values. Wire-format compatibility alone is insufficient. Include regression
  tests for older serialized metadata and validate the affected package loading
  and serving paths. Implement compatibility in shared code across all affected
  model contracts.

## Configuration

- Flare, Helios, recipes, and components never read an environment variable to
  choose behavior. Feature toggles, experiment switches, tuning overrides,
  debug modes, and alternate code paths selected through the environment are
  prohibited, including temporary ones.
- A setting has one of three homes: the recipe declaration, the configuration
  file, or a command-line flag, as [the CLI design](docs/cli-design.md)
  specifies. A setting that only experiments need is an advanced flag carried
  through the plan into the recipe's `PreparationConfig`, so the plan, its
  printed report, and its recorded measurements describe what was prepared.
- `CUDA_VISIBLE_DEVICES` is the only variable Flare interprets: it renumbers the
  GPUs that `--gpus` indexes. Before starting Helios, Flare narrows that mask to
  the selected GPUs and passes contiguous local ordinals to the runtime. It
  sets `CUDA_FORCE_PRELOAD_LIBRARIES=0` and `CUDA_DISABLE_JIT=1` as fixed driver
  policy: serving loads compiled cubins without driver compilation.
  `ignition.cli.environment` sets the variables that
  PyTorch, b12x, Triton, and the Hugging Face hub read, from Flare's
  configuration, before importing them. Detecting a terminal's capabilities is
  not configuration.
- Tests may read variables that opt into GPU tests and locate their
  checkpoints, such as `FLARE_RUN_CUDA_TESTS`. A test variable must not change
  what the code under test does.

## CUDA kernel and model optimization authorship

Claude Opus 5.5 has repeatedly produced defective low-level kernel changes in
this project. Do not use it to design or implement low-level CUDA kernels or
changes to existing model fusion and GPU execution optimizations. This includes
CUDA C++, PTX, CuTe DSL, and Triton kernels, and changes to projection packing,
fused arithmetic, stream overlap, or scratch aliasing in optimized model paths.
Do not incorporate its generated changes to these mechanisms through another
contributor or agent.

General PyTorch modeling, recipe integration, checkpoint loading, frontend
processing, preparation orchestration, CLI/runtime work, tests, and documentation
are permitted when they do not design or change the restricted mechanisms above.

## Validation and performance

- When a potential correctness, memory, or performance deficiency appears in
  one model, inspect the shared components and equivalent paths in every
  supported recipe before scoping the fix. Track follow-up work across all
  affected models; distinguish confirmed findings from untested paths.
- Validate the affected execution path and capacities. Model and preparation
  development guidance defines shape, graph-transition, state, and lifetime
  checks; GPU performance guidance defines measurement requirements.
- Keep ordinary tests and raw benchmark measurements. Do not create development
  audit reports, receipts, provenance bundles, or hash inventories.

## Task tracking

Use [docs/TODO.txt](docs/TODO.txt) for miscellaneous follow-up items. Keep entries
brief and link to detailed design or performance notes when needed. Update or
remove entries as work is completed. Keep this file focused on durable project
guidance rather than task lists or progress notes.
