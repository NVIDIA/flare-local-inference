# Model development

Recipes in `ignition/recipes/` own model composition, checkpoint mapping, state
layouts, and graph preparation. Shared execution mechanisms belong in
`ignition/components/`. Follow [repository guidance](../AGENTS.md),
[preparation development](preparation-development.md), and the relevant
[model preparation and serving instructions](model-preparation.md).
Performance changes also require [GPU performance work](performance-work.md).

## Canonical model structure and semantics

- Treat Transformers as the canonical source for model structure and semantics.
  Preserve its forwards where their tensor layouts, state handling, and emitted
  GPU work fit the serving contract. Avoid independently reproducing model
  logic that can be reused directly.
- Inspect the installed model's actual call sites before choosing an extension
  point. Prefer attention, mask, expert, kernel, and loading interfaces when
  their contracts fit. Interface availability does not imply that every model
  uses it or that it controls the entire operation we need to optimize.
- Replace the smallest coherent module that gives control over the required
  dataflow. A whole attention module may need replacement to avoid expanding
  latent KV or constructing dense masks before the attention dispatcher. A
  whole MoE module may need replacement to overlap shared and routed experts.
- Use instance-local module replacement or supported construction hooks when
  sufficient. Avoid global monkey patches and Transformers forks without a
  concrete need. Do not force b12x through a Transformers integration that adds
  complexity without improving reuse or execution.
- Judge reuse by the GPU work it records and the semantics it preserves.
  Python dispatch during preparation and capture is not per-token overhead
  during native graph replay. A canonical forward is not inherently too slow,
  and a rewritten forward is not inherently faster.

## Sampling and frontend metadata

- Agents adding or changing a recipe must read the model card and manually verify
  its recommended sampling parameters for the default serving mode. Declare the
  complete defaults in the Ignition recipe and cite the recommendation at that
  declaration. Never infer sampling policy from `generation_config.json`,
  `config.json`, Transformers defaults, or another model. Model-card recommendations
  take precedence over conflicting config files; unspecified filters are disabled
  and penalties are neutral. Declare optional inclusive sampling ranges only from
  the model author's recommendations. Clamp request distribution parameters to
  those ranges; an absent range fixes the parameter at its declared default.
  Seed remains request-controlled.
- Ignition validates checkpoint configuration for structural data, including
  EOS and template tokens, image preprocessing, and vision position geometry.
  Export only the fields Helios consumes in first-class Protobuf messages.
  Runtime assets are the tokenizer and chat template; do not package raw generation,
  tokenizer, processor, or model configuration files. Helios consumes prepared
  settings without per-model sampling policy or config-file interpretation.

Gemma 4 exports Google's canonical chat templates for every supported publisher
and quantization. `ignition/recipes/gemma4/assets/chat_template.google-e2b-e4b.jinja`
serves E2B/E4B; `chat_template.google.jinja` serves 12B/26B-A4B/31B. The sources are
[Google E4B at ee0ef602](https://huggingface.co/google/gemma-4-E4B-it/blob/ee0ef6023621cff504d758262d4e04895a5af4a2/chat_template.jinja)
and [Google 26B-A4B at 4d7ae498](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/4d7ae4984b7db7de8f8457170b3f1a419ee76d52/chat_template.jinja),
with message-boundary annotations. Preparation selects by model hidden size:
1536/2560 for E2B/E4B and 2816/3840/5376 for 26B-A4B/12B/31B. The larger models
append an empty thought channel when thinking is disabled. Publisher-supplied
templates do not select rendering behavior. Helios normalizes JSON-string tool
arguments into objects before rendering. Tests compare the assets with the pinned
Google sources and exercise prompt bytes, reasoning, tool rounds, and cache boundaries.

## Weight formats and arithmetic precision

- **Read quantization from checkpoint metadata.** Use `config.json` declarations
  to choose supported weight formats. Keep geometry and encoding validation
  explicit; do not hardcode a checkpoint's per-layer precision assignments.
  Ordinary NVFP4, CSF NVFP4, and MXFP4 expert storage use the same b12x MoE
  execution API. Dense weight formats likewise share `blockscaled.mm`; format
  handling belongs in loading and preparation, not recipe inference branches.
  Reject fusion of NVFP4 projections except routed MoE w1/w3, whose dedicated
  loader validates the shared scales. Never silently rescale or requantize
  weights to make a fusion possible.
- **Separate weight storage from arithmetic precision.** Keep BF16 weights in
  BF16 when the required accumulation/output is FP32; use
  [UnquantizedProjection](../ignition/components/linear.py) for that contract. Do not
  expand router weights to FP32 merely because routing requires FP32 logits.
  Preserve genuinely FP32 checkpoint weights and model-required arithmetic.
- **Let b12x select activation precision and kernels.** Use
  `QuantizedProjection(..., activation_mode="auto")` where the model's numerical
  contract permits it. Packed FP4/FP8 weights do not imply quantized activations
  are fastest at every row count. Keep production autotuning enabled and do not
  pin tile shapes, split-K, MoE plans, or a GEMM implementation to conceal an
  integration defect. Check what the selected plan actually executes.

## Projection fusion, overlap, and tensor parallelism

- **Fuse projections sharing an input.** Pack compatible weights once during
  loading, including their quantization scales and TP slices. Candidates include
  Q/K/V, MLP gate/up, MLA query/KV low-rank inputs, and recurrent attention's
  Q/K/V, beta, and forget-gate input projections. Preserve each output's model
  semantics and calibrated scales. Inspect the full dependency chain rather
  than reproducing one kernel call per canonical module.
- **Respect layouts after fusion.** A column slice of a fused output can be
  contiguous for batch one and strided for multiple rows. Check downstream b12x
  input, output, stride, and alignment contracts for every graph capacity.
  Allocate reusable staging storage where needed; do not introduce runtime
  allocation through an implicit `.contiguous()`.
- **Overlap independent branches.** Shared experts should be considered for
  overlap with routing and routed experts; recurrent output-gate projections
  can overlap independent input projections. Use
  [ConcurrentSharedExpert](../ignition/components/expert_overlap.py) and
  [ConcurrentProjectionChain](../ignition/components/projection_overlap.py) where their
  contracts fit. Measure the relevant row capacities: branches competing for
  bandwidth or SM capacity can make overlap slower. Graph node construction
  order also affects scheduling. `ConcurrentSharedExpert(..., defer_launch=True)`
  records routed kernels before shared kernels while preserving the fork and
  join dependencies; measure this ordering in the complete model graph.
- **Make overlap safe through compilation.** Declare every concurrently live
  plan with [ScratchArena.register_parallel](../ignition/preparation/scratch.py).
  Sequential work reuses storage; concurrent work gets disjoint storage sized
  from the tuned plans. Refresh bindings after arena resizing. The fork must
  follow input production, and the join must precede output consumption or
  input/scratch reuse. Custom ops must expose reads, writes, aliases, and join
  dependencies so compilation preserves those lifetimes.
- **Shard the vocabulary projection under TP.** Use
  [PreparedVocabularyProjection](../ignition/components/vocabulary.py) to project
  contiguous vocabulary partitions and gather logits before designated-rank
  sampling. Keep all ranks participating, including resume and other paths
  that produce logits. Respect the collective's supported geometry and retain
  its resources through native execution. A large head on one rank can dominate
  otherwise sharded decode.
- **Fuse residual and normalization boundaries.** Use b12x fused residual,
  normalization, activation, and mHC operations when their rounding matches the
  model contract. Consider adjacent layers as well as operations within a
  layer; [PreparedMHC](../ignition/components/mhc.py) supports fused post/pre mixing.
  Avoid materializing an intermediate solely because a Python module ends.

## Speculative decode

- **Capture zero-draft rounds as decode.** Verification width one contains only
  the pending target token. Use ordinary decode attention and direct cache updates;
  do not inherit prefill kernels, speculative staging, or draft rollback work
  merely because the graph belongs to the speculative catalog. Retain the draft
  context updates and recurrent metadata required to alternate with wider rounds.
  Compare the complete width-one graph against ordinary decode, attribute any
  extra cost, and test transitions in both directions with partial/inactive rows
  and cache-boundary crossings. Apply this check in each recipe's existing
  preparation and tests.
