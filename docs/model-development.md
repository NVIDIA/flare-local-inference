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

## Prompt renderers and checkpoint boundaries

Every model integration must provide prompt rendering, tokenization, image
ordering, and reusable-prefix endpoints through the shared native frontend.
These requirements apply to text Jinja, segmented Jinja, and programmatic Rust
encoders. A plausible generated answer does not establish frontend correctness.
Follow the [frontend metadata contract](../helios/model-contract.md#frontend-implementations)
and [native frontend interfaces](../helios/README.md#native-frontend).

### Canonical prompt semantics

- Identify the exact checkpoint tokenizer and authoritative template or encoder
  revision. Preserve prompt bytes, token IDs, special-token treatment, whitespace,
  tool serialization, reasoning history, generation headers, and continuation
  behavior. A deliberate inference transformation needs an explicit semantic
  requirement and independent reference tests.
- Render the normalized conversation supplied by the shared request path.
  Protocol decoding, instruction lowering, and tool-result normalization belong
  in shared code. Renderer selection describes prompt syntax; it must not select
  scheduling, cache policy, or model execution behavior.
- Return image sources in the same order as their rendered placeholders. If
  normalization reorders tool results, move each result's images with its text.
  Preserve the tokenizer's distinction between ordinary text and special tokens,
  including literal special-token spellings in user and tool content.
- Explicit request options override packaged defaults. Validate unsupported
  options and malformed content explicitly. Preserve previously exported
  packages' renderer selection, absent-field defaults, and packaged semantics.

### Complete-message endpoints

- Render the conversation once and record endpoints while producing its output.
  Tokenize each required text or ordinary/special-token segment once. Do not
  recover boundaries by rendering or tokenizing every conversation prefix,
  searching for message text, or inserting marker text into the prompt.
- Record the completed preamble and every completed normalized message in output
  order, including system, assistant, and tool messages where supported. A merged
  user run has the endpoint of the merged message; reordered results have
  endpoints in rendered order. Include closing wrappers and separators belonging
  to a completed message, and exclude the open assistant generation header.
- Text renderers report UTF-8 byte positions, not character counts or token IDs;
  segmented Jinja reports segment counts as specified below. Shared boundary
  mapping uses the full encoding's untrimmed offsets and accepts
  an exact cut only when every preceding token ends at or before it and every
  following token starts at or after it. Omit an exact cut crossed by a BPE token
  or overlapping Unicode spans; do not round it to a nearby token or cache page.
- The frontend's final token endpoints must be sorted, unique, and strictly
  inside the prompt. Continuation truncation must exclude later endpoints. Image
  expansion must shift endpoints with the token stream without splitting an
  atomic image span. The scheduler owns prompt/terminal checkpoints and complete
  state capture; renderers supply positions, not cache-management instructions.

### Jinja integration

- Package an annotated recipe template. Removing its mark annotations must
  reproduce the authoritative template bytes, including any explicitly supported
  inference transformation. Preserve Jinja whitespace-control behavior and test
  actual rendered output; source equality alone does not validate placement.
- Text templates use `{% mark %}` to record a zero-output byte position.
  Templates producing token segments use `{% mark expression %}`, where the
  expression is the number of completed segments. For example,
  `{% mark ns.segments | length %}` refers to segment entries, not bytes in the
  serialized JSON representation. Preserve each segment's `allow_special` value.
- Place marks at root output sites after a completed preamble or message.
  Marks inside macros, captured assignments, or filter blocks are forbidden;
  mark the root call site after the helper returns. Handle each output branch,
  including tool-only messages and empty content with nonempty framing.
- Verify the recipe exports the annotated asset and Helios loads it through the
  shared compiler. Do not bypass the missing-annotation error or add a broad
  template fallback. Compatibility substitutions must match the exact template
  bytes of an existing supported package.

### Programmatic encoder integration

- Use the authoritative encoder where available. Expose normalized-message byte
  endpoints from its actual append loop, and adapt them through the shared
  tokenizer/boundary path. Avoid a parallel implementation of the encoder's
  formatting or normalization solely to discover positions.
- Carry positions through system insertion, adjacent-message merging, tool-result
  sorting, and wrapper emission. Keep image sources, message text, and endpoint
  ownership together through those operations. Both Jinja and native encoders
  must satisfy the same complete-message and token-boundary invariants.
- Maintain required upstream changes as focused commits in a pinned dependency
  fork. Keep related crates' revisions and shared types coherent, preserve the
  upstream license, and document dependency updates. Compare against the
  independent upstream renderer when changing prompt behavior; calling the same
  patched encoder through another adapter is not an independent reference.

### Content-block checkpoint hints

For requests carrying advisory block hints, every supported rendering mode must
satisfy these requirements:

- Decode protocol-specific fields into neutral message/block positions before
  rendering. Hints must not alter prompt bytes, token IDs, image order, reasoning
  state, or explicit request options, and must not reach model-visible output.
  Requests from protocols that do not supply hints retain their ordinary behavior.
- Preserve block identity through normalization and rendering. Carry positions
  through byte-preserving concatenation, captures, macros, segment construction,
  and trimming where their output positions remain defined. Repeated equal text
  must not cause a hint to move to a different block. Never guess a position after
  an unmappable transformation.
- Cover text blocks, supported image blocks, and whole tool results. A tool-result
  endpoint includes its closing wrapper; an image endpoint must survive atomic
  image expansion. Text-only coverage does not establish image or segmented
  renderer support. Report unsupported cases explicitly rather than silently
  treating an accepted hint as implemented.
- Keep advisory cuts distinct from exact message marks. A hint may use the last
  safe complete-token prefix before its byte endpoint; it must not weaken exact
  message-boundary validation or change scheduler/cache policy.
- Preserve an unhinted path without hint-specific string copies or per-value
  hint allocations. Measure unhinted latency and allocations when changing shared
  rendering machinery; protocol-specific decoding alone does not isolate its cost.

### Required validation

1. Compare the production renderer with the pinned authoritative reference using
   the actual template/encoder and checkpoint tokenizer. Check rendered bytes,
   token IDs, image order, supported options, and expected errors. Exercise
   system/developer lowering, user and assistant turns, reasoning on/off where
   supported, tool choices, multiple/reordered results, and continuation.
2. Test boundary behavior with empty and repeated content, UTF-8, whitespace/BPE
   joins, merged users, wrappers, generation headers, and image expansion.
   For block hints, include a stable history plus a changing suffix inside one
   user message; assert the expected extra cut and identical hinted/unhinted
   prompts. Exercise captures and macros with the real recipe template, and
   segmented and image paths where supported. Synthetic tokenizers are useful
   edge-case tests but do not establish checkpoint-tokenizer parity.
3. Exercise the native request preparation used by each supported HTTP protocol
   and load the exported package's frontend assets/settings. Validate older
   packages when changing their loading or default semantics.
4. Demonstrate repeated-prompt and changed-suffix cache reuse through the actual
   service, including ordinary and speculative execution where supported. Check
   consumed-prefix accounting and continuation correctness. Frontend CPU tests
   alone do not establish runtime cache reuse or a serving performance gain.
   Report missing checkpoints, unrun hardware paths, and skipped tests explicitly.

Use [frontend fixtures](../tests/frontend),
[native boundary tests](../helios/tests/message_boundaries.rs),
[native encoder/request tests](../helios/tests/deepseek_requests.rs), and
[HTTP protocol tests](../tests/server/test_protocols.py) as starting points.
Keep ordinary tests and measurements; do not add development audit artifacts.

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
