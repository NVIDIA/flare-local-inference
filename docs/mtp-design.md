# Multi-token prediction and speculative execution

Flare's speculative engine supports the embedded MTP modules in
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` and
`local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD`, and the DSpark drafter in
`deepseek-ai/DeepSeek-V4.1-Flash` on SM120/121. Python prepares
weights, kernels, storage and CUDA graphs; Rust schedules requests and launches
graphs. Acceptance and token feedback remain on the device. Speculative serving uses a versioned token-vector completion contract.
Ignition prepares the recipe's learned drafter unless `--no-mtp` is given. The advanced flags
select the draft-token maximum, the default verification policy, and the
proposal source. Qwen's `--mtp-proposal-source copy` drafts from shared decoded
history without loading an MTP module. `--history-drafts` adds those proposals
to the learned MTP source and is disabled by default.

With copying enabled, each loaded model retains the most recent 65,536 committed
generated tokens across requests in a host-memory ring. Exact eight-token and four-token indexes
match the active request's suffix, preferring eight-token matches and then the
most recent occurrence. They return up to seven observed continuation tokens.
Per-response links preserve sequence boundaries under interleaving and
preemption. Prompt ingestion, prefix restores, and disk checkpoints do not
populate the ring; only validated completion outputs, including accepted drafts
and target samples, extend it. Rejected proposals and failed completions do not.
The ring survives request completion and is discarded when the model unloads.
Queries may use prompt context without inserting it into the shared corpus.

The design combines adaptive verification budgets, native MTP proposals and
continuations copied from shared decoded history. Compact recurrent commit and token
trees are additional capabilities with explicit operator requirements. A
separately trained parallel drafter is optional; it is not needed to use the
checkpoint's MTP weights.

The [core design](core-design.md) defines ownership, admission and graph
lifetime.

## Qwen checkpoint model and reference

The exact checkpoint has one MTP decoder layer, 512 routed experts with top-10
routing, four 2,560-wide hidden streams, and a hyperconnection output mixer.
Its sparse-attention layer has 24 query heads and two KV heads of dimension 256.
The MTP decoder has no GDN or PLE layer. Target verification still executes all
36 GDN layers, 12 QSA layers and target PLE operations.

| MTP storage | Contract |
| --- | --- |
| 4,637 tensors, 1,596,720,640 bytes | About 1.487 GiB before preparation overhead |
| 1,536 packed expert weights | NVFP4 W4A16; E4M3 block scales and FP32 global scales |
| Dense projections, shared expert and norms | BF16 checkpoint tensors |
| Embedding and vocabulary head | Shared target weights; no separate stored MTP payload |
| Stored layer name | `mtp.layers.0`; the `mtp.layers.48` quantization alias is explicit metadata |

MTP expert preparation must preserve W4A16 arithmetic. The target expert loader
uses activation-quantized NVFP4 with CSF scales and cannot be applied unchanged.
The [checkpoint manifest](../ignition/preparation/qwen38.py) validates MTP
tensor geometry. Target loading excludes those tensors; the separate
[MTP loader](../ignition/recipes/qwen38/mtp_weights.py) prepares their weights.

Transformers `4cc2aa84301c9aa210b5513dab9fefae03981f6e` ignores `mtp.*`
and has no MTP module. The [MTP recipe](../ignition/recipes/qwen38/mtp.py) composes
b12x operators for feedback, attention, expert execution, and output mixing.
Feedback arithmetic is checked against `b12x.sequence.mtp_feedback.reference`;
state alignment and graph replay are exercised by the recipe tests.

For a consumed target token `x[i]`, let `h[i]` be the target's 10,240-wide
residual **before its final mixer**. One teacher-forced MTP row receives
`(h[i], embedding(x[i+1]), position[i])`. Token IDs shift forward; hidden states
and RoPE positions do not. Prepared `b12x.sequence.mtp_feedback` normalizes and
projects those inputs into four streams. The MTP decoder and mixer produce both
a sampling vector and a multi-stream residual for recursive drafting. Repeating
the target's collapsed 2,560-wide sampling vector across streams is incorrect.

Recursive MTP states are proposal states. Even when their tokens are accepted,
their cached projections may depend on approximate MTP hidden states. Reusable
draft history must be absorbed from the **actual target** hidden states and
accepted next-token IDs.

## Available operator contracts

| Responsibility | Existing library interface | Bound or invariant |
| --- | --- | --- |
| MTP feedback | `b12x.sequence.mtp_feedback` prepared plan/bind/run | Four streams; caller-owned output and scratch |
| Target recurrent verification | `b12x.sequence.gdn_decode`, or Ignition `PreparedRecoveredRecurrence` with `PreparedGDNCommit` | At most eight inputs; checkpoint columns or per-token correction records followed by accepted-prefix commit |
| Target sparse attention | Ordinary prepared `b12x.attention.qsa` | Exact accepted-count, raw-tag and compression-group semantics |
| Draft selector reuse | QSA `DraftSelectionState`, `run(reuse=DraftSelectionReuse(...))` | Proposal-only; reset on round or request identity changes |
| Convolution history | `b12x.sequence.short_conv` and PLE operators | Speculative history extent and accepted-token selection |
| Stochastic chain verification | `flashinfer.sampling.chain_speculative_sampling` | Draft probabilities `[B,K,V]`, target probabilities `[B,K+1,V]`; explicit device RNG |

GDN's ordinary checkpoint path supports Qwen's 16 key heads and 48 value heads.
Ignition's [GDN recovery component](../ignition/components/gdn_recovery.py) also
supports that geometry: verification retains correction vectors, normalized keys
and decays, and commit replays only the accepted prefix into the live recurrent
row. The Qwen recipe uses this path for wider rounds and updates the live state
directly for width one. b12x's KDA recovery mode requires equal head counts;
FlashInfer's separate ReplaySSM implementation is restricted to SM100/103.
Public QSA draft reuse omits selection work and must never be used in target verification.

Library kernels remain the compute implementation. Server-owned captured
operations pack metadata, select committed state and update request records.
Missing tree or compact-state semantics require a public library contract;
there is no private eager model fallback.

## GLM learned proposals and continuation state

The [GLM MTP layer](../ignition/recipes/glm53/mtp.py) loads
`model.language_model.layers.45`. It combines the shared token embedding with
the normalized target output through separate RMS norms and an 8192-to-4096
projection. The decoder uses indexed MLA and ordinary residuals, followed by
288 routed experts with top-eight routing, a shared expert, and a final norm.
It shares the target's vocabulary projection. Its experts use NVFP4 weights
with E4M3 block scales and BF16 activations; target experts use CSF storage.
The draft decoder has no mHC streams, KDA recurrence, or rotary positions.

GLM prepares TP2 or TP4 graphs and defaults to a maximum of five draft tokens.
With concurrency four, batches one, two, and four each capture draft
counts `{5,4,3,2,1,0}`.
The shared native controller selects among the available counts. All ranks
sample identical gathered distributions with identical device RNG state.
History-copy proposals are not part of the GLM recipe.

Target KDA layers retain one recurrent checkpoint per verification input.
Commit selects the accepted checkpoint and convolution suffix. Indexed
attention uses twelve-entry C4 tail rings so speculative writes preserve the
preceding accepted tail. Prompt absorption and post-verification absorption
write only the draft layer's latent KV and C4 state, using target feedback.
The draft cache contains `max(consumed_tokens - 1, 0)` pairs. Exact-prefix
snapshots therefore preserve the draft partial page at that endpoint as well
as target pages and recurrent state. Absorption, proposals, verification,
sampling, and commit execute inside prepared CUDA graphs.

## DeepSeek DSpark proposals and continuation state

The [DSpark preparation code](../ignition/recipes/deepseek41/dspark.py) composes
three draft layers with a noncausal block sized to the preparation's draft-token
maximum, five by default. This matches the five draft positions described in
the [DeepSeek-V4.1 report](https://arxiv.org/html/2609.19969v1#S2.SS4.SSS3)
and the checkpoint's `dspark_block_size`. Preparation accepts explicit draft
limits through seven; block length controls prepared query and scratch shapes,
not learned weight dimensions. Every graph offering drafts evaluates the complete block before
sampling and verifying its selected prefix. Their context comes
from target inputs to layers 37, 38, and 39: collapsed mHC states are concatenated,
projected, normalized, and written into each draft layer's 128-token window.
The block begins with the pending target token and fills the remaining positions
with the checkpoint's noise token. Sequential Markov logits condition each
proposal on the preceding sampled token. The target and draft share token
embeddings and the sharded vocabulary projection.

Verification with draft tokens stages sliding-window writes and pending compression
state. Width one uses ordinary decode attention and direct target-cache writes;
it performs no draft forward or staged-cache commit. Both paths absorb committed
target feedback into the draft windows so the scheduler can alternate widths.
With drafts, commit publishes only the accepted input prefix. The draft windows
contain `consumed_tokens` inputs and occupy
continuation rows; DSpark requires no separate draft-page pool. Exact-prefix
snapshots retain the draft windows and committed Engram token history together
with target state. Graphs publish Engram lookup IDs for candidate tokens before
target execution; Helios delivers the requested rows through the shared embedding
table interface.

DSpark uses the shared token-vector completion and adaptive verification contracts.
Ignition captures verification widths one through six by default for every batch capacity;
Flare can reduce draft count and concurrency when loading. Helios schedules and
replays the selected graphs without interpreting DSpark layers or proposals.
Every retained or packed proposal producer executes the full configured draft
block. Cohort and verification-budget selection use historical joint prefix survival
from the DSpark confidence head and measured proposal-plus-verification cost. After
proposal completion, fresh confidence drives causal prefix selection using
costs, service prices, and historical continuations frozen before submission.
Verification outcomes remain separate from these confidence forecasts.

## A speculative round is a state transaction

At its start, a request has `N` consumed tokens, one pending token `p`, an
accepted-prefix target state, and an MTP cache derived from committed target
history. A round offers `K` drafts after `p`.

1. **Draft:** complete the feedback pair using the saved target tail and `p`,
   then recursively propose up to `K` tokens. Copy candidates may replace or
   extend a proposal under the same verification budget. Proposal RNG and
   histories are separate from committed sampling state.
2. **Verify:** execute the target on `[p, d[0], ..., d[K-1]]`. Retain all
   `K+1` vocabulary distributions and relevant target residuals, not only the
   final row. Each row sees exactly its causal candidate prefix.
3. **Accept:** select the accepted draft prefix and the target correction or
   bonus token. With `a` accepted drafts, ordinary chain verification consumes
   `a+1` inputs and produces `a+1` newly generated tokens: `a` drafts and one
   pending target sample. The incoming `p` was generated previously.
4. **Commit:** clip at device-visible EOS or output budget; select the matching
   target state, advance committed histories and penalties once, repair draft
   history using actual target residuals, and write a bounded token-vector
   completion. Unaccepted rows remain private and unreachable.

Let `t` be the emitted prefix length after EOS and output-budget clipping,
`1 <= t <= a+1`. Advance consumed and generated counts by `t`, select the state
after verification input `t-1`, and retain emitted token `t-1` as pending.
For example, emitting only `d[0]` consumes `p`, not `d[0]`, even if later draft
rows were verified. A terminal token remains pending. Requests already at their
limit execute no state transition. Generated, consumed and visible token counts
are distinct. Device terminal flags suppress any later queued round for that
request. Resource retirement waits for every referencing graph and transfer.

### Target state and prefix publication

GDN verification reads the checkpoint column selected by the preceding accepted
count and writes one checkpoint per input row. Convolution and PLE buffers must
retain all candidate histories until acceptance. PLE hashing reads committed
history; a proposed token affects only the candidate prefix that contains it.
Inactive rows never update another request's state.

QSA may physically retain rejected KV and selector rows, but logical lengths,
raw tags and interval anchors must exclude them. Rejected writes cannot mutate
an immutable shared page. Reserve private suffix pages, including boundary
pages touched by a speculative window, before submission.

A nonempty prefix checkpoint at `P` consumed tokens retains target state through
`x[P-1]`, its final sampling vector and its pre-mixer residual `h[P-1]`. Reusable
MTP history contains pairs through `(h[P-2], x[P-1])`: **`P-1` draft-cache
entries**. The pair containing `x[P]` depends on the next continuation and
cannot belong to an immutable checkpoint of the shorter prefix.

This creates an alignment difference. At `P % 64 == 0`, target QSA has no open
four-token compression group, while the MTP cache has three entries in its open
group. Its checkpoint must retain raw selector keys, tags, positions and anchor
needed to complete that group. The target restore operation's blanket raw-ring
invalidation is not a valid MTP restore operation. Draft-only selection-reuse
anchors are reset separately.

A full prompt hit restores both model histories, samples `p` from the saved
target sampling vector, and completes the MTP feedback pair from the saved
pre-mixer tail. Request RNG and penalty state are initialized from the new
request. Branches share immutable target and draft pages and own private suffix
state. At a crossed 64-token boundary, publish only the state corresponding to
an actually accepted consumed prefix.

### Memory and compact recurrence

The target's FP32 recurrent matrices occupy **108 MiB per saved verification
position**. Four requests with seven additional positions require 2.953 GiB;
fifteen require 6.328 GiB, excluding histories, logits and weights. Recurrent
writes also cost 108 MiB per verified row: batch one at width eight writes
864 MiB, approximately 0.5 ms at 1.79 TB/s before other traffic. A
batch-four width-eight shape would write 3.375 GiB, approximately 2 ms.
Measured round timings must include these writes. Existing GDN
supports a maximum of seven drafts plus the guaranteed input per request.

One additional BF16 QSA layer adds 2,112 bytes per cached token, including
compressed selector keys: 528 MiB for one 262,144-token context. One target
feedback tail occupies 20 KiB. A complete 4,096-row target residual occupies
80 MiB and must survive until MTP absorption; it cannot alias scratch overwritten
by the draft model. Preparation reports exact persistent, graph-pool and scratch
costs before allocating KV pages, preserving configured headroom and snapshot
budgets.

The current compact recurrence retains one committed matrix plus per-token
correction vectors, normalized keys and decay records. `PreparedGDNCommit`
applies accepted updates before the following round or checkpoint publication.
The [recovery tests](../tests/recipes/test_qwen38_gdn_recovery_cuda.py) compare
this arithmetic with b12x checkpointed decode. Full checkpoint columns remain
the component reference; capacity and write-traffic savings still require
complete-round measurements.

## Graph catalog, control and scheduling

The chain catalog has request capacities `B ∈ {1,2,4}` and integer verification
widths `1 <= W <= K_max + 1`, where `K=W-1`.
`PreparationConfig.speculative_widths` lists the prepared widths; its largest entry is
`K_max + 1`, and its default gives five drafts for Qwen and GLM.
Capture includes the Cartesian product of prepared batch capacities and widths.
With concurrency four and five drafts, this is eighteen speculative graphs.
Flare reduces concurrency and draft depth by selecting the required subset
before loading GPU resources; adaptive or fixed verification is a serving policy.

Every package also captures ordinary prefill, decode, resume, and prefix-cache
operations without draft computation or state maintenance. A maximum draft count
of zero selects those operations and disables speculative scheduling. Width one
inside an enabled speculative catalog maintains draft state, allowing the next
round to use a wider verification graph. Graphs declare their serving mode through
`speculation_enabled`; shared operations leave the field absent.

Per-width cost measurements must include the total target rows `B × W`.
The small-row MoE band through eight rows covers widths through eight at
batch one, four at batch two, and two at batch four. Larger shapes can select
grouped expert kernels with different costs. The Cartesian catalog includes
those shapes; preparation does not enforce an eight-row cap. Measure each
captured shape and locate the kernel-band transition before choosing serving
defaults. Odd widths require the same validation as even widths.
Adaptive width-one rounds continue absorbing target feedback into the MTP cache so
requests can resume drafting when concurrency falls. The native adaptive
controller chooses only among that batch's captured counts.
`speculative_widths` may be any increasing set of widths that includes one, so
the prepared draft counts need not be contiguous.
`speculative_fixed_width=4` selects three speculative graphs at runtime: one for
each batch capacity, each evaluating three drafts plus the pending input. Capture
retains every configured width regardless of the default verification policy. A
fixed width must belong to `speculative_widths`. Fixed drafting has no aggregate row cap;
batch four with three drafts evaluates sixteen target rows. Context tails,
host stop boundaries, and grammar constraints limit each request's valid prefix
inside that same graph. Masked positions cannot write KV or commit request state.
Verification widths include the pending input and remain an internal graph-layout
detail. A zero-draft graph consumes the pending input and emits one target token;
MTP history still absorbs the committed feedback pair. All shapes are prepared,
tuned, warmed and captured before handoff.
The Qwen metadata contract declares at least:

```text
operation: prefill+mtp_absorb | resume+mtp_seed | speculative_round | maintenance
request_capacity: B
verification_width: W
sampling_mode: general
proposal_source: mtp | copy
history_drafts: boolean
state_layout_version, arithmetic_policy, checkpoint_identity
```

Prefill-plus-absorb uses the configured prefill buckets. A speculative round
captures draft, target verification, acceptance and commit in one dependency
chain. A larger width cannot be selected until its kernels, history extent,
page reservations and completion storage exist. Padding changes capacity, not
causal history. All MTP variants retain the same seven-draft history horizon:
eight GDN state columns and matching QSA raw-ring/convolution capacities. The
preceding round's accepted count can exceed the following round's width. A
width-eight to width-one transition must preserve that count and state layout;
it cannot reuse an ordinary one-column, non-speculative binding by truncating
metadata. Width one also retains MTP absorption, so its measured cost differs
from a request with MTP disabled. Same-shaped graph variants share the existing
`ScratchArena.views(*shapes_and_dtypes)` storage only where lifetimes permit;
candidate logits, feedback, accepted IDs and state surviving another operator
have persistent storage. Final bindings refresh after tuning and freeze before
capture.

| Control | Owner |
| --- | --- |
| Graph handles, weights, plans, capacities, scratch envelope | Python preparation |
| Admission, page reservations, request grouping, graph choice | Rust |
| Acceptance, committed counts, next positions, pending tokens | Captured device operations |
| Acceptance/cost statistics and bounded exploration policy | Rust, using completed-round records |
| Grammar advancement, stop sequences, output presentation | Rust with ordered token-vector completions |

Rust adapts among prepared widths from request-local prefix survival and measured
full-round costs. Combined graphs learn from emitted/offered/accepted records,
or optional verification probabilities supplied by prepared graphs. Split graphs
use proposal-confidence forecasts. A separate funded exploration budget can
probe beyond the ordinary choice without replacing its historical mean;
see [the scheduling policy](scheduling-runtime-design.md#funded-acceptance-learning).
The fixed-width catalog gives each active row the same maximum draft
count. Different logical counts under a shared row budget require matching
verification and acceptance contracts; the selected graph's real padded cost
is what the controller prices.
Masking later iterations of an unrolled draft graph does not automatically
eliminate their compute. The design requires neither runtime recapture nor
device-side graph construction.

Speculative execution permits one generation round in flight. Rust consumes its
completion before selecting the following width, copy continuation or grammar
mask. CUDA dependencies serialize model work and protect shared scratch. Rust
reserves the round's maximum possible page/output use before submission. ABI3
uses two transfer frames of four 24-field `int64` records; each record retains
the scalar ABI2 prefix and adds emitted/offered/accepted counts, copy-source
bits, candidate and draft-cache lengths, verification width, and eight padded
token IDs. A frame cannot be reused until its transfer finishes. Ordinary ABI2
decode retains its two-submission pipeline. A two-round speculative pipeline
requires a separate measurement and explicit configuration before deployment.

Copy-only grammar rounds clone tentative native grammar states and upload a
mask for each copied prefix, including the target bonus position. Rust advances
the committed grammar with only the accepted tokens. MTP proposals are generated
on the device; grammar-constrained MTP requests use width one until the runtime
can provide valid masks for those conditional prefixes. Host-evaluated custom
stops also use width one. Rust acknowledges each pending token before another
graph consumes it. Prefix publication cannot include unacknowledged continuation
tokens. One grammar mask reused across draft positions is invalid. Such rounds
remain single in-flight until the native grammar state and uploaded masks are ready.
Multi-token custom stops similarly require ordered matching across token and
round boundaries, retaining candidate checkpoints until the selected stopping
prefix is known.
The tool/schema union grammar described in the
[core design](core-design.md#integration-and-remaining-work) must use the same
candidate-mask mechanism. Grammar acceptance tests require at least one accepted
draft, valid output, and exact committed state. Token-for-token comparison with
one-token execution requires deterministic arithmetic and compatible sampling
semantics. Tentative device progress beyond a host-selected stop remains
private, cannot publish snapshots, and retires only after all referencing work
drains. Host output truncation alone does not commit the corresponding model
state.

## Sampling correctness

Stochastic verification is the primary serving path. Qwen's checkpoint defaults
are temperature 1.0, top-k 20, and top-p 0.95; explicit request values win.
Acceptance and round-cost measurements use those defaults. Temperature-zero
results are reported separately.

All chains use FlashInfer's rejection sampler: accept
with `min(1, p(token)/q(token))`; on rejection, sample the normalized positive
residual `max(p-q,0)`, or sample the target bonus distribution after full
acceptance. `q` must be the distribution that actually generated the proposal.
Temperature, penalties, grammar and top-k/top-p/min-p filtering must be reflected
at each conditional prefix. A copied token has a point-mass proposal law:
`q(copy_token)=1`, with zero mass elsewhere. Copy substitutions use that law
at every temperature, including when replacing a native MTP proposal.

Temperature must be positive. Requests use the checkpoint's sampling defaults
unless explicit values override them. Target distributions use the same
FlashInfer probability sampler and chain verifier across supported temperatures.

The chain sampler's counters accumulate in place. Reset them per round; use
**emitted draft count plus one**, not independent acceptance count, for output
length. Store returned token IDs in persistent completion storage before graph
pool reuse. No PyTorch generator-backed random operations are captured for raw
Rust replay.

The public chain API takes a fixed `K` and has no per-request valid-length
argument. Stochastic requests therefore use the captured width, or a declared
width-pattern graph with separate fixed-length one-row invocations. Arbitrary
per-round heterogeneous lengths require a qualified public length-aware sampler.
Masking target rows or inventing padded proposal probabilities is insufficient.

The existing sampler uses one-row FlashInfer calls so Philox subsequence zero
is independent of packed lane position. Preserve this property for speculative
requests: use per-request one-row chain invocations with explicit device seeds
and offsets, unless a library API supplies an equally qualified request-keyed
stream. Draft randomness and target acceptance/correction randomness have
separate declared domains. Their offset-consumption law must be verified from
the selected library kernel. The inspected chain kernel consumes `K+1` scalar
Philox draws when all drafts are accepted and `K+2` after a rejection: its
independent-acceptance statistic samples the rejection position again. Reserve
nonoverlapping ranges by converting that bound to the RNG API's offset units,
and qualify the compiled kernel's progression. Merely adding emitted count
to the ordinary single-token counter would reuse random draws.

Distribution-preserving verification does not guarantee the same token sequence
as ordinary sampling for a fixed seed. Acceptance and correction consume a
different random stream. Verification widths can also change arithmetic near
logit ties; comparisons use measured operator bounds and repeatability.

## Adaptive policy and operator extensions

The controller selects bounded captured widths from completed-round token counts
and measured round cost. Copied request continuations use the same verification
budget as MTP proposals. Compact recurrence is a separate state-representation
extension; token trees additionally require parent-indexed GDN, convolution,
PLE and QSA histories.

## Implementation dependencies and acceptance

1. **Baseline and reference.** Record exact-checkpoint non-MTP serving at
   prefill capacities 512 and 4,096: cold TTFT, 256-token decode at concurrency
   1/2/4, 4K/64K prefill, 4,096-prefix-plus-11-tail TTFT, startup, and measured
   host submission gaps. Retain actual token IDs, EOS behavior and workload
   identities. CUDA completion events with timing disabled do not provide a GPU
   timeline; use explicit timing instrumentation or correlated profiler traces.
2. **Copy transaction and ABI.** Implement bounded vector completions,
   verification-row logits, state selection and persistent outputs using public
   operators and host-supplied copied continuations. Qualify every accepted
   count, prefix publication, and cloned per-candidate grammar states without
   loading MTP weights. Stochastic copy verification uses a point-mass proposal
   distribution at each copied token.
3. **MTP weights and arithmetic.** Load the exact W4A16 experts and BF16 tensors;
   qualify feedback, target-hidden alignment, draft attention, routed/shared
   experts and logits against component operator references. Match activation
   policy, KV dtype and draft-head policy explicitly. Add target-derived MTP absorption to prefill,
   accepted rounds and prefix checkpoints. Keep canonical eager comparisons
   diagnostic.
4. **Adaptive budget.** Measure fixed depths including zero first, with the
   checkpoint's stochastic defaults as the primary workload. Temperature-zero
   fixtures qualify probability semantics separately. Compare the controller
   against the same inputs and completed-token
   counts; record offered/accepted counts, draft/verify/commit timings,
   throughput, tail latency and memory. Preserve regressions and rejected cases.
5. **Compact commit and trees.** Require component state equivalence against
   full snapshots before replacing that representation. Add tree ancestry only
   with a corresponding target reference and measured benefit. Parallel-drafter
   training remains a separate model-development decision.

Correctness includes all-accept/all-reject and every intermediate prefix; QSA
groups of four and page boundaries of 64; all graph buckets and padded lanes;
width-eight to width-one transitions with eight preceding accepted inputs;
cold/prefix-hit/branch continuation; image-conditioned prompts; PLE history;
EOS, stop and budget clipping; penalties and RNG progression; slot generations;
cancellation and the single-round submission barrier; grammar barriers; and snapshot/page
pressure. Prompt counts are initialized once, generated counts advance once,
and rejected proposals never enter committed state or public output.

Compare Python and Rust replay of the same graphs and same-shape restores.
Metadata and state transfers require exact equality. Deterministic plans require
identical outputs; plans with variable reduction order use measured component
repeatability bounds. Cross-capacity and ordinary-versus-verification arithmetic
use tolerances established from measured same-input variation. Teacher-forced diagnostics retain full-vocabulary logits,
relative L2 with the reference norm as denominator, top-10 overlap and FP32
`KL(reference || actual)`. Numerical acceptance requires component comparisons
on identical operands, tolerances derived from measured arithmetic differences,
and measured repeatability for kernels whose reduction order can vary.
Generation tests exercise the common probability sampler with explicit settings
and fixed output budgets. Whole-model logit relative L2 is diagnostic. A higher
speculative acceptance rate cannot close an unexplained component discrepancy.

Every admitted graph must replay without allocation, compilation, recapture or
Python callbacks. Poison shared scratch between calls; alternate graph families
and capacities; verify no escaping output aliases volatile storage; drain both
execution and transfer streams before releasing owners. Startup fails if weights,
verification state, graph pool and reserved pages cannot fit.
Qualify each supported GPU architecture independently; SM120 results do not
establish SM121 execution or performance.

### Workload and paired performance measurements

The [serving performance corpus](../qualify/performance.py) uses exact-length
numbered-list requests padded with repeated `apple` tokens. It measures serving
shapes, cache paths and replay overhead; it does not establish representative
MTP acceptance or speedup.

Freeze a separate MTP workload before measurement: agentic multi-turn exchanges
with recorded tool results, tool-call and JSON output, code edits with substantial
copiable context, non-copying reasoning and general chat, and long-context
continuations. Include both repeated prefixes with different tails and repeated
identical tails. Record source identities, exact messages/tools, rendered token
IDs, turn boundaries, tool-result fixtures, sampling settings and expected output
semantics. Preserve every category, including cases with little copy opportunity;
do not execute external tool side effects during the benchmark.

Pair no-MTP execution, fixed MTP depths, adaptive MTP with copying disabled, and
the same adaptive policy with copying enabled. Keep request contexts, concurrency,
prefix-cache treatment, EOS/custom-stop handling, output budgets and grammar
constraints identical across arms. Unsupported speculative grammar or custom-stop
cases exercise the declared one-token path and remain in the workload. Repeated
runs alternate arm order and retain individual observations and failed requests.

Report actual completed-output token counts, termination reasons, end-to-end
latency, TTFT, throughput and latency tails. Separate offered, verified and
emitted draft counts; condition acceptance on draft depth, proposal source and
workload category, and record copy eligibility. Compare draft, verification and
commit costs with the graph buckets actually executed. A longer output, early
EOS, repeated-text artifact or truncated response cannot be counted as equivalent
completed work. Synthetic serving throughput does not predict the result of
these workload measurements.

Performance compares ordinary decoding, minimal replay of the prepared
speculative graph sequence, and the complete native service with matched settings.
The native overhead target remains within 5% of that **measured replay floor**, separately
from frontend delivery. The approximate 3.9 ms single-token weight-traffic bound
is a non-speculative batch-one reference: amortized verification can reduce bytes
per emitted token, while extra expert routes and MTP work can increase round
cost. No fixed MTP speedup is assumed. Keep raw measurements and the configuration needed to interpret them; large
tensor and trace files remain outside the repository.
