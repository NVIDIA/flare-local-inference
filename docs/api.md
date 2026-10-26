# Native inference APIs

Helios serves the loaded runtime package through four generation protocols.
Every protocol uses the same tokenizer, prepared chat template, image processor,
sampling defaults, scheduler, prefix cache, and CUDA graphs. Clients execute tools
and return their results in subsequent requests.

| Route | Input | Output |
| --- | --- | --- |
| `POST /v1/chat/completions` | Chat messages | Assistant message or chat chunks |
| `POST /v1/completions` | Raw text or token IDs, optionally batched | Text choices or completion chunks |
| `POST /v1/messages` | Anthropic messages and system blocks | Message content blocks or named SSE events |
| `POST /v1/messages/count_tokens` | Anthropic message request without generation | Exact prepared input token count |
| `POST /v1/responses` | Responses input items and instructions | Ordered output items or named SSE events |

`GET /v1/models` returns one card for the loaded model. Its `id` is the first
nonempty configured alias, or the checkpoint identifier when no alias is set.
`GET /v1/models/{model}` returns that same card for the checkpoint identifier or
any configured alias; repository IDs containing slashes are supported. Unknown
names return HTTP 404 with an OpenAI-style `model_not_found` error. Both routes
use the configured bearer authentication.

Model cards include the vLLM-compatible fields `id`, `object`, `created`,
`owned_by`, `root`, `parent`, `max_model_len`, and `permission`. `max_model_len`
is the effective serving context limit in tokens, including prompt and output.
`root` is the loaded model's identifier, `parent` is null, and `owned_by` remains
`local`. `created` is the serving instance's discovery initialization time in
Unix seconds; cards and permission IDs stay stable until the server restarts.
Permissions advertise sampling and viewing. `allow_logprobs` reflects the loaded
package’s generation logprob outputs; fine-tuning, engine creation, and search
indices remain disabled.

The additional `flare` object reports prepared and selected serving metadata:

| Field | Meaning |
| --- | --- |
| `vocab_size` | Prepared output vocabulary size |
| `max_top_logprobs` | Packaged alternative-token capacity; 20 for new packages, zero for older packages |
| `max_concurrency` | Selected maximum active requests |
| `prefill_chunk_size` | Maximum tokens in a prefill chunk, not the context limit |
| `default_max_tokens` | Default output-token budget when a request omits it; remaining context still bounds output |
| `sampling_defaults` | Packaged seed, temperature, filters, and penalties used for omitted request fields |
| `sampling_ranges` | Inclusive `{min, max}` bounds for distribution parameters; null fixes a parameter at its packaged default; seed remains request-controlled |
| `max_images_per_prompt` | Context-derived upper bound on image count; image sizes, framing, text, output, and HTTP body size further constrain admission; zero for text-only packages |
| `max_image_tokens_per_prompt` | Context-derived upper bound on image tokens across the entire conversation; text, framing, and output also occupy context; zero for text-only packages |
| `image_feature_window_tokens` | Prepared feature storage per active image request; chunked prefill reuses this storage across windows, so it does not limit total conversation image tokens |
| `max_image_tokens_per_image` | Image tokens one image may cost after the profile's normalization; the smaller of one prepared vision graph and one feature window; zero for text-only packages |
| `max_image_pixels` | Largest decoded image extent (width times height) accepted before normalization; zero for text-only packages |
| `max_image_bytes` | Largest encoded image accepted; zero for text-only packages |
| `input_modalities` | `text`, plus `image` when vision is prepared |
| `output_modalities` | `text` |

Generation requests always use the loaded model, regardless of the request's
`model` string. Returned generation objects identify the loaded model.

## Ollama model discovery

`GET /api/tags` lists the loaded model using Ollama's discovery response format.
Its `name` and `model` match the identifier in `/v1/models`. `POST /api/show`
accepts `{"model": "NAME"}` for that identifier, the checkpoint identifier, or any
configured alias. An optional boolean `verbose` is accepted; both modes return
the same metadata. Missing, empty, or incorrectly typed names return HTTP 400;
unknown names return HTTP 404. Discovery errors use `{"error": "message"}`.
Both routes require the configured bearer key, when enabled. `/api/show` accepts
uncompressed JSON within the HTTP body-size limit. Authentication errors retain
the OpenAI error format.

`/api/show` reports `completion`, plus `vision` when vision graphs are prepared,
`tools` when a tool parser is declared, and `thinking` when a reasoning parser is
declared. `model_info` contains `general.name`, `flare.context_length` (the
effective prompt-plus-output limit), and `flare.vocab_size`. These values come
from the loaded contract and serving limits, without model-name inference.

`details.format` is `flare`. Unavailable family, parameter-count, and quantization
labels are empty strings; `families` is empty. The listing reports unknown artifact
size as zero and unknown modification time as `0001-01-01T00:00:00Z`. Its `digest`
is a SHA-256 cache key over the prepared model identity and discovery details;
it changes when the model or advertised metadata changes. It is not an Ollama
manifest digest. Templates and tensor inventories are not exposed.

OpenCode v2 can discover Flare through its built-in Ollama provider. Merge into
`~/.config/opencode/opencode.json`:

```json
{
  "providers": {
    "ollama": {
      "settings": {"baseURL": "http://127.0.0.1:8000/v1"}
    }
  }
}
```

Add `settings.apiKey` when authentication is enabled. OpenCode polls the discovery
endpoints and uses `/v1/chat/completions` for generation; choose a discovered model
with `/models`. Its Ollama integration reads context, tool, and vision metadata;
reasoning controls may still require client configuration. Flare implements
Ollama discovery only: `/api/chat`, `/api/generate`, and model-management routes
are not supported.

## Generation log probabilities

Logprobs describe the raw next-token distribution over the complete vocabulary:
`logit(token) - logsumexp(logits)`, with FP32 accumulation, before temperature,
penalties, grammar masks, and top-k, top-p, or min-p filtering. Speculative output
uses the target model's distribution at each committed position. The chosen token
is always included, even when outside the alternatives. Natural alternatives are
ordered by decreasing logprob, with increasing token ID for ties.

Every newly exported package provides up to 20 natural alternatives and supports
explicit token IDs. Older packages retain their natural request limit (1..20),
or advertise `allow_logprobs: false` and `flare.max_top_logprobs: 0` when they have
no generation logprob outputs. Unsupported requests return HTTP 400.

For Chat Completions, set `logprobs: true` and optionally `top_logprobs: 0..20`.
A positive `top_logprobs` requires `logprobs: true`; zero alone is accepted and
requests no logprobs. Buffered responses and streaming choices carry:

```json
{"logprobs":{"content":[{"token":"OK","logprob":-0.25,"bytes":[79,75],
  "top_logprobs":[{"token":"OK","logprob":-0.25,"bytes":[79,75]}]}]}}
```

There is one entry per returned generated token, including reasoning, tool-call,
and control tokens. Stream chunks may carry entries with an empty text delta,
particularly when the tokenizer buffers incomplete UTF-8. Concatenate entries in
stream order. A token's text can span multiple deltas, including a final parser
flush; each text delta does not require a new logprob entry.
A terminating EOS or token-ID stop has no entry. An EOS under
`ignore_eos` keeps its entry. A token partially clipped by a string stop keeps its
entry and full bytes; tokens entirely beyond the stop have no entries. `bytes`
contains the token's exact bytes, including partial UTF-8; `token` uses UTF-8
replacement characters where needed for JSON.

Legacy Completions accepts integer `logprobs: 0..20`. Each choice's `logprobs` has
parallel `tokens`, `token_logprobs`, `top_logprobs`, and `text_offset` arrays.
`top_logprobs` contains one token-to-logprob object per generated token.
`text_offset` counts Unicode characters from the start of the completion, including
positions inside a decoder span released for several tokens. `echo: true` also
returns prompt entries when the package was exported with
`--prompt-logprobs`; see below.

Responses accepts `include: ["message.output_text.logprobs"]` with optional
`top_logprobs: 0..20`. Entries use the Chat token shape and appear in each
`output_text` part's `logprobs`, and in `response.output_text.delta` events.
Only tokens whose first text byte belongs to output text receive entries there;
reasoning, tool calls, and delimiters do not. A token spanning a delimiter and
answer belongs to the delimiter, even if its suffix contributes answer text.
A reasoning delimiter spelled out in ordinary answer text is output text and
receives entries.
Other `include` values retain their existing behavior.

On these three endpoints, `return_tokens_as_token_ids: true` changes chosen and
alternative token strings to `token_id:<id>`; bytes remain exact. `/v1/messages`
explicitly rejects logprob fields. Temperature zero remains rejected. Single-token
continuations work with `max_tokens: 1`, `continue_final_message: true`,
`add_generation_prompt: false`, `logprobs: true`, and `top_logprobs: 20`, omitting
temperature to use the model's prepared default.

On Chat, Completions, and Responses, `logprob_token_ids: [id, ...]` replaces
natural generation alternatives with those IDs in caller order. Existing generation
logprob enablement is required. Supply at most 128 elements before deduplication;
IDs must be nonnegative integers within the model vocabulary. Duplicates are
removed in first-occurrence order. Null and empty arrays behave as omission.
The sampled token is appended only when absent, so explicit results contain at
most 129 entries. Values retain full-vocabulary normalization. Numeric top-k
options still accept only 0..20 and do not truncate explicit results.

Prompt scores and echoed prompt entries remain natural. Completions accepts
validated IDs with `echo: true, max_tokens: 0`, which returns no generation
positions. Completions maps retain their existing decoded-string collision
behavior; use `return_tokens_as_token_ids: true` to distinguish IDs. Messages
rejects `logprob_token_ids`. Older packages reject nonempty selections before
admission with an explanation to re-export.

### Prompt log probabilities

Export with `ignition export … --prompt-logprobs` to capture scored prefill graphs.
The flag defaults off. Packages without those graphs return HTTP 400 for prompt
logprob requests and continue serving other requests unchanged.

Completions and Chat Completions accept integer `prompt_logprobs: 0..20`, independently
of generation logprob options. Streaming with this field is rejected, including
zero. Completions place the result at `choices[i].prompt_logprobs`; chat places it
at the response's top-level `prompt_logprobs`. The result is a list aligned with
prompt token positions (the rendered prompt for chat). Its first entry and image
positions are `null`. Each other entry maps token IDs to the actual token and up
to the requested number of alternatives. `decoded_token` remains decoded text
when `return_tokens_as_token_ids` changes the legacy token strings:

```json
[null, {"42":{"logprob":-0.25,"rank":1,"decoded_token":" answer"},
        "17":{"logprob":-2.5,"rank":2,"decoded_token":" question"}}]
```

Values describe the raw distribution before sampling transforms, as generation
logprobs do. Every rank is one plus the number of vocabulary entries with a
strictly greater reported logprob. Equal values share a rank. Alternatives use
decreasing logprob and increasing token ID to break ties. Non-finite logprobs are represented
as `-9999.0`. The actual token is always included, even outside the alternatives.

For Completions, `echo: true` with integer `logprobs: 0..20` requests prompt scoring
and prepends prompt entries to the legacy `logprobs` arrays. Position zero and
image positions have null `token_logprobs` and `top_logprobs`. Prompt `text_offset`
counts characters in the preceding decoded token strings; generated offsets start
after the echoed text. Streaming echo emits one logprob frame per completed
prompt chunk, before generated entries. Buffered completions also include
`choices[i].prompt_logprobs`. An explicit `prompt_logprobs` value overrides the
inferred count for that field; `logprobs` controls the legacy arrays.

For Completions, `max_tokens: 0` with prompt scoring submits the prompt and
returns zero generated tokens with `finish_reason: "length"`. Chat Completions
requires a positive output limit, including for prompt scores, as vLLM does.
Prompt and generation logprobs can share a request. Scoring starts from an uncached
prompt and may publish prefix checkpoints for other requests. Responses and Messages reject
`prompt_logprobs`.

## Transport and admission

Generation uses HTTP JSON or `stream: true` with SSE. WebSockets, background
execution, and hosted tools are unsupported. Anthropic requests require
`anthropic-version: 2023-06-01`.

When a serving key is configured, OpenAI routes use `Authorization: Bearer KEY`.
Anthropic routes also accept `x-api-key: KEY`; supplying both headers requires
both to match. `/health` and `/metrics` are unauthenticated. Flare reads its
serving key from `--api-key-file`, independently of client configuration.

Generation request bodies support `identity`, `gzip`, zlib `deflate`, `br`, and `zstd` content
encodings. Both encoded and decoded bodies are bounded by `max_body_bytes`
(32 MiB by default). Decompression and frontend processing share the bounded
frontend worker pool. Invalid requests receive protocol-specific JSON errors;
failures after streaming begins produce protocol-specific terminal error events.

Streaming requests return HTTP 200 and SSE headers after frontend validation and
bounded enqueue, without waiting for GPU admission or prefill. Admission failures
after enqueue are terminal SSE errors under HTTP 200; buffered requests retain
the corresponding HTTP error status. Chat Completions emits its initial
`{"role":"assistant","content":""}` delta only after the first generated token
is available, including tokens hidden by stop matching or text decoding. Chunked
prefill and cache restoration produce no chat data chunks before that token.
A failure before the first token produces an error without a role chunk or
`[DONE]`. SSE keepalive comments may precede generation during waits longer than
15 seconds.

Each completion batch reserves one admission slot per generated choice. Every
child is prepared and validated before the group is submitted to the native
worker. An oversized group is rejected; temporary admission exhaustion returns
429. Disconnecting a stream cancels all unfinished children, and native work
drains before its resources are reclaimed. Output queues remain bounded.

## Health and metrics

`GET /health` returns `{"status":"ok"}` while the model runtime is healthy and
HTTP 503 with an error object otherwise. Neither this route nor `/metrics`
requires the serving key: both report aggregate state without request content.

`GET /metrics` returns Prometheus text exposition format 0.0.4 by default.
Counter and gauge names start with `flare_`, such as
`flare_generated_tokens_total`, `flare_active_requests`, and
`flare_free_kv_pages`; there are no histograms. Beyond the JSON fields below, the
text includes speculative and proposal counters by batch, width, or depth, graph
launches by kind, embedding row-cache counters, HTTP admission gauges, and
uptime. `flare_history_drafts_enabled` reports whether shared decoded history
augments learned proposals in the serving process.

The route returns one JSON object instead when the query contains `format=json`,
or when the `Accept` header gives `application/json` a higher quality than
`text/plain`; at equal quality the earlier media range wins. `format=prometheus`
selects text regardless of `Accept`, and any other `format` value returns HTTP
400. Requests without an `Accept` header, and Prometheus scrapers, receive text.
`flare_helios::server::Snapshot` deserializes the object, reading absent fields
as zero, and `flare_helios::server::view` holds the rate history and plots of
the serving terminal dashboard.

JSON fields are integers except `learned_drafter`. Totals count from server
start; `active`, `pending`, the page fields, and the host record fields are
instantaneous gauges.

| Field | Meaning | Prometheus metric |
| --- | --- | --- |
| `prefill_tokens` | Prompt tokens consumed by completed prefill, excluding padding and prefix reuse | `flare_prefill_tokens_total` |
| `generated_tokens` | Committed generated tokens, including EOS and tokens withheld by text stops | `flare_generated_tokens_total` |
| `decode_weight_bytes` | Bytes of weights one decode step reads for one request, as the package's recipe declares for its checkpoint; zero when undeclared. Times verification steps per second it is the weight bandwidth the model consumes, shown as WEIGHTS on the dashboards | `flare_decode_weight_bytes` |
| `draft_steps` | Draft positions run by the learned drafter; zero without one | none |
| `verify_steps` | Completed speculative or verification graphs | sum of `flare_speculative_rounds_total` |
| `verification_drafts` | Draft positions allocated for progressing request rounds, including rejected drafts and excluding padding | `flare_verification_draft_tokens_total` |
| `verification_rounds` | Progressing decode or verification request rounds, including zero-draft rounds | `flare_verification_request_rounds_total` |
| `accepted_drafts` | Accepted draft tokens, excluding the bonus token | `flare_accepted_draft_tokens_total` |
| `draft_rounds` | Request rounds that offered at least one draft | `flare_draft_request_rounds_total` |
| `active` | Model slots with a live request | `flare_active_requests` |
| `pending` | Requests waiting in the model scheduler | `flare_pending_requests` |
| `completed` | Requests whose generation completed successfully | `flare_requests_completed_total` |
| `failed` | Requests failed by admission, execution, grammar, or output backpressure | `flare_requests_failed_total` |
| `replay_gpu_ns` | Sum of maximum rank-local CUDA durations of completed graphs, including readiness waits | `flare_replay_gpu_nanoseconds_total` |
| `replay_readiness_wait_ns` | Exposed embedding readiness delay measured across ranks, for both RAM and disk delivery | `flare_replay_readiness_wait_nanoseconds_total` |
| `replay_readiness_samples` | Completed graphs with prepared readiness timing outputs; zero for packages without them | `flare_replay_readiness_samples_total` |
| `reused_tokens` | Tokens covered by GPU prefix-cache hits | `flare_cache_reused_tokens_total` |
| `cached_pages` | KV pages owned by immutable checkpoints | `flare_cached_kv_pages` |
| `free_pages` | KV pages reusable after completed GPU fences | `flare_free_kv_pages` |
| `total_pages` | Pages in the per-rank KV pool | `flare_total_kv_pages` |
| `page_tokens` | Logical history tokens per KV page | `flare_kv_page_tokens` |
| `host_hits` | Host prefix-store lookups whose chain admission fetched | `flare_host_cache_hits_total` |
| `host_reused_tokens` | Tokens covered by host prefix-store chains beyond the GPU checkpoint | `flare_host_cache_reused_tokens_total` |
| `host_memory_records` | Checkpoint records in the host store's registered memory | `flare_host_cache_memory_records` |
| `host_disk_records` | Checkpoint records in the host store's record directory | `flare_host_cache_disk_records` |
| `host_bytes_out` | Bytes copied from GPU checkpoints to the host store | `flare_host_cache_bytes_out_total` |
| `host_bytes_in` | Bytes copied from the host store into GPU pages and rows | `flare_host_cache_bytes_in_total` |
| `learned_drafter` | Boolean: the model has a learned drafter | none |

Dashboard DEPTH is the change in `verification_drafts` divided by the change in
`verification_rounds` over the sampling window. It averages each request's planned
draft-prefix length, including zero-draft rounds, rather than graph width or batch
capacity. For packed speculation, completion records report the prefixes allocated
on the GPU before target verification. Prefill, resume sampling, inactive padding,
failed lanes, and terminal rounds without progress do not enter this average.

## Conversation and tool semantics

The neutral request representation in `helios/src/server/inference.rs` separates
conversation messages from raw prompts. Conversation parts represent text,
images, reasoning, tool calls, and tool results. Protocol decoders populate these
types directly. `server/request.rs` lowers them into the packaged chat template
and model frontend; no endpoint translates through another endpoint's JSON.
When developer instructions are present, system and developer instructions become
one leading system message. Instruction sections retain their role labels and
request order; ordinary conversation messages retain their relative order.
Conversations without developer messages use the template's native system-role
rules where supported. The shared frontend probes the prepared template once for
inline system-message support. If the template rejects or omits inline system
text, noninitial system messages become user messages at the same positions,
with their text wrapped in `<system>...</system>`. Text blocks are joined with
blank lines. The leading system message and preceding conversation remain
unchanged. This conversion supplies a textual role cue; it does not provide a
native system role to models whose templates cannot represent one inline.

Model output parsers emit neutral semantic events. The ordered accumulator in
`server/output.rs` retains text, reasoning, and tool blocks. Protocol encoders use
the same accumulated output for buffered responses and SSE. Tool arguments are
JSON objects, bounded to 8 MiB. Existing model parsers buffer tool envelopes until
they are complete, so argument events do not imply incremental generation of an
unfinished call. Truncated tool envelopes are omitted and the response reports
its output limit; recognized but incomplete calls at natural completion are errors.

DeepSeek reasoning and both DSML tool syntaxes use the pinned official
`deepseek-recipe` state machine. Helios adapts its actions into the same neutral
events, validates nonempty tool names and JSON objects with nonempty, unique
argument names, and publishes each call when its arguments complete. Closing the
outer calls block ends parsed output; subsequent generated text is discarded.
The official parser owns markup recognition, including its tolerance of omitted
or partial tags. Structured answers become verbatim when ordinary answer content
begins, preserving scalar JSON, Lark output, and markup inside string values.
Shared runtime handling owns token accounting, stop filtering, and termination.

A reasoning delimiter that the tokenizer encodes as one token, such as
`<think>`, `</think>`, `</mm:think>`, or Gemma 4's `<channel|>`, opens or
closes reasoning only when the model generates that token. When a model writes
the same text in ordinary pieces, for example while explaining a chat template,
the text remains reasoning or answer text. Delimiters that span several tokens,
such as Gemma 4's `<|channel>thought`, are matched as text, as are DeepSeek's
delimiters, which the official parser recognizes. Structured-output grammars
follow the same rule: they end reasoning only at the delimiter token. A
structured answer may contain the delimiter token, which it returns as text. If
a model ever writes its real closing delimiter in ordinary pieces, the rest of
its output remains reasoning. Under a structured-output grammar, reasoning then
cannot end, so the request runs to its token limit without an answer.

Qwen treats `<tool_call>` as an implicit end to reasoning. Function-tag calls
complete at `</function>` and accept omitted outer tool tags in answer content.
A `<tool_call>` or `<function=` tag starts a call only when followed by a
`<function=NAME>` header and a parameter or `</function>` tag; other mentions,
such as a quoted tag in prose, remain reasoning or answer text. A complete
call written in prose, inline or on its own lines, still parses as a call and
ends reasoning. A tag still undecided when generation ends is parsed as an
incomplete call. GLM's `<tool_call>` likewise starts a call only when followed
by a function name and an `<arg_key>` or `</tool_call>` tag.
Gemma 4 and MiniMax keep their current behavior: Gemma's `<|tool_call>` opener
and MiniMax's namespace-prefixed opener do not take part in this check. DeepSeek
DSML and XTML are parsed by their own parsers rather than this path.
A missing `</parameter>` is closed by the next parameter or function boundary.
Parameter schemas guide
best-effort type conversion; non-strict schemas do not reject invalid values.
Qwen removes one leading and one trailing wrapping newline from parameter values;
MiMo's compact tags preserve their values verbatim, including indentation and
newlines.

Supported tools are client-executed JSON functions with `tool_choice` set to
`"auto"`, `"none"`, `"required"`, or a named tool. Anthropic's `{"type":"any"}`
selects required mode. Named selection uses
`{"type":"function","function":{"name":"lookup"}}` in Chat Completions,
`{"type":"tool","name":"lookup"}` in Messages, and
`{"type":"function","name":"lookup"}` in Responses. Responses custom tools use
`{"type":"custom","name":"lookup"}`; namespaced tools use `namespace__name`.
The selected tool must be declared and its type must match.

Named selection presents only the selected tool definition to the model, retaining
conversation history and adding no instructions or decoding constraints.
Buffered and streaming responses omit ordinary assistant answer text while
preserving reasoning and tool arguments. Usage includes all generated tokens,
including omitted answer text. Every emitted call must target the selected tool,
and at least one completed call is required. A call to another tool fails before
that call is emitted. Missing and incomplete calls fail as in required mode.

With tools enabled and `tool_choice: "auto"`, parseable calls to
undeclared tools are returned with their generated names and arguments in both
buffered and streaming responses across all three chat APIs. Clients can repair
or reject these calls and provide feedback in a subsequent turn. Declared schemas
guide argument conversion where the tool syntax requires it; unknown tools have
no schema for that conversion. Malformed envelopes, empty names, and invalid
argument syntax still fail parsing. Required mode needs at least one declared
tool, adds an instruction to call a tool to the prompt, and validates that
generation produced at least one
completed call to a declared tool. Sampling is unchanged; no calls are fabricated
and no retries are attempted. Missing calls, including output truncated before a
complete call, produce HTTP 500 for buffered requests or a terminal error event
for streams. A required-mode stream may deliver partial text before that error, but does not
emit a success terminator. Responses also supports text custom tools by presenting a
JSON `{ "input": string }` function to the model and restoring the custom call on
output. Anthropic's `type: "custom"` uses its ordinary `input_schema` contract.
Responses namespaces become unique `namespace__name` names in the model template
and regain their namespace on output; ambiguous names are rejected.

Responses custom tools may specify Lark or regex grammars. Helios compiles them
before admission, includes their contract in the model's tool description, and
validates completed input against the exact bytes before emitting a completed
call. Invalid output fails the response; generation is not constrained to the
grammar and no automatic repair is attempted. Grammar compilation uses the same
bounded llguidance compiler as structured output.

`parallel_tool_calls: false` permits at most one tool call per response.
Anthropic's `tool_choice.disable_parallel_tool_use: true` applies the same limit.
The output validator fails the response when a second call begins, before emitting
that call. Buffered requests receive HTTP 500; streams receive a terminal error
and may already have delivered the first call. Combined with required mode, a
successful response contains exactly one call. Generation uses ordinary sampling.

Parallel tool results with unique IDs matching the preceding assistant's calls
are normalized to call order before chat-template rendering. Each result's text
and images move together; intervening non-tool messages keep their positions.
Missing, duplicate, or unknown IDs leave that group in its supplied order.
Results never move across another
assistant message.

Named tool selection and strict function schemas are rejected. Hosted search,
computer execution, file stores, audio, and remote image downloads are unsupported. Images must be
base64 PNG, JPEG, or static WebP data. Images are encoded sequentially into
bounded feature windows. Chunked prefill consumes each window before its storage
is reused; the total image tokens across a conversation may exceed
`image_feature_window_tokens`. There is no separate four-image limit. Each image
must fit a prepared vision graph and one feature window. The whole request must
fit the HTTP body limit and the context shared by images, text, and generated
output. Animated images are unsupported.

Image costs are the token counts that usage reports in
`prompt_tokens_details.multimodal_tokens.image`: the profile normalizes each
image to its own grid, then merges a fixed number of encoder patches into one
token. Admission costs every image before rejecting, so a budget rejection
(HTTP 400) states the whole request's image tokens against
`max_image_tokens_per_prompt`, with the patch counts in parentheses, and names
the first image that crossed the budget with its position, pixel size, and
tokens. Oversized single images are rejected against
`max_image_tokens_per_image` or `max_image_pixels` with the observed size.
Each profile applies its packaged resizing rules independently to each image.
Flare does not reduce image resolution as conversation history grows or to fit
a feature window. Chat templates add fixed framing tokens around each image that
count toward `prompt_tokens` but not toward the image budget or
`multimodal_tokens.image`.

Requests include the image data again when continuing a conversation. Prefix
identity includes the image content and its processing settings. When a reused
GPU or host checkpoint covers an image's entire input span, Flare skips its
preprocessing and vision graph. Appending an image preserves reuse of earlier
unchanged messages. A prefix ending inside an image requires that image's
complete vision computation.

Window transitions use a private snapshot of complete model state and restore
it under a fresh execution generation before reusing the feature arena. They
work with existing packages and with both host cache tiers disabled. These
internal transitions do not count as cached prompt tokens.

Configured host memory and disk retention also cache source descriptions and
CPU vision inputs. These records can avoid decoding and preprocessing even when
changed text prevents prefix reuse; the vision graph still runs for images
outside the reused prefix. No persistent GPU image embeddings are retained.
Cache eviction or corruption causes reconstruction from the supplied image
data. Reuse does not increase image, context, or HTTP body limits.

Anthropic histories accept inline system text and preserve text, thinking, tool
use/results, errors, and image blocks. Message-level control metadata, including
`output_config.effort`, is ignored; top-level generation controls apply to the
request. Empty system messages produce no prompt text. Assistant-prefill messages
use the packaged template's continuation support. `context_management` may
request preservation of all thinking blocks
(`clear_thinking_20251015` with `keep: "all"`); active context edits are rejected.
Cache-control hints do not alter the native prefix-cache policy.

## Stateless Responses

Responses requests contain the complete conversation. `store` defaults to false;
`store: true`, `previous_response_id`, `conversation`, background requests, and
automatic truncation are rejected. No response retrieval or deletion service is
provided. Clients append returned output items and tool-result items to their
next request.

Output items retain their order and have stable IDs throughout a response.
SSE includes response, output-item, content-part, text/reasoning, and tool-call
lifecycle events with increasing sequence numbers. Success ends in
`response.completed`; exhausted output budgets end in `response.incomplete` with
`max_output_tokens`; generation failures end in `response.failed`. Responses SSE
does not use an OpenAI Chat `[DONE]` sentinel.

Readable reasoning is returned as `reasoning_text` content with an empty summary.
Opaque encrypted-only histories are rejected. Helios does not fabricate hosted
provider reasoning signatures or encrypted continuation state.

## Reasoning, sampling, and usage

Prepared model defaults fill omitted sampling fields. Numeric request values for
`temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `presence_penalty`,
and `frequency_penalty` are clamped to the inclusive ranges declared in the
package. A parameter without a range is fixed at its packaged default, including
in packages exported without range metadata. Ranges must contain their defaults
and valid sampler values; temperature bounds are positive. For example,
`temperature: 0` becomes the declared minimum, or the default if no range exists.
Malformed types still return HTTP 400; `top_k` must be an integer. `seed` remains
request-controlled and uses its packaged default when omitted. Sampling defaults
and ranges come from the model author's recommendations declared in the recipe.

The recipes use these default-mode settings. Parameters without an interval in
the table are fixed at their defaults. Bounds span the minimum and maximum values
across each card's task or mode presets, including intermediate values and
independent combinations. Qwen presence-penalty bounds are an explicit interval
in both cards. Changing request reasoning mode does not change sampling defaults.

| Recipe | Temperature | Top-p | Top-k | Declared interval | Model-card basis |
| --- | --- | --- | --- | --- | --- |
| `qwen38` | 1 | 0.95 | 20 | Temperature [0.7, 1]; top-p [0.8, 0.95]; presence penalty [0, 2] | [Qwen3.8-Flash-Next thinking/instruct presets](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices) |
| `qwen38_dense` | 1 | 0.95 | 20 | Temperature [0.7, 1]; top-p [0.8, 0.95]; presence penalty [0, 2] | [Qwen3.8-27B thinking/instruct presets](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices) |
| `glm53` | 1 | 0.95 | Disabled | Temperature [0.95, 1]; top-p [0.95, 1] | [GLM-5.3-Flash evaluation presets](https://huggingface.co/zai-org/GLM-5.3-Flash#footnotes) |
| `deepseek4` | 1 | 0.95 | Disabled | None | [DeepSeek-V4-Flash-Vision-Exp agent evaluations](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp#introduction) |
| `deepseek41` | 1 | 1 | Disabled | None | [DeepSeek-V4.1-Flash-1017 inference recommendations](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017#minimal-inference) |
| `mimo_v2` | 1 | 0.95 | Disabled | None | [MiMo-V2.6-Pro-MOPD deployment](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD#4-deployment) |
| `kimi3` | 1 | 1 | Disabled | Top-p [0.95, 1] | [Kimi-K3 single-step/agentic evaluations](https://huggingface.co/moonshotai/Kimi-K3#3-evaluation-results) |
| `step5` | 1 | 0.95 | Disabled | None | [Step-5-Preview-NVFP4 inference recommendations](https://huggingface.co/stepfun-ai/Step-5-Preview-NVFP4#evaluation-notes) |
| `nemotron_h` | 1 | 0.95 | Disabled | None | [Nemotron-3.5-Bolt text recommendations](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB#model-summary) |

All recipes default to `min_p=0`, `repetition_penalty=1`, and zero presence and
frequency penalties. Unspecified filters are disabled and unspecified penalties
are neutral. Disabled top-k is stored as `-1`; Step 5's recommended `0` has the
same meaning. Packages exported without ranges continue to fix every distribution
parameter at their stored defaults, including Qwen presence penalties.

Native extensions such as `top_k`, `min_p`, repetition penalties, token stops,
`grammar`, and `chat_template_kwargs` remain available. JSON output formats and
Lark output grammars constrain the final answer while permitting the model's
reasoning and tool-call envelopes. Tool arguments do not use the final-answer
schema. Active tools and thinking remain available; assistant continuation and
custom stops cannot be combined with output constraints.
For conversational JSON output, the schema and a JSON-only formatting instruction
are included in the system prompt as well as enforced by token masks.

Anthropic thinking budgets are advisory effort hints, not independent token
budgets. Budgets up to 5,000 map to low effort, up to 20,000 to high effort, and
larger budgets to maximum effort. Responses and Anthropic named effort settings
(`reasoning.effort` and `output_config.effort`) accept named levels and map into
the same neutral levels. They do not accept numeric scores. The frontend resolves
those levels against the effort vocabulary accepted by the packaged template: for example Qwen uses
`low/medium/xhigh`. Native DeepSeek V4 and V4.1 prompt encoders accept
`low/high/xhigh/max` and default to thinking enabled and high effort.
DeepSeek V4 rejects numeric effort. DeepSeek V4.1 additionally accepts a JSON
integer from **1 through 100** in the native `reasoning_effort` option, either
at the request's top level or in `chat_template_kwargs`. Its named levels map
to scores `low=50`, `high=75`, `xhigh=75`, and `max=100`.
For `POST /v1/chat/completions`, for example:

```json
{
  "model": "DeepSeek-V4.1-Flash",
  "messages": [{"role": "user", "content": "Explain why this algorithm terminates."}],
  "reasoning_effort": 80,
  "max_tokens": 1024
}
```

Alternatively, replace `"reasoning_effort": 80` with
`"chat_template_kwargs": {"reasoning_effort": 80}`. Values outside 1–100,
floating-point numbers such as `80.0`, booleans, and numeric strings such as
`"80"` are rejected. Omitted values retain prepared defaults; explicit `null`
selects the encoder's high default. Disabling thinking suppresses the effort
prefix. The score is a prompt hint, not a token count or hard reasoning budget;
Anthropic `thinking.budget_tokens` retains the threshold mapping described above.

Both native DeepSeek encoders reject `drop_thinking`, assistant continuation,
and `add_generation_prompt=false`.
Explicit values are validated by the selected encoder. Packages exported with
Jinja templates retain their packaged template semantics.
Kimi K3 always uses thinking: its template ignores `enable_thinking=false` and
`thinking=false` without an error, preserves assistant reasoning history, and
accepts `low`, `high`, or `max` effort (default `max`).
The generation output limit includes reasoning and tool syntax.

Anthropic thinking blocks carry a deterministic `flare-thinking-v1:` marker
derived from their visible text. Replayed local markers are checked for matching
text. This marker is a local round-trip consistency check, not provider
authentication; readable thinking from other providers can be replayed.

Usage counts native prompt and generated tokens, including model control tokens.
Chat and Completions always include `prompt_tokens_details.cached_tokens`,
equivalent to enabling vLLM's `--enable-prompt-tokens-details`. This counts the
prompt tokens actually reused at admission, including restored prefixes; it is
zero when no prefix is reused. Cached tokens remain part of `prompt_tokens`
and `total_tokens`. Image requests also include
`prompt_tokens_details.multimodal_tokens.image`, the image feature tokens already
included in the prompt total. Responses reports reuse through
`input_tokens_details.cached_tokens`.

Anthropic reports reused tokens in `cache_read_input_tokens` and the remaining
prompt tokens in `input_tokens`; their sum is the token-count endpoint's result.
`cache_creation_input_tokens` remains zero because Flare does not separately
account for cache writes. These are native reuse counts, independent of
Anthropic cache-control billing policies. In streams, `message_start` initially
reports the full prompt with zero cache reads; `message_delta` supplies the final
input, output, and cache counts. Chat and Completions include final usage when
`stream_options.include_usage` is true. Chat token observation extensions
(`return_token_ids` and `return_token_timestamps`) preserve usage details,
including the legacy terminal usage when token IDs are requested without a
separate usage chunk. Stop sequences report the exact matched string through
Anthropic `stop_sequence`.

For cumulative usage during Chat and Completions streaming, set
`"stream_options": {"include_usage": true, "continuous_usage_stats": true}`.
The `continuous_usage_stats` option defaults to false and requires
`include_usage: true`. Enable it when measuring streams that may be cancelled
before the final usage chunk arrives. Each emitted chunk carries cumulative
native token counts; text chunks can combine multiple tokens, so counting
chunks does not measure generated tokens. Final usage is still emitted.

## Raw completions

`prompt` accepts a string, an array of strings, a token-ID array, or an array of
token-ID arrays. Text uses the prepared tokenizer's normal special-token policy;
explicit IDs are preserved. Empty text requires a packaged BOS token; empty token
arrays are rejected. Raw prompts bypass chat templates and reasoning/tool parsers.

`n` generates independent choices for each prompt. Choice indices are
`prompt_index * n + choice_index`; buffered results use that order and streams may
interleave indices. The configured or explicit seed advances by choice index. `echo` prepends the
original text or decoded token prompt. `max_tokens: 0` performs tokenization and
optional echo without GPU submission. Prompt usage counts each input once;
cached usage uses the first choice's admission count for each input, independent
of completion order. Zero-token requests report zero cached tokens because they
do not perform a cache lookup. Completion usage sums all generated choices.
`best_of` must equal `n`; ranked
sampling, suffix insertion, prompt scoring, and log probabilities are rejected.

## Clients and tests

The OpenAI SDK uses `base_url="http://127.0.0.1:8000/v1"`. The Anthropic SDK uses
`base_url="http://127.0.0.1:8000"`. Supply the configured serving key as `api_key`;
an arbitrary nonempty key satisfies SDK construction when server authentication
is disabled. The SDK tests use OpenAI 2.54.0 and Anthropic 0.125.0.

[Claude Code settings](clients/claude-code.json) configure its base URL, model,
and API key. Client settings may use environment variables; Helios behavior is
configured through its CLI, configuration file, and prepared package.

[Codex provider settings](clients/codex.toml) and its
[model catalog](clients/codex-models.json) select HTTP Responses, disable hosted
search, and describe local model capabilities. Set the catalog's context limit
to the loaded package's capacity. The catalog uses Codex's freeform `apply_patch`
tool with completed-input grammar validation.
The coding-client tests exercise Codex 0.162.1 and Claude Code 2.1.296 in normal
and bare modes with actual file inspection, editing, and test execution in
temporary repositories.

Install SDK test dependencies and run against an idle native service:

```sh
uv sync --locked --extra api-test
FLARE_API_TEST_URL=http://127.0.0.1:8000 \
  .venv/bin/python -m pytest tests/server/test_protocols.py -q
FLARE_API_TEST_URL=http://127.0.0.1:8000 FLARE_RUN_CLIENT_TESTS=1 \
  .venv/bin/python -m pytest tests/server/test_coding_clients.py -q
cargo test --manifest-path helios/Cargo.toml --features server,runtime-package --lib
```

`FLARE_API_TEST_KEY` supplies an authenticated test endpoint's key. Coding-client
tests use temporary repositories and check a real read/edit/test task. Scheduler
regressions in `tests/server/test_scheduler.py` run against ordinary, fixed-width,
and adaptive speculative execution. These environment variables configure tests,
not the native server.

## System One decisions

`POST /v1/systemone` accepts the [TypeSafe System One schema](https://api.typesafe.ai/openapi.json)
for `noul`, `choice`, and `score` questions:

```json
{"model":"local","state":{"message":"Charged twice"},"questions":{
  "refund":{"type":"noul","instructions":"Is a refund requested?"},
  "team":{"type":"choice","criteria":{"billing":null,"support":"Other help"}},
  "urgency":{"type":"score","criteria":["Routine","Urgent"]}}}
```

The response contains the loaded `model`, a named `answers` map, and
`usage: {input_tokens, output_tokens}`. Noul returns `noul: P(true)`; Choice
returns `choice`, named `probabilities`, and `confidence`; Score returns the
probability-weighted zero-based `score`, original levels in `legend`, indexed
`probabilities`, and `confidence`. Confidence follows the
[Jev formulas](https://docs.typesafe.ai/confidence), with first-option tie breaking
and singleton confidence one. Normalizing label probabilities does not provide
empirical calibration for the loaded model.

Requests require a model string, a string/object/array state and 1–64 named
questions. Instructions and descriptions preserve structured JSON. Choice has
1–128 criteria; Score has 1–10 nonnull levels. Available single-token labels and
prepared context/output capacities may lower these limits. Schema failures return
Jev-style HTTP 422 `detail` entries. Malformed JSON, incompatible templates and
unsupported packages return 400; authentication and operational errors use the
existing generation error format.

The native scorer compiles independent questions after a common state message,
verifies distinct single-token synthetic labels after the complete rendered
prompt, and reads raw `logprob_token_ids` with one generated token per question.
Questions execute sequentially, allowing ordinary prefix reuse after the first
question. Usage sums actual logical prompt counts, including cached input, and
actual generated read tokens. No prose or streaming is returned. Dropping a
request cancels its current child through the ordinary generation lifetime.

The scorer uses the normal assistant generation prompt with thinking disabled
where supported and an `Answer:` cue in the final user turn. Labels are verified
as the first assistant token against the actual rendered prompt, including with
the native DeepSeek encoders. Older packages without requested-ID outputs require
re-export.

PNG/JPEG [image assets](https://docs.typesafe.ai/images) require
`TypeSafe-Danger-Zone-Images: true` and a vision package. Supply base64 assets or
uncompressed multipart/form-data with an application/json `request` part first,
followed by uniquely named image parts. An asset's `part` takes precedence over
`data_base64`; every part must belong to exactly one asset. References replace
string values, including nested state/instructions/criteria, with whole-value
matches first and then nonoverlapping left-to-right matches. Keys remain text;
legends retain original references. Unused assets are rejected. Existing body,
context and image limits apply, including to expanded prompts. Multipart content
encoding must be identity (or absent); compressed multipart returns HTTP 415.
JSON supports the usual compressed encodings. Box,
audio/video and custom question types are unsupported.

## Decisions

`POST /v1/decisions` adapts the System One scorer to the concrete vLLM Decisions
text protocol. It accepts `model`, text `input` (or user messages containing text
and `input_text` parts), and an ordered array of 1–64 `questions`:

```json
{"model":"local","input":"Charged twice","questions":[
  {"type":"predicate","name":"refund","instructions":"Refund requested?"},
  {"type":"choice","instructions":"Which team?","choices":[
    {"value":"billing","description":"Payments"},{"value":false}]},
  {"type":"score","instructions":"Urgency?","levels":[
    {"label":"Routine"},{"label":"Urgent","description":"Act today"}]}]}
```

Instructions are required strings. Names are optional strings; omission returns
`name: null`, explicit null is invalid, and duplicate names are permitted.
Choices contain 2–128 distinct string/boolean values and optional string
descriptions. Scores contain 2–10 labeled levels with optional descriptions.
Single-token label availability and prepared capacities can further restrict
these bounds. Text parts join with newlines and messages with blank lines.
Nullable `safety_identifier` accepts up to 128 characters as inert caller metadata.
Unknown fields, images/assets, other roles and unsupported question types return
OpenAI-style HTTP 400 errors. Multipart is unsupported on this text-only route.

The response has the loaded `model`, ordered `answers`, and `usage`. Predicate
answers contain `probability`; Choice answers preserve the selected typed
`choice` and an array of `{value, probability}` entries. Score answers contain
`score` and `{value: index, label, probability}` entries. Choice/Score confidence
is the winning label's full-vocabulary probability: its normalized probability
multiplied by total raw label mass. System One retains its distinct Jev confidence
formulas. Neither route promises empirical calibration.

Usage sums actual logical input and generated read tokens across questions,
including `input_tokens_details.cached_tokens`, `output_tokens_details.reasoning_tokens: 0`
and `total_tokens`. Cache writes are not measured and are reported as zero.
The route shares System One's prompt compilation, exact-ID reads, sequential
child lifetime, template constraints and older-package rejection; it adds no
second scoring path.
