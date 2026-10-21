# Runtime packages: the preparation boundary and native serving

Flare serves a prepared model from a runtime package: a metadata-only primary
`.safetensors` file and immutable sibling payload files holding everything a Rust
process needs to rebuild the model's CUDA graphs and storage without a Python interpreter. Preparation still happens in Python
with PyTorch, Transformers, and b12x. Serving requires the NVIDIA driver,
the package, the Rust `flare` executable, and its system library dependencies
(including libturbojpeg). Python and preparation tools are not required.
The [core design](core-design.md) defines the model contracts, scheduling,
and cache behavior that the served process implements; this document defines
how the prepared state leaves the Python process and how it is reconstructed.

## 1. Components

![Flare component relationships](images/component-architecture.png)

| Component | Role | Implementation |
|---|---|---|
| ignition | Internal preparation command: plans, exports, and prepares cached packages for native serving | The Python command line under `ignition/cli/` |
| heliopause | CPython extension imported by Ignition. It supplies the native preparation allocator and resource lifetime hooks | The `heliopause/` Rust crate; Python distribution `flare-heliopause`, import `flare_heliopause` |
| Helios | Rust runtime library. It rebuilds a package into a live CUDA context and serves it | The `helios/` crate, including `src/runtime_package/` and the server |
| flare | Public serving command. It links Helios to load a package and run the server | The `flare/` Rust crate; section 5 |

Preparation knowledge, recipes, planning, hub access, the kernel cache,
compilation, tuning, and capture stay on the preparation host. Ignition imports
`flare_heliopause` and exports through `ignition/preparation/runtime_package.py`
in its own Python process. `ignition run` exports a cached package when needed,
closes the prepared model, and calls `execv` to replace its entire address space
with the Rust `flare` executable. A cache hit goes directly to `execv`.
`ignition export` writes the package and exits. Graphs and Python resource owners
do not cross the executable boundary. Helios has no CPython dependency;
heliopause owns the CPython hooks and preparation allocator and uses Helios's
CUDA interfaces. Heliopause has no executable or embedded interpreter. `flare` and Helios are
the public deployment surface. Ignition and heliopause are internal preparation
tools. Dependency direction is `flare → Helios` and `heliopause → Helios`;
Helios has no dependency on either preparation tool.

### Distribution builds

From the repository root, build public executables with:

```sh
cargo build --locked --release --manifest-path flare/Cargo.toml
```

The executable is `flare/target/release/flare` (`flare.exe` on Windows). Docker and
native-serving CI use this build. The release profile strips symbols and retains
Rust panic messages, source locations, and unwinding behavior. Builds use the
stable toolchain without unstable compiler flags. Source paths, required dynamic
imports, API routes, metrics, and operational errors remain observable. Publisher
stamping through `python -m ignition.release` preserves the embedded
verification-profile section.

## 2. The preparation boundary

Ignition records four things while preparing a model, and refuses to
write a package when any of them is incomplete.

**Storage.** PyTorch's persistent device allocations come from one address range
reserved before model loading: 1 TiB per rank from `0x0800_0000_0000` on Linux or
`0x0400_0000_0000` on Windows, backed on
demand with `cuMemCreate` and `cuMemMap` at allocation granularity. Packages
reserve GiB-aligned ranges covering stored allocations and the declared maximum
extents of dynamic caches; unused gaps in the preparation arena are omitted. PyTorch
allocates through a `MemPool` whose pluggable allocator is the Rust arena in
`heliopause/src/allocator.rs`, so weights, state, pages, and the
graph pool all live inside the range. Ordinary preparation and capture use
separate PyTorch pools backed by that arena. Mapped host buffers, including
n-gram request mailboxes, use fixed addresses from `0x0c00_0000_0000` when
the device rank ranges fit below it. Larger Linux layouts start host storage
at `0x2000_0000_0000` to avoid overlap. Each rank has a disjoint 1 TiB host
range; the native allocator reports the selected base to Ignition.
Linux maps and registers host pages; Windows uses host-backed CUDA VMM.
Both expose the recorded address to the CPU and GPU. A CUPTI subscriber logs
driver allocations and removes records when their context is destroyed.
Persistent device allocations outside the arena are grouped into complete
VMM pages and placed within reservations aligned to the device's allocation
granularity. Reservations without retained allocations are omitted. Only their
live byte spans are considered for payload storage; page gaps are zero. Before
copying, the exporter checks that each external allocation belongs to the
serving context. On Linux integrated GPUs, preparation reserves the low address
window through 256 GiB without physical backing. Windows retains the driver's
low allocation pool. Export on either integrated platform rejects reachable
external device storage below 256 GiB, where context mappings can prevent replay.

The model exporter retains allocations reachable from graph arguments, copy
and memset endpoints, module globals, native execution buffers, and declared
peer storage. It follows pointer-bearing allocation contents while writing
tensor payloads. Checkpoint transfers record numeric tensor spans with weak
storage ownership; expired or resized storage cannot reuse those declarations.
Live floating-point tensor views supply additional numeric spans, excluding
integer aliases and holes in strided layouts. Native input/output buffers are
numeric by their execution contract. These spans are leaves in the traversal:
their storage is retained when referenced, but their values are not interpreted
as addresses.

Numeric-span discovery skips zero-initialized allocations, which have no
payload to scan, and deduplicates identical live tensor views. Strided layouts
retain their holes; fragmented persistent views without bounded span metadata
remain conservative rather than expanding an unbounded interval catalog.

Preparation uses temporary PyTorch pools backed by the package allocator during
loading and tuning. Retiring a pool releases whole free segments before forward compilation;
live tensors retain their storage and addresses. Address candidates within those
pools must fall within requested allocation sizes, including in retired pools
with surviving tensors. Inactive blocks and the padding between a live block's
requested size and its allocator-rounded size are cleared and excluded from
payload reads. Their stale contents cannot retain other allocations, and stale
address candidates cannot retain those unused bytes' backing segments. This
filter excludes the private graph pool: its intermediates remain live through
captured replay even when their Python tensors have been released. Explicit
resource bindings retain their allocations independently of candidate scanning.

Compiler-declared scalar arguments are also leaves. Triton signatures are
checked against the driver's parameter sizes; kernels with integer-to-pointer
conversions, inline assembly, or opaque calls retain conservative argument
scanning. Untyped structs, tensor maps, globals, and integer/byte storage without
a numeric role are scanned for address candidates. This can retain false
positives and retains complete backing segments, including unused bytes within
a reachable segment. The preparation metadata does not enter the serving index.
Omitted arena allocations keep their virtual ranges reserved and physically unmapped,
so an unexpected access faults instead of reaching another allocation.

Before export, model recipes declare disposable KV and scratch storage,
including target and MTP page pools, operator workspaces, decoder and vision
intermediates, and sampling and verification temporaries. The graph memory
pool holds only graph-local intermediates and is also cleared, as are unused
bytes in the preparation allocator. Weight tensors, MTP weights, rotary
tables, pointer tables, and required initialized metadata retain their contents.

The primary file is a valid safetensors container with three one-dimensional
`U8` tensors and `__metadata__: {"format": "flare-snapshot"}`. Its little-endian
u64 header length precedes a JSON header padded to a 4 KiB data boundary; the
header is limited to 64 KiB. `flare.payload` has zero length. `flare.metadata`
contains the signed `ProtectedIndex`; `flare.layout` follows it and maps signed
logical byte ranges to immutable payload files. Each metadata tensor is bounded
at 128 MiB. No model payload bytes reside in the primary.

The envelope uses version 4. Its authenticated `ModelIndex` uses version 4 for
logical monolithic addressing or version 5 for logical companion addressing.
The snapshot storage mapping does not change signed offsets or encryption
associated data. Helios also loads `format: flare` packages with physical payloads
in the primary or numbered companions. Absent filename and stored-part fields
retain their original meaning. Conversion preserves the signed envelope exactly.
Unsigned rank staging uses `format: flare-staging` and cannot load in Helios.

`ModelIndex.contract` is a typed `ServingContract`. It declares serving defaults
and a versioned `ModelContract` containing vocabulary bounds, vision capacities,
attention-history storage, page-table views, graph value sources and dependencies,
completion cursors, and frontend syntax. Field 4 is reserved; arbitrary Protobuf
`Struct` configuration is not accepted. Execution ABI 2/3 and protected envelope
version 4 are versioned separately from model-index versions 4 and 5.

Two prepared-contract fields fail validation when absent or out of
range. `vision` must decode to a well-formed `VisionCapabilities`: a
text-only package encodes the explicit all-zero value (`feature_slots`,
`features_per_request`, `max_images`, and `patches_per_feature` of 0
with empty `patch_capacities`), and an absent or null `vision` fails
the contract decode (a null submessage raises `invalid type: null,
expected struct VisionCapabilities`). Every graph contract also carries
`embedding_tables`, whose entries index the package's embedding-table
catalog and are validated against the catalog size at package
validation: a package that ships no embedding-table resources must
give every graph an empty `embedding_tables`, and any entry outside
the catalog fails with `graph requires an absent embedding-table
resource`.


Semantic bindings name a destination buffer and an enum-selected value source.
Indexing distinguishes scalar, submission row, active row, physical request row,
and packed-token storage. Offsets and padding are explicit. Omitted transfers
retain device bytes. Persistent state views contain byte offsets, strides, and
extents within declared buffers; graph state accesses describe reads and writes.
Paged key/value and latent histories refer to actual cache planes. Continuation
rows hold recurrent state, sliding windows, partial pages, and other captured
model histories. Shared cache allocation sizes the physical backing independently
of model identity. Native code implements no attention state-update computation.

Frontend enums independently select tool/reasoning parsers and image preparation
and layouts. Marker strings, feature tokens, and atomic
image spans are data. Enum zero and unknown identifiers are rejected; enum
numbers are stable and removed values must be reserved. Frontend identifiers
cannot select scheduler, cache, text-graph packing, or completion policy.
The serialized `prefix_boundary` field remains readable for package compatibility;
message checkpoints derive from exact rendered prefixes without role filters.

The index stores complete recipe-declared sampling parameters in
`DefaultSamplingParameters` and structural frontend settings in
`FrontendSettings`. Sampling defaults are manually verified against model-card
recommendations for the default serving mode; checkpoint configuration does not
select sampling policy. Missing values reject loading rather than inheriting
another model's settings. `DefaultSamplingParameters.ranges` contains optional
inclusive `min` and `max` bounds for temperature, top-p, top-k, min-p, and the
repetition, presence, and frequency penalties. Each supplied range must contain
its existing default, and both endpoints must be valid sampler values. Numeric
request values are clamped to these bounds; omitted ranges fix parameters at
their defaults. Packages without range fields remain loadable without re-export
and use this fixed-default policy. Seed remains request-controlled.

`FrontendSettings` contains EOS token IDs, the special-token strings used by the
chat template, and optional image settings. Qwen image settings contain resize
area limits, learned-position geometry, and image/video token IDs. MiMo image
settings contain resize limits and channel normalization. DeepSeek's image profile
has fixed geometry validated by Ignition. Preparation reads and validates structural
checkpoint config files, then exports only these consumed fields. The only frontend
payload assets are `tokenizer.json` and the Jinja chat template; raw generation,
tokenizer, processor, and model configuration files are not included.

Payload blobs contain whole resources: stored allocations, embedding planes,
graph bodies, cubins, module globals, or frontend assets. Blob names are SHA-256
hashes of the complete safetensors file. Adjacent stored resources are packed up
to the selected shard limit without crossing rank or logical-file boundaries.
Executable and supporting resources use separate shards. Whole resources remain
contiguous; a graph change can reuse unchanged weight shards. Changing any bytes
in a weight shard gives that entire shard a different identity. Blobs remain
uncompressed. New exports
reuse identical stored bytes through hardlinks, and upload clients can deduplicate
the same named content across revisions. Upload clients may still scan local files.

Every resource span stores a logical sibling `filename`, a byte `offset` relative
to that file's logical `flare.payload`, and its byte length. An empty filename
selects the logical primary. Snapshot layout segments resolve that span to one
contiguous physical blob range. In a `format: flare` package the logical filename
is the physical filename.
Spans cannot extend into `flare.metadata`. The index is
limited to 128 MiB and contains ordered rank bundles, peer-memory ownership,
serving configuration, frontend asset spans, allocation descriptions, kernel
metadata, and graph declarations. Each tensor-parallel size requires a distinct export.

Graph bodies, cubins, module globals, packed tensor data, tokenizer data, and
other frontend assets live in separate indexed payload spans. A decrypted graph body contains its declared number of nodes followed by its
edges. Each record is a little-endian u32 length followed by one Protobuf
`Node` or `Edge`, with a 1 MiB record limit. Kernel arguments, node attributes,
and edge data are Protobuf byte strings. Graph bodies are independent; the
loader need not read an unselected graph to locate or select another graph.
Rust bindings are generated by `prost-build`; Ignition uses checked-in Python
bindings generated by `protoc` from the same schema.

Allocation extents are separate from stored payloads. Each allocation has
ordered, disjoint `data_spans` containing destination byte offsets and lengths.
These are memory destinations and must leave `filename` empty.
Its payload concatenates those spans without padding. Declared disposable
ranges are excluded before device reads; remaining ranges are checked for
zero-filled 64 KiB blocks, which are also omitted. An entirely zero allocation
has an empty span list and no payload. Helios zeroes omitted regions and
copies only the stored spans to their declared offsets. Runtime-sized row and
KV allocations carry no tensor payload. The loader accepts the indexed file
format; incompatible packages must be exported again with Ignition.

Each preparation rank writes its resource file before the model file is
assembled. After all ranks finish exporting and peer ownership is recorded,
Ignition drains and closes the prepared models, joins peer processes, and
releases registered host buffers. Final file assembly requires only rank files
and their indexes. N-gram table payloads are appended directly from checkpoint
file ranges during rank export, without materializing their contents in
preparation memory.

### File placement and addressing

`ignition export`, `ignition run`, `ignition sign`, and `ignition convert` pack
published payloads to at most 64 GB (64,000,000,000 bytes), including the
4096-byte header. The reader retains its 64 GiB limit for existing packages. The
`[publishing] shard_size` setting or `--shard-size SIZE` can select a smaller limit.
Zero is invalid. The package cache key includes the resolved byte count. An
indivisible resource exceeding the limit fails publication; no oversized file is
published. All model payloads reside outside the primary.

The layout JSON has `version: 1` and a `files` mapping. Each logical filename maps
to ordered `[length, blob_filename]` segments. An empty blob filename represents
zero padding. The empty logical filename is required. At most 100 logical files
and 65,536 segments are allowed. Blob names are 64 lowercase hexadecimal SHA-256
digits followed by `.safetensors`. Each blob contains only a one-dimensional
`U8` `flare.payload` tensor and `__metadata__: {"format": "flare-data"}`. Its header
is exactly 4096 bytes including the length prefix; its payload must exactly match
the segment length. A repeated blob name must have the same length.

The layout supplies physical addressing, not publisher authority. Signed resource
bounds, executable ciphertext hashes, and optional stored-data verification apply
to the reconstructed logical bytes. The native loader pins all selected blobs and
rejects resources spanning multiple blobs or padding. Hosts need sufficient open
file descriptors for their selected blobs and runtime resources.

Ignition stores blobs under `paths.package_cache/blobs` and hardlinks them into
snapshot directories. Existing blobs are checked before reuse; new blobs are
hashed while written, synchronized, installed without overwriting, and made
read-only. Snapshot and cache must share a filesystem. Publication installs all
payload links before atomically publishing the metadata primary. A failed export
can leave reusable blobs and links, but never a primary naming incomplete payloads.

The converter validates the source publisher's public profile and signatures,
reads all stored data and authenticated resources, and checks the relocated blob
bytes before publishing the original envelope with a new layout. It needs no
private key, checkpoint, or GPU. The source remains intact. `--no-weights` export
requires all stored-data blobs to exist; preparation, capture, and content reads
still run. Captured allocations are scanned into sparse 64 KiB nonzero spans;
cache hits do not write staging weight files.

Companion references are portable ASCII basenames ending in `.safetensors`, with
an alphanumeric first character and only letters, digits, dots, underscores, and
hyphens thereafter. Paths, `..`, and device names in signed references are
rejected. A companion directory entry may be a symbolic link to a regular file,
including a file on another filesystem. The loader follows the link when opening
the companion, validates the opened file's type, header, length, and payload
ranges, and retains that file handle. Broken links, link loops, and nonregular
targets are rejected. Retargeting a link does not replace an already opened
companion. The operator controls these filesystem links; signed basenames do not
confine their targets to the package directory. Package files must remain
immutable while they are being transferred or served.

The loader retains the package directory and opened files and caches each
payload base and direct-I/O handle. Allocation uploads add the resource offset
to that base. Embedding readers retain their file handles before serving; block
coalescing compares both file identity and offset.

Logical file placement precedes encryption, so executable keys and authenticated
chunk data use logical offsets. Physical blob placement can change without
re-signing or decrypting a resource.

`StoredData.parts` describes the ordered file-qualified ranges of one owner's
stored data. Its SHA-256 covers their concatenation, including declared alignment
padding. With parts present, `offset` is zero and `nbytes` is the sum of their
lengths. With parts absent, `offset` and `nbytes` select the legacy primary range.
Each allocation or plane must fit within one signed part. Bounds, resource roles,
and overlaps are checked per file even when the optional content scan is disabled.

**Offloaded n-gram tables.** The model index declares token normalization or
EOS-bounded history, per-head n-gram orders, XOR multipliers, modulus sizes,
row offsets, quantization, and separate weight and scale payload spans.
DeepSeek V4.1 uses FP8/E8M0 rows; Qwen PLE uses group-16 NVFP4 rows with FP8 block
scales and a global FP32 scale. Each table is stored once for all TP ranks.
Indirect delivery is the sole n-gram execution mechanism. Graphs bind compact
GPU row buffers and named external event waits at their lookup consumers.
Table storage is selected when loading and does not change graph bindings.

DeepSeek V4.1 and Qwen hash tokens in their graphs, including GPU-produced proposals.
DeepSeek V4.1 retains the three preceding tokens in continuation state across prefill
chunks and reused prefixes; Qwen preserves its EOS-bounded history.
The graph copies row IDs to a mapped host mailbox and publishes its submission
sequence with a system-scope release operation. The reader observes that
sequence with an acquire operation before accessing the IDs. Both models publish
after input-state preparation, before target layer 0. Qwen consumes rows at
layer 1; DeepSeek V4.1 consumes its two tables at layers 1 and 14.

Mapped mailbox addresses and synchronization are independent of the host operating
system. Linux backs them with registered host pages; Windows uses host-backed CUDA
VMM with CPU and GPU access at the recorded address. Windows rounds and merges
adjacent host allocations to its CUDA allocation granularity while preserving
each buffer's address and extent in the package. No OS-specific mailbox graph or
package encoding is required.

Ranks with identical table planes share a reader that gathers each step's rows once.
`EngramTable.shards` declares disjoint, contiguous row ranges in rank order, with
separate weight and scale spans for each owner. These partitions cover the full
table and publish partition-local row IDs; excluded lookup lanes publish `-1`.
Each distinct partition has its own reader, staging, and submission state. An
absent partition list retains the shared planes and delivery semantics of
previously exported packages. The default, `--embedding-table-storage disk`, uses
direct I/O, deduplicated 4 KiB blocks, and a 128-entry queue with 64 KiB
registered buffers per table. `--embedding-table-storage ram` holds tables in ordinary
resident memory and gathers the same rows; on GB300, Qwen's disk reads lengthened
32-request rounds by a quarter. RAM startup loads each plane with four readers
issuing contiguous 4 MiB direct reads into aligned resident storage. It does not
construct row lookups, zero-fill the table, or copy through row staging while
loading. Both modes write bounded pinned
staging buffers. At 4,096 tokens, Qwen needs 5.625 MiB of row staging and a
512 KiB request mailbox per rank. The two DeepSeek V4.1 tables need 49.5 MiB of
pinned row staging per reader group. Partitioned DeepSeek tables retain the
24-lane compact buffer layout, with unowned lanes zero-filled. Direct-I/O buffers
add 8 MiB per table in disk mode.

`--embedding-table-cache-size SIZE` adds a bounded cache on each host above the
disk reader. Each entry holds one immutable packed-weight row and its scale row.
Four-way sets protect rows by approximate usage frequency, with recency breaking
ties. A bounded four-counter estimate records each distinct requested ID per
gather, including misses. Counters saturate at 255 and halve after ten times the
counter-array length in observations, allowing workload changes to retire prior
popularity. An incoming row with a lower score than every resident in its set
is served without admission. Misses use the same coalesced direct-I/O reader and
enter the cache only after a successful gather. The budget includes payload,
index, usage counters and preallocated miss workspace. The host budget is divided
equally among distinct table catalogs, then across each catalog's tables in
proportion to their stored bytes. A table reader owns its cache for the loaded
package's lifetime; ranks with identical catalogs share that reader.
Each server start creates an empty row cache, populated by successful request
gathers.

The loader commits this storage before the existing KV/continuation capacity
planner reads available memory. On GB10, the fixed host cache therefore reduces
the physical memory available to those pools. Prefix-checkpoint pinning,
publication, and retirement remain owned by the KV cache and scheduler. The
Engram cache copies hits into the existing pinned staging buffers before GPU
upload, so its entries require no GPU retirement fence. Staging reuse continues
to wait for every rank's consumer completion.

Before launching a graph, each rank queues a readiness-generation wait, row
copies, and an event record on a copy stream per table. Reader threads publish
the generation only after filling the host buffers. Each table requires two
H2D copies per rank per step: one contiguous weight plane and one contiguous
scale plane. In RAM mode the CPU gathers each row into those planes before
either upload. The captured graph waits
on the recorded event at the table consumer, allowing preceding layers to run
while reads and copies are pending. Storage mode does not change the model
graph catalog. All rank completions must finish before staging storage is
reused. Read failures release pending waits with zeroed rows, poison the
executor, and suppress completion delivery. Shutdown cancels pending mailbox
waits and drains the streams before releasing storage.

**Kernels.** The same subscriber copies the image of every module and library
the driver loads, from the driver-API load calls and from the resource
callback that also covers modules loaded internally by CUDA libraries. Each
kernel node's function maps to its image through `cuFuncGetModule`. Resource
callbacks during lazy loading are associated with the driver call that
materializes the module, including launches using context-independent
`CUkernel` handles. A unique symbol-name match is a fallback. The exporter records
each function's parameter layout from `cuFuncGetParamInfo` and its dynamic
shared-memory, carveout, and cluster attributes. ELF symbol enumeration and
`cuModuleGetGlobal` capture initialized module globals and constants; native
loading restores their contents after loading each cubin.
Source-bearing debug sections, embedded PTX, compiler comments, and ELF source-file
names are erased without changing CUDA section or symbol indices. Assertion
filename globals are replaced with `kernel`; assertion conditions and messages
remain intact. Defined function and global symbols receive module-local opaque
names. Section names and package lookup names use the same aliases; undefined
imports and CUDA-reserved symbols retain their driver-facing names. String tables
are rebuilt and their original bytes cleared. Instructions, relocation indices,
CUDA metadata, and initialized model data retain their executable layout.

These transformations remove descriptive metadata, not executable semantics.
Cubins remain disassemblable, and instruction patterns, required CUDA metadata,
and remaining runtime diagnostics can identify compiler and kernel families.
Package signatures authenticate the publisher. Executable-resource encryption
does not provide confidentiality against an operator controlling the runtime.

**Graphs.** Capture keeps the graph templates alive, and the walker records
each template node by node: kernel nodes with grid, block, shared memory,
every nonzero launch attribute, and the parameter bytes copied verbatim;
memset and contiguous device-to-device copy nodes with their driver
parameters; empty nodes; named external event-wait nodes; and every edge
with its edge data. CUPTI node ids preserve construction order, including
independent branches. Kernel parameters may use argument arrays or a packed
`extra` buffer; the exporter reads individual parameters using the driver's
offsets and sizes. Parameterless kernels retain an empty argument list. External
waits refer to declared readiness events that the loader recreates. Host
nodes, child graphs, undeclared event waits, event-record nodes, conditional
nodes, and device-updatable nodes fail export.

**Contract.** The package carries the Helios execution manifest with buffer
addresses rewritten as allocation references, the server configuration, the
frontend assets, the device architecture, and the context flags, cache
configuration, and limits. Parameter bytes are stored exactly as captured.
Pointer and tensor-map fields may additionally be declared as relocations,
which lets a loader place storage elsewhere and rewrite the fields, tensor
maps through `cuTensorMapReplaceAddress`; the model export does not declare
them and relies on identical addresses instead.

### Preparation rebuild metadata

Exports through `ignition export` and `ignition run` record an optional
`ModelIndex.rebuild` manifest in the signed metadata of the primary shard.
It identifies the recipe, preparation source identity, qualified checkpoint
repository and revision, tensor-parallel size, concurrency, chunk size,
speculation, and selected preparation settings. A local checkpoint override
is marked explicitly and carries no asserted upstream revision.

Each recipe also supplies its named weight owners. After reachability pruning,
the exporter records retained tensor views in `BundleIndex.rebuild_weights`:
owner paths, packing-owner types, dtypes, shapes, element strides, and
allocation-relative byte offsets. The containing bundle identifies the rank.
Allocation resource spans and the package layout locate those bytes in the
payload shards; repacking and re-signing preserve the references. Aliased views
share storage. Catalog creation reads tensor metadata and creates no weight copy.
Meta tensors, discarded storage, and disposable scratch are listed as
unavailable rather than treated as recoverable weights.

These catalogs describe prepared storage, whose names and packing can differ
from the original checkpoint. They do not serialize Python objects or execute
imports. A package-based graph rebuild still requires a compatible recipe
importer that validates the catalog and supplies any unavailable inputs;
Ignition does not yet provide that importer or a package rebuild command.
`--no-weights` continues to prepare from the source checkpoint and reuse cached
payloads. Offloaded embedding planes retain their existing typed table metadata.

Flare exposes the manifest and per-rank tensor counts through `flare inspect`;
`--all` includes the tensor catalogs. Absent manifests are reported as null.
Helios ignores rebuild metadata during loading and replay, including unfamiliar
catalog versions. Previously exported packages need no conversion to serve.
Changing signed rebuild metadata requires signing the package again.

### Publisher authentication and executable access

Every final package carries three ordered Ed25519 signature groups for what
every rank shares, and one signed `RankProtection` per rank for that rank's
resources:

| Group | Authenticated contents | Payload checks |
|---|---|---|
| 1 | Model index without rank bundles, publisher namespace, source URL, development marker, model license name and link when recorded, rank count, and SHA-256 commitments to groups 2–3 | Before publisher acceptance or CUDA initialization |
| 2 | Frontend asset spans and hashes | Before those contents are consumed |
| 3 | Ranges and SHA-256 of shared embedding tables, including padding | With `flare run --verify-weights` |
| Rank *r* | Rank *r*'s bundle; its cubin and graph spans, plaintext and ciphertext sizes, and ciphertext chunk hashes; its module global spans and hashes; and the ranges and SHA-256 of its stored allocations and embedding partitions, including padding | Bundle before loading; resources as for groups 2–3 |

Stored data is ordered by owner: shared tables, then each rank's allocations and table partitions.
Each owner's signed parts describe its contiguous ranges across files. Every signature
of the ranks a process serves is mandatory even when the stored-data scan is
omitted. A process verifies every rank, except a host serving one rank of a
multi-host package, which verifies the shared groups and its own rank only, so
a copy holding only that rank's bytes verifies completely and other ranks'
bytes or signatures do not affect it. Peer transfer sends ranges separately for
each logical file for `format: flare` packages, including headers and signed
padding. Snapshot transfer sends the complete metadata primary and only the whole
blobs selected for a rank. Agent protocol 3 stores package files in a rank-specific
directory. Content-addressed files are hash-verified on arrival and hardlinked
from the agent blob cache, which survives package-directory eviction. A peer needing companions rejects older agents
before sending any files; monolithic transfer retains protocol 1/2 compatibility.
Resource classification follows actual consumers in the signed index. Missing,
overlapping, unclassified, or incorrectly classified spans fail loading. The
128 MiB metadata limit applies to the complete signed envelope. The package
contains the signing-key fingerprint, not the verification key. Keys come from
Helios's embedded publisher profiles or explicitly supplied profile files.

Cubins and graph records use AES-256-GCM in independent 4 MiB plaintext chunks,
each followed by a 16-byte tag. Each package has a random 32-byte identifier.
HKDF-SHA256 derives a resource key from the publisher verification key, with
the package identifier as salt and a domain-separated encoding of the resource
offset and the acceptance string as context. The exporter encrypts under the
publisher namespace; Helios derives from the string the user applied to accept
the package, which the trust store remembers, so acceptance is the key material
and any other string decrypts nothing. Chunk nonces encode the chunk index;
authenticated data binds the package identifier, offset, plaintext length, and
chunk index. Field encodings use little-endian u64 byte lengths followed by the
field bytes; the exporter and native reader define the matching domain strings
in `ignition/preparation/protection.py` and
`helios/src/runtime_package/protection.rs`. Re-signing ciphertext with another
key or namespace does not produce a loadable package. Weights and tables stay
plaintext and retain direct-I/O compatibility and storage deduplication.

`PackageClaim::read` returns what a package claims before verification: signer
fingerprint, namespace, official source, and development marker, syntax-checked
so a caller can resolve the signer's identity from them. `VerifiedPackage::open`
authenticates the index and publisher binding against the supplied profiles,
for every rank or for one rank of a multi-host package, and exposes the recorded
license. `VerifiedPackage::authorize` records the acceptance string and
optionally checks the stored-data contents. Only its `AuthorizedPackage` result can enter
`NativeModel::load` or `NativeResources::load`. Executable resources are
hash-checked and decrypted before CUDA receives them. Frontend assets retain
signed digest checks. Open package descriptors remain alive through native
use, including deferred frontend and embedding-table reads; replacing the
pathname cannot redirect those reads. Package files must remain immutable while
loaded. Signatures authenticate bytes and do not sandbox executable code.
Encryption does not prevent a user controlling the runtime from recovering code.

Source builds contain no embedded publisher profiles:
`helios/src/runtime_package/publishers.json` is an empty array. Helios reserves an
ELF `.flare_publishers` section for release-provisioned profiles.
Embedded profiles verify signatures when the publisher's repository cannot be
consulted; they confer no verified identity. The private release command stamps
a JSON profile array into a separately named executable:

```sh
.venv/bin/python -m ignition.release flare/target/release/flare \
  --output /path/to/distribution/flare --publishers /private/publishers.json
```

Each profile has `name`, `namespace`, and `public_key` (32 integer bytes), plus
an optional boolean `development`. The public key belongs to the Ed25519 key
used by Ignition. Names in profiles are claims; profile import does not grant
execution trust or verify identity. A key with conflicting profile attributes is
rejected.

Flare establishes identity from the publisher's own repository. On every load
that is not offline it fetches `publisher.json` from the root of the signed
official source on huggingface.co and requires the file to list the signing key
under the package's namespace. A file that omits the key rejects the package
outright, remembered or not, which is how a publisher revokes a key. An absent
file or an unreachable repository leaves the identity unverified unless an
earlier load confirmed it, and loading then needs the key from a profile or
remembered acceptance. Flare displays the publisher,
how its identity was established, the source, the recorded license and its
link, and the full key fingerprint. First use asks for the exact namespace,
which agrees to the license terms at the link, and remembers the key, its
namespace, its verification key, and how its identity was established.
Unattended approval requires both `--dangerously-approve-key FINGERPRINT` and
`--dangerously-accept-publisher NAMESPACE`; both must match even if trust was
previously remembered. Trust never transfers to a replacement key merely
because it uses the same namespace. `flare trust revoke FINGERPRINT` removes
remembered acceptance. Identity rests on control of the Hugging Face namespace,
the same account the model itself is published from.

## 3. Address rules

Parameter bytes are replayed verbatim, so a rebuilt process must reproduce
every recorded address. Three driver behaviors, measured on driver 610.57.04
with the `cr-probe` experiments under `~/projects/flare-research`, fix how
that is done.

- The driver satisfies a `cuMemAddressReserve` request from its current
  64 MiB block whenever the request fits, ignoring the requested address, and
  maps a fresh block at the requested address only when the request does not
  fit. Four 16 MiB fixed requests in a fresh process are all moved and the
  fifth is honored; a first request of 64 MiB is moved and a first request of
  128 MiB, 1 GiB, 64 GiB, or 1 TiB is honored anywhere from 8 TiB to 127 TiB.
  A package therefore uses large enclosing reservations and maps allocations
  at their recorded offsets inside them; it never relies on small fixed
  reservations. Preparation reserves a 1 TiB arena; runtime packages use
  GiB-aligned enclosing ranges for retained arena storage and external spans.
- Host memory mapped with `MAP_FIXED_NOREPLACE` and registered with the driver
  receives the same device address in every process.
- Tensor-map parameters embed a global address and are 128 opaque bytes;
  module constant and global symbols hold whatever preparation wrote. With
  identical addresses both replay unchanged.

Conditions that break address reproduction fail loudly: the loader compares
the reserved address with the recorded one and the registered host pointer
with the recorded device pointer before it loads any contents.

## 4. Rebuild and serving

`NativeResources::load` in `helios/src/runtime_package/mod.rs` rebuilds a
package in this order: create a context with the recorded flags, limits, and
cache configuration; reserve the recorded range and compare addresses; back
each device allocation at its recorded address and map each host region;
upload contents or zero-fill; apply storage relocations; load modules and
resolve functions, checking each kernel's parameter layout against the
package; set function attributes; add nodes, attributes, and edges; and
instantiate each graph with its recorded flags. The execution manifest is
then rebased onto the live addresses and validated, and the server starts
from this reconstructed contract. `flare run PACKAGE` in
`flare/src/main.rs` starts the shared server directly from the typed contract.

Before any CUDA initialization or worker threads, Flare restricts
`CUDA_VISIBLE_DEVICES` to `--gpus`, resolving indices within an inherited mask.
Rank order follows the selected list; Helios receives contiguous device ordinals
starting at zero, while loading progress retains the CLI's GPU labels. Flare
sets `CUDA_FORCE_PRELOAD_LIBRARIES=0` and `CUDA_DISABLE_JIT=1`. The loader requires
compiled cubins; missing native GPU code fails instead of compiling PTX.

`LoadOptions` selects runtime concurrency, MTP draft-token count, and adaptive
or fixed verification before any CUDA context is initialized. Ignition captures
every power-of-two request bucket through the padded concurrency maximum.
Width one covers every bucket; wider speculative and verification graphs require
`batch_capacity * token_capacity <= speculative_token_capacity`. The capacity
defaults to the larger of padded concurrency and maximum verification width.
Proposal graphs independently cover every bucket and selectable draft depth. Mixed graph coverage is sparse over prompt capacity
and verification width; see [mixed execution](scheduling-runtime-design.md).
Execution metadata
identifies graph kind, sampling mode, batch capacity, token capacity, draft
source, draft count, and verification count; the
loader uses these fields rather than graph names to select variants. It checks
required batch coverage and identical selected catalogs across TP ranks.
The scheduler compares declared prefill capacities by useful service and
estimated graph duration. A package must include its configured maximum prefill chunk;
intermediate capacities are optional and need not follow a generated sequence.
Draft count and verification count are independent declaration fields.
Verification count includes the pending input token. Proposal graphs declare
only draft count and source; verification graphs declare only verification count.
A split catalog in adaptive mode retains every captured proposal depth up to the
loaded limit. The scheduler chooses the cheapest available proposal shape that
covers its planned verification prefix and reuses verification graphs across
depths. Packages containing only the maximum proposal depth remain supported.
Fixed mode retains the selected proposal depth. Combined graphs declare both
counts. For a selected combined batch and width, the loader chooses one permitted draft count;
it never loads duplicate MTP alternatives for that shape. The index lists
each graph's function dependencies, so kernel selection also precedes graph
body reads. Selection does not infer semantics from graph names.

`flare run` accepts `--max-concurrency N`, `--mtp-draft-tokens N`, and either
`--mtp-adaptive-verification` or `--no-mtp-adaptive-verification`. Unspecified
settings retain package defaults. Reductions must be supported by captured
graphs; a concurrency limit of three still requires the batch-four bucket.
Zero drafts selects ordinary prefill, decode, mixed, resume, and prefix-cache operations.
The graph catalog's optional `speculation_enabled` field declares which serving
mode an operation supports; absent means shared by both modes. Preparation includes
ordinary operations alongside speculative operations, including width-one rounds
that maintain draft state for subsequent wider rounds. Selection removes all
unneeded variants before reading graph bodies. Captured weights and storage layouts
remain shared; pinned zero does not execute draft generation or state maintenance.
Missing variants required by the declared envelope and increases beyond package
capacities fail before allocation. Split catalogs also require proposal-free
width-one speculative graphs. Fixed loading retains those graphs and width-one
verification alongside the full loaded width where the envelope permits it.

Unselected graph templates and executables are never reconstructed. Functions
used only by omitted graphs are dropped, as are unused modules without globals.
Modules with globals retain their persistent contents. Weights, MTP weights,
and fixed scratch remain intact because opaque kernel parameters and persistent
device storage can embed their addresses. `SchedulerLimits.max_active` controls
admission, while `request_slots` preserves the captured physical buffer strides,
and completion-frame extent. The scheduler selects from the
loaded graph catalog instead of deriving a capture budget.

Continuation storage and KV pools use runtime-sized allocation records. Each
record declares its capacity group, bytes per unit, fixed virtual base, alignment,
and maximum units. The package reserves the full virtual extent and stores no
tensor payload or prescribed physical size for these records. Per-rank 1 TiB
reservations place fixed resources in the lower half and runtime-sized regions
in the upper half. Maximum extents are chosen during preparation and validated
for overflow and overlap before loading.

Qwen preparation backs only a prefill chunk plus two pages per captured request
row unless its programmatic configuration supplies an explicit page count.
Forward compilation sees the full virtual tensor shapes. Physical backing is
allocated after compiler workers exit and before warmup and capture; it does
not constrain the serving cache or the checkpoint's context limit.

Shared cache records identify the owning rank, reader ranks, and full virtual
extent. Preparation exports a shareable handle for each physical chunk added
while growing the cache. Helios maps the owning rank's selected physical prefix
and grants every reader access to that prefix; peer records allocate no storage.

Helios maps physical prefixes after loading fixed storage, kernels, and graph
executables and uploading every selected graph. It leaves 64 MiB for driver
bookkeeping, execution streams, and events, plus the package's `reserve_bytes`
when configured. Ignition exports `memory.reserve` as this optional serving
reserve, which defaults to zero; preparation's warm-up and capture headroom does
not reduce the serving budget. All ranks use the same
row and page counts, limited by the rank with the smallest remaining budget.
On integrated GPUs such as GB10, that budget is Linux `MemAvailable`, capped
by CUDA's device total. CPU allocations, registered buffers, and GPU allocations
share physical DRAM; reclaimable page cache contributes to available memory,
while swap does not. Discrete GPUs use CUDA's free device-memory query.
Dedicated rows comprise one null row and the selected request capacity times
the recipe's execution rows per request. KV pages receive the remaining budget,
within their virtual maximum. The planner accounts for mapping alignment and
requires at least 8,192 cached tokens per active request.

Up to eight immutable snapshot slots per active request share physical backing
with KV pages. At load, CUDA virtual mappings expose the same allocation chunks
through the page planes and the snapshot part of the continuation range. Small
interleaved runs of dense page planes keep each snapshot's conflicting page range
close to its byte cost. Planes whose allocation granules span more pages than a
row's byte cost remain dedicated to KV; the densest planes always participate.
Across ranks, the scheduler uses the union of conflicting ranges.
Ranks loaded in separate processes must report identical cache capacities and
snapshot exclusions during the control-channel handshake; a mismatch rejects
startup before replay. Before cache allocation, hosts agree on the smallest row
and page capacities that all ranks can back.
An owned snapshot excludes all conflicting page ids; overlapping exclusions
are reference-counted. A snapshot fitting entirely in execution-allocation
alignment padding excludes no pages. Execution rows never alias KV pages.

Admission reserves the request's KV pages. System, prompt, and terminal snapshots
acquire backing at their capture boundaries, evicting unpinned checkpoints when
needed. If active requests or transfers hold all suitable backing, optional
retention is skipped. An inbound host restore reserves both its row and pages
before transfer; if its snapshot prevents admission, the request recomputes its
prompt. Snapshot and page ownership remain exclusive through graph
and transfer completion, including cancellation and duplicate publication.
Once generation drains, unused output-page reservations are released before
retaining the final checkpoint.

The loader writes the chosen physical row and page counts into `memory_capacity`.
Maintenance kernels consume those device scalars; kernels preserve fixed row
strides and base addresses. Request-to-row bindings separate transport slots
from execution rows. Virtual addresses and all physical mappings are fixed at
load; changing ownership requires no CUDA allocation or remapping. Model recipes declare row
contents and KV geometry; the memory planner does not depend on recurrent or
attention architecture.

The prepared context extent determines page-table shapes and captured workspace
sizes. The runtime sequence limit is the smaller of that extent and loaded KV
capacity after the model's per-request draft-page allowance. Runtime input packing
preserves the prepared strides. Existing packages retain their prepared ceiling;
raising that ceiling requires export.

The `flare_snapshot_backing_pages` metric counts page ids excluded by owned
snapshots, including rows awaiting retirement. It counts overlapping exclusions
once. `flare_free_snapshots` counts unowned virtual slots; their backing may be
occupied by KV pages.

The loading display separates allocation capacity from payload bytes.
`Allocate model memory` reports GPU and host capacity. `Load weights` counts
only bytes present in allocation payload spans, including initialized model
metadata, and excludes disposable storage and elided zero blocks.

The loader reads only the bounded index before selection. It decodes selected
graph bodies one record at a time, retains dependency bookkeeping for one
graph, and destroys each CUDA graph template after instantiation and upload.
Cubins are mapped from their file spans while loading; no anonymous copy of
the full kernel catalog is required. Frontend assets are read from their own
file spans. The index and loader metadata are released after startup.

Cooperative kernels use resident worker grids and must tolerate a reduction in
their largest grid dimension, whether X, Y, or Z. If restoring the cooperative
launch attribute returns `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`, the loader
queries the kernel's maximum active blocks per SM for its captured block
dimensions and dynamic shared memory.
It multiplies that occupancy by the target device's SM count, reduces the largest
grid dimension to fit while preserving the other two, and retries the attribute
once. A grid that cannot fit by reducing one dimension rejects loading. Fitting
cooperative grids and ordinary launches retain their captured dimensions without
occupancy queries.

Payload loading opens the package with `O_DIRECT`, bypassing the page cache.
Each rank uses four file-reader threads and four pinned staging buffers of
`4 MiB / rank_count`. Total staging is 16 MiB plus at most one 4 KiB alignment
block per buffer. Reads cover complete 4 KiB blocks; only declared payload
bytes are copied to their destinations. A partial final file block is accepted
when it contains the entire requested payload; missing payload bytes fail loading.
Each rank uploads device chunks asynchronously on its own CUDA stream, with
completion events guarding buffer reuse. Host-resident payloads are copied
from staging into their final mapped allocations. Readers join and uploads
drain before relocations, module loading, or graph execution.

The serving process contains no Python interpreter. Loader tests inspect
`/proc/self/maps` after reconstruction. Startup measurements separate context
creation, address reservation, payload transfer, module loading, graph upload,
and cache allocation.

## 5. The Flare command

Flare resolves its argument to a package, calls the linked Helios library,
supplies the bind address and API key, handles signals, and reports readiness.
Flare owns the terminal banner. Helios supplies loading stages with measured
progress, cycling bars for indeterminate work, results, and elapsed time.
Helios validates the package before reconstruction:
the package's GPU architecture matches the selected device, the recorded
address range is free, free device memory covers the recorded footprint, and
the package's execution contract version is one the Helios build
understands. It does not pin a driver version or a Helios build, because sm_120
SASS loads on any driver that supports the architecture and a package stays
valid across Helios builds that keep the contract.

Flare has no recipe catalog, no planner, no hub access, and no kernel cache.
Those belong to ignition.

## 6. Tensor-parallel ranks as threads

Tensor-parallel serving uses one Rust process with one CUDA execution thread
per rank. Rank
zero's thread also runs the scheduler. In-process channels carry prepared
steps to peer threads; every rank validates the same fence and uploads before
commit. Completion records must agree on request epochs, token consumption,
and status. A 120-second deadline bounds a stranded collective. Shutdown
confirms drain on every rank before releasing any shared storage.

Preparation uses one Python process per rank. Each process reserves every
rank's 1 TiB arena; rank r owns device base `0x0200_0000_0000 + r * 2^40`
and host base `0x0c00_0000_0000 + r * 2^40`. b12x collective allocations
use the native package allocator. POSIX descriptor passing shares VMM physical
allocations, and importers map each allocation at its owner's recorded address.
Collective signals and generation-tagged payloads are reset after all ranks
drain. Imported mappings are recorded as peer dependencies and stored only in
the owning rank's bundle.

Helios checks that rank allocations do not overlap, creates every device
context, loads rank storage, and grants the declared peer access through
`cuMemSetAccess` before loading modules and instantiating execution state.
Address reservations are established before allocation. Allocation, payload
loading, kernel loading, and execution preparation each run concurrently across
ranks, joining all rank threads between stages. Loading retains a process-wide
16 MiB pinned staging bound and displays aggregate progress for each stage.
Serving uses ordinary shared pointers; no IPC import,
Python process, or socket protocol is involved. Each model and TP size requires
independent numerical and serving validation. The
[rank-thread design](same-process-tp.md) specifies coordination and failure
handling.

## 7. Process images

A CRIU image taken with the driver's process checkpoint support restores a
loaded Helios process bit-exactly: every state class a prepared model holds
was verified, including tensor maps, module symbols, pinned and registered
host memory, two devices with peer access in one process, and legacy IPC
across a process tree launched inside a cuda-checkpoint job. Images are not
the production path. Restore moves the same bytes a package load moves, once
into anonymous host pages and once to the device, so it is bandwidth-bound in
the same way; measured at a 70 GB footprint, restore from an NVMe image took
29 s and from RAM about 13 s. An image is valid only on the driver build that
produced it, bakes in the Helios binary, needs root and CRIU on the host,
holds the whole device footprint in host memory during dump and restore, and
is corrupted if any thread touches GPU-side driver state during the
checkpoint window, including a thread polling `cuEventQuery`.

Images remain a validation tool: image a loaded Helios once, and after a
kernel, driver, or loader change compare a fresh load's replay against the
restored image. The probes, orchestration, and measurements live under
`~/projects/flare-research`.

## 8. Open items

- Full-model relocation requires complete pointer and tensor-map bindings in
  parameters, persistent storage, and module globals. The loader accepts typed
  relocations, but the model exporter preserves addresses for opaque resources.
- The sm_120 TMA tile path links against driver-provided code and reserves
  about 3.6 GiB of device memory per context when loaded; the memory plan
  should account for it once b12x kernels that use the path are identified.
