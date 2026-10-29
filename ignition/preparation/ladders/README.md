# Offline collective ladders

`--collective-tuning runtime --collective-ladder PATH` selects a joint PCIe
kernel's compiled geometry from a JSON file. Preparation compiles that geometry
without racing alternatives, regardless of `--autotune-effort`. Captured tuning
graphs select among the file's submode and active-CTA profiles when the package
loads. Serving replays the same compiled kernel with the selected profile.

From the repository root, add these options to `ignition export`:

```sh
--collective-tuning runtime \
--collective-ladder ignition/preparation/ladders/gb202_pcie_tp4.json
```

The option is required whenever the resolved collective policy is `runtime`,
including configuration-file and recipe defaults. Static policy rejects the
option. File contents travel with the preparation plan and package rebuild
settings; workers do not reopen the path.

Startup interleaves 21 timing samples per profile across the required reduction
groups. Each sample measures a complete chain of reductions and uses the slowest
rank's duration. Selection minimizes their sum, so stalls contribute to the
decision; exact ties follow the profile order in the file. Every profile must
pass the arithmetic checks before serving can start.

The groups share one captured startup graph. Reducing serving concurrency or
draft width at load time retains this graph and its workspace whenever a selected
serving graph uses a runtime collective. A package containing only one reduction
group has no other groups to separate its samples in time.

## Version 1 format

```json
{
  "version": 1,
  "name": "example_tp4",
  "compute_capability": [12, 0],
  "world_size": 4,
  "dtype": "bfloat16",
  "inplace": false,
  "cache_policy": "cold",
  "min_bytes": 1024,
  "buckets": [
    {"max_bytes": 1048576, "implementation": "joint_bandwidth", "threads": 512, "blocks": 64,
     "profiles": {"bulk_push": [32], "bulk_pull": [8]}}
  ]
}
```

Payload bytes count the full per-rank input before reduction. Bounds must be
16-byte aligned. The first bucket includes `min_bytes` through `max_bytes`;
subsequent buckets include payloads above the preceding maximum through their
own maximum. Selection does not extrapolate outside those bounds.

Each bucket declares one kernel family, thread count, and physical CTA grid.
`joint_latency` permits `full`, `payload_rsag`, and `pull`; `joint_bandwidth`
permits `payload_rsag`, `bulk_push`, `bulk_pull`, and `pull`. Each profile lists
active CTA counts between one and the physical grid size. Several counts may
belong to the same submode. With more than two ranks, both families also permit
`serial_push` and `queued_push`, which send payloads to one peer at a time.
`serial_push` requires exactly one active CTA; `queued_push` distributes tiles
among active CTAs without requiring the entire grid to be resident.
`cache_policy` controls the captured profile race's
cache treatment and accepts `cold` or `warm`.

The ladder applies to eligible plain BF16 PCIe reductions. TP size, compute
capability, and input/output aliasing must match. Other collective transports
and fused operations retain their preparation policy. Runtime profile selection
cannot change the compiled family, threads, or physical grid.

## GB202 PCIe TP4 recommendations

`gb202_pcie_tp4.json` contains 210 buckets from 16 bytes through 8 MiB.
The 16–1008-byte bucket uses one active CTA of the 64-thread latency kernel
in `full` mode for readiness barriers and small validation reductions; this
profile has correctness coverage but no measured performance ranking. The
remaining 209 buckets cover 1 KiB through 8 MiB at 16 intervals per doubling.
The ladder targets out-of-place BF16 reductions on four SM120 GPUs. The
measurements cover a same-switch group (GPUs 0,1,2,3) and a
separate-switch group (GPUs 0,5,7,9) on Pulsar, plus same-switch GPUs 12,13,14,15
and host-bridge GPUs 0,4,8,12 on Frank2. Frank2's same-switch group is classified
by its physical wiring; its OS topology report does not describe that wiring
correctly. Timing protocols remain separate measurement cohorts, and relative
latencies are normalized within each report.

Buckets with upper bounds from 32 through 64 KiB use `joint_latency`, 256 threads,
and a physical grid of 64 CTAs. Their runtime profiles include `payload_rsag`
for switched paths and `pull` for host-bridge paths. Other buckets use the
combined reports' switched-first geometry choices: keep both switched cohorts
within 2% of their respective best where possible, otherwise use the smallest
shared penalty, then minimize the worst host-bridge relative latency.

Each bucket contains at most four runtime profiles. Menus cover every cohort
within 1% of that geometry's best profile when four profiles suffice; otherwise
the tolerance is 5%. At the measured 32, 41.5, and 64 KiB crossover points, menus
also retain an exact best profile from the Pulsar switched sweep using
partition-boundary comparisons for payload ownership. The 5% rule concerns
profiles within a fixed geometry, not the cost of sharing one geometry across
topologies.

With comparison-based ownership in the 256-thread latency kernel, the Pulsar
switched penalties against the best geometry using that ownership calculation
are 2.1%, 0.7%, and 3.1% at those three crossover points. These measurements
require that kernel implementation; the JSON file only selects geometry and
runtime profiles. Frank2 measurements use division-based ownership. The combined
menus have not had independent replay verification, and bucket interiors and
unmeasured topologies have no measured performance bound.
