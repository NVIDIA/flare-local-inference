"""Bootstrap coordination of b12x collectives over an established Gloo group."""

import json
from datetime import timedelta
from pathlib import Path

import torch.distributed as dist


def host_identity():
    """This boot of this host; ranks sharing it can map each other's memory through CUDA IPC."""
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def create_roce_runtime(group, device, *, max_tokens, pipe_tokens, hidden_size, gather_bytes, package_storage=None,
                        varlen_bytes=0, ring=False, ring_open=False):
    """Allocate RoCEnante for ranks on separate DGX Sparks: BF16 hidden reductions and logit gathers.

    One runtime serves every reduction and gather in rank-serial order. Reductions of up
    to `max_tokens` rows, and at least the two-shot threshold, run one-shot: every rank
    sends its whole payload to every peer, which is fastest for decode rows. Longer
    reductions, up to `pipe_tokens` rows, run as a pipelined two-shot reduce-scatter and
    all-gather, which moves a ring's bytes for prefill chunks.

    With `package_storage`, the pinned region is package host storage at a recorded address and
    the runtime is registered for export, so a native rank can rebuild the transport over it.
    `varlen_bytes` sizes each rank's contribution to variable-length gathers; zero prepares none.
    """
    from b12x.comm import roce

    if any(type(value) is not int or value <= 0 for value in (max_tokens, pipe_tokens, hidden_size, gather_bytes)):
        raise ValueError("RoCE capacities must be positive integers")
    # Below this size the runtime always reduces one-shot (b12x's two-shot threshold is 1 or 2 MiB).
    nbytes = max(max_tokens * hidden_size * 2, 2 << 20)
    pipe_bytes = pipe_tokens * hidden_size * 2
    if nbytes % 16 or pipe_bytes % 16 or gather_bytes % 16:
        raise ValueError("RoCE capacities must occupy complete 16-byte packs")
    import torch

    region = None if package_storage is None else (
        lambda size: package_storage.host_allocate("roce", (size,), torch.uint8).host_view)
    if type(varlen_bytes) is not int or varlen_bytes < 0 or varlen_bytes % 16:
        raise ValueError("variable-length gather capacity must be a nonnegative multiple of 16 bytes")
    runtime = roce.AllReduce(exchange_group=group, device=device, max_size=nbytes, max_gather_bytes=gather_bytes,
                             pipe_bytes=pipe_bytes, region_allocator=region,
                             ring=ring, **({"ring_open": True} if ring_open else {}),
                             **({"max_varlen_bytes": varlen_bytes} if varlen_bytes else {}))
    if package_storage is not None:
        package_storage.transports.append(runtime)
    return runtime

def create_all_reduce_runtime(group, device, *, max_tokens, hidden_size, ipc=None, element_bytes=2):
    """Allocate one b12x IPC channel for the largest hidden activation.

    Every rank supplies identical capacities before preparing graph variants.
    The bootstrap retains the runtime until every native rank worker drains.
    """
    from b12x.comm.pcie import AllReduce

    if any(type(value) is not int or value <= 0 for value in (max_tokens, hidden_size)):
        raise ValueError("all-reduce capacities must be positive integers")
    if element_bytes not in (2, 4):
        raise ValueError("all-reduce storage must use two- or four-byte elements")
    nbytes = max_tokens * hidden_size * element_bytes
    if nbytes % 16:
        raise ValueError("all-reduce capacity must occupy complete 16-byte packs")
    return AllReduce.from_exchange_group(exchange_group=group, device=device, single_channel=True,
                                         max_size=nbytes, eager_buffer_bytes=nbytes, ipc=ipc)


def create_all_gather_runtime(group, device, *, max_rows, columns, ipc=None):
    """Allocate b12x staging for serial BF16 column gathers in prepared graphs."""
    from b12x.comm.pcie import DcpAllToAll

    world_size = dist.get_world_size(group)
    if (world_size < 2 or type(max_rows) is not int or max_rows <= 0
            or type(columns) is not int or columns <= 0 or columns % (world_size * 64)):
        raise ValueError("column gathers require positive rows and 64-column-aligned rank partitions")
    return DcpAllToAll.from_exchange_group(exchange_group=group, device=device,
        max_batch_size=max_rows, total_heads=columns // 64, head_dim=64, stream_affine=False, ipc=ipc)


def create_result_broadcast_runtime(group, device, *, layouts, ipc=None):
    """Allocate one local PCIe channel covering every declared result payload.

    Layouts describe captured operation fields, independently of request-slot storage.
    RoCE callers retain their existing variable-payload transport.
    """
    from ignition.components.collectives import packed_broadcast_layout

    sizes = [packed_broadcast_layout(layout.values())[1] for layout in layouts]
    if not sizes or min(sizes) <= 0:
        raise ValueError("result broadcasts require nonempty payload layouts")
    # Gather warps copy 512-byte segments. Use one segment per rank per row
    # so short results transfer fewer rows while large results keep every
    # copying lane active. All payload sizes share the same channel storage.
    max_rows = (max(sizes) + 511) // 512
    return create_all_gather_runtime(group, device, max_rows=max_rows,
        columns=256 * dist.get_world_size(group), ipc=ipc)


def create_vocabulary_argmax_runtime(group, device, *, local_vocab_size, max_rows, ipc=None):
    """Allocate b12x exchange slabs for exact greedy tokens over vocabulary shards (TP8, TP12 or TP16)."""
    from b12x.comm.pcie.pcie_vocab_argmax import PCIeVocabParallelArgmax

    return PCIeVocabParallelArgmax.from_exchange_group(exchange_group=group, device=device,
        local_vocab_size=local_vocab_size, max_batch_size=max_rows, ipc=ipc)


def create_dma_all_reduce_runtime(group, device, *, max_tokens, hidden_size, ipc=None, element_bytes=2):
    """Allocate a b12x copy-engine ring for prefill reductions in caller-owned shared storage."""
    from b12x.comm.pcie import DmaAllReduce

    if any(type(value) is not int or value <= 0 for value in (max_tokens, hidden_size)):
        raise ValueError("all-reduce capacities must be positive integers")
    if element_bytes not in (2, 4):
        raise ValueError("all-reduce storage must use two- or four-byte elements")
    return DmaAllReduce(exchange_group=group, device=device, max_bytes=max_tokens * hidden_size * element_bytes, ipc=ipc)


def create_head_exchange(group, device, *, max_rows, heads, head_dim, query_head_dim=None, ipc=None):
    """Allocate b12x staging for all-gathering head shards and LSE-combining partial attention.

    Queries may be wider than the combined values: latent attention gathers absorbed queries of
    `query_head_dim` and combines `head_dim` outputs.
    """
    from b12x.comm.pcie import DcpAllToAll

    world_size = dist.get_world_size(group)
    query_head_dim = head_dim if query_head_dim is None else query_head_dim
    if world_size < 2 or max_rows <= 0 or heads % world_size or head_dim % 8 or query_head_dim % 8:
        raise ValueError("head exchanges require multiple equal head shards and positive capacity")
    return DcpAllToAll.from_exchange_group(exchange_group=group, device=device, max_batch_size=max_rows,
        total_heads=heads, head_dim=head_dim, query_head_dim=query_head_dim, stream_affine=False, ipc=ipc)


def _winners(contributions):
    """Each sharded race's fastest complete contribution, ties going to the lower candidate index."""
    races = {}
    for item in contributions:
        best = races.get(item.key)
        if item.assignment is not None and (
                best is None or (item.latency_us, item.candidate_index) < (best.latency_us, best.candidate_index)):
            best = item
        races[item.key] = best
    if missing := [key for key, item in races.items() if item is None]:
        raise RuntimeError(f"no rank completed a candidate of tuning race {missing[0]!r}")
    return tuple(races[key] for key in sorted(races))


class CollectivePreparation:
    """Authorize matching collective declarations and synchronize kernel entry.

    Local compilation may complete at different times on each rank. This
    coordinator releases a collective only after all group members report the
    same declaration. When the session shards tuning, it also relays b12x's
    two exchanges: every rank's cached selections, which b12x reconciles in
    rank order, and each race's fastest result among the ranks' disjoint
    candidate shares. It is unused after graph handoff.
    """

    def __init__(self, group, *, timeout=timedelta(minutes=2)):
        if dist.get_backend(group) != "gloo":
            raise ValueError("collective preparation requires a Gloo control group")
        self.group = group
        self.ranks = tuple(dist.get_process_group_ranks(group))
        self.timeout = timeout
        hosts = [None] * len(self.ranks)
        dist.all_gather_object(hosts, host_identity(), group=group)
        # Each rank's host, in group order; ranks on one host share its CPUs and CUDA IPC.
        self.hosts = tuple(hosts)
        self.local_ranks = self.hosts.count(host_identity())

    @property
    def spans_hosts(self):
        return len(set(self.hosts)) > 1

    def same_on_every_rank(self, value):
        values = [None] * len(self.ranks)
        dist.all_gather_object(values, value, group=self.group)
        return all(item == values[0] for item in values)

    def share_selections(self, cache):
        """Adopt rank zero's cached kernel selections on every rank.

        Ranks on other hosts keep their own tuning caches, and independent races choose
        different kernels for about a quarter of GLM's requests. Compiled forwards bind rank
        zero's scratch layouts, so every rank must run the same selections.
        """
        records = [cache.records if dist.get_rank() == self.ranks[0] else None]
        dist.broadcast_object_list(records, src=self.ranks[0], group=self.group)
        if dist.get_rank() != self.ranks[0]:
            # The session's agreed records, which b12x's own cross-rank reconciliation sets, take
            # precedence over the rank's file. Saving a newly tuned key reloads that file, so
            # replacing only the loaded records would restore the rank's own choices.
            cache._agreed_records = dict(records[0])

    def agree_selections(self, selections):
        """Fail before compilation when ranks selected different kernels for the same request."""
        local = {item["name"]: (item["component"], json.dumps(item["config"], sort_keys=True))
                 for item in selections}
        gathered = [None] * len(self.ranks)
        dist.all_gather_object(gathered, local, group=self.group)
        root = gathered[0]
        differing = sorted(name for other in gathered[1:] for name, value in other.items()
                           if name in root and root[name] != value)
        if differing:
            raise RuntimeError(f"{len(differing)} kernel selections differ from rank zero's, such as "
                               f"{differing[:3]}; preparing again adopts rank zero's tuning decisions")

    def __call__(self, progress):
        if progress.ready_cache is not None or progress.ready_tuning:
            raise ValueError("collective preparation does not shard tuning candidates")
        pending = tuple((item.key, item.ranks) for item in progress.ready_collectives)
        states = [None] * len(self.ranks)
        dist.all_gather_object(states, pending, group=self.group)
        if not all(states):
            return None
        if any(state != states[0] for state in states[1:]):
            raise RuntimeError(f"collective preparation declarations differ across ranks: {states}")
        key, ranks = states[0][0]
        if tuple(ranks) != self.ranks:
            raise ValueError("collective preparation requires the complete control group")
        return key

    def prepare(self, session, requests, *, progress=None, autotune=None):
        """Drive a mixed local/collective job until every rank has finished.

        A finished rank must still exchange readiness while its peers compile
        or finish local work. Returning on local completion can strand peers in
        their next control-group exchange.
        """
        job = None
        state = None
        error = None
        authorization = {"collective_key": None}
        try:
            while True:
                try:
                    if job is None:
                        job = session.begin(requests, autotune=autotune)
                    if state is None or not state.done:
                        state = job.advance(**authorization)
                        if progress is not None:
                            progress(state)
                except Exception as exc:  # noqa: BLE001 - Every rank must receive preparation failures.
                    error = exc
                pending = () if state is None else tuple((item.key, item.ranks) for item in state.ready_collectives)
                local = (state is not None and state.done, pending,
                         None if error is None else f"{type(error).__name__}: {error}",
                         None if state is None else state.ready_cache, () if state is None else tuple(state.ready_tuning))
                states = [None] * len(self.ranks)
                dist.all_gather_object(states, local, group=self.group)
                failures = [(rank, item[2]) for rank, item in zip(self.ranks, states, strict=True) if item[2] is not None]
                if failures:
                    raise RuntimeError(f"rank preparation failed: {failures}") from error
                if all(item[0] for item in states):
                    return job.result()
                if any(item[0] for item in states) and any(item[1] or item[3] is not None or item[4] for item in states):
                    raise RuntimeError("a rank finished before its peer's collective declaration")
                authorization = {"collective_key": None}
                if all(item[3] is not None for item in states):
                    # Each rank reconciles every rank's cached selections, rank zero's first.
                    authorization = {"cache": tuple(item[3] for item in states)}
                    continue
                if all(item[4] for item in states):
                    races = [sorted(contribution.key for contribution in item[4]) for item in states]
                    if any(keys != races[0] for keys in races[1:]):
                        reference = set(races[0])
                        counts = dict(zip(self.ranks, map(len, races), strict=True))
                        differences = {}
                        for rank, keys in zip(self.ranks[1:], races[1:], strict=True):
                            missing, extra = reference - set(keys), set(keys) - reference
                            if missing or extra:
                                differences[rank] = {"missing": len(missing), "extra": len(extra),
                                                     "examples": sorted(missing | extra)[:2]}
                        raise RuntimeError(f"tuning races differ across ranks: counts={counts}, differences={differences}")
                    winners = _winners(contribution for item in states for contribution in item[4])
                    authorization = {"tuning": winners[0] if len(winners) == 1 else winners}
                    continue
                ready = [item[1] for item in states]
                if all(ready):
                    if any(item != ready[0] for item in ready[1:]):
                        raise RuntimeError(f"collective preparation declarations differ across ranks: {ready}")
                    key, ranks = ready[0][0]
                    if tuple(ranks) != self.ranks:
                        raise ValueError("collective preparation requires the complete control group")
                    authorization = {"collective_key": key}
        except BaseException as failure:
            try:
                if job is not None:
                    if state is not None and state.done:
                        job.result().close()
                    else:
                        job.close()
            except BaseException as cleanup:  # noqa: BLE001 - Preserve the distributed failure.
                failure.add_note(f"preparation cleanup failed: {cleanup!r}")
            raise

    def barrier(self, key, ranks):
        if tuple(ranks) != self.ranks:
            raise ValueError(f"collective {key!r} does not span its control group")
        dist.monitored_barrier(group=self.group, timeout=self.timeout, wait_all_ranks=True)
