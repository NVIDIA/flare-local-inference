"""Attribute physical export storage to live prepared owners without counting aliases twice."""

from collections import defaultdict, deque
from collections.abc import Mapping
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache
import heapq
import inspect
from pathlib import Path
import re
from types import FunctionType, MethodType, ModuleType

from .tensor_metadata import merge_ranges, subtract_ranges, tensor_ranges


@dataclass(frozen=True, slots=True)
class Claim:
    start: int
    end: int
    category: str
    owner: str
    source: str = ""
    priority: int = 0
    tensor: bool = False


@lru_cache(maxsize=None)
def _source(cls):
    try:
        path = Path(inspect.getfile(cls)).resolve()
        line = inspect.getsourcelines(cls)[1]
    except (TypeError, OSError):
        return ""
    root = Path(__file__).resolve().parents[2]
    try:
        path = path.relative_to(root)
    except ValueError:
        pass
    return f"{path}:{line}"


def owner_claims(prepared):
    """Walk retained fields and function closures, never properties or module globals.

    Tensor claims cover elements rather than entire backing storages. Pool claims
    cover their full allocations, including unused chunk capacity. Object identity
    bounds traversal; overlapping tensor elements use their first owning field.
    """
    import torch
    from .aot import BoundForward
    from .scratch import GraphBufferPool, ScratchArena

    seen, covered, result = set(), [], []
    pending = deque()

    def enqueue(entries):
        for value, path, source in entries:
            if id(value) in seen or value is None or isinstance(value, (str, bytes, int, float, bool, type, ModuleType)):
                continue
            seen.add(id(value))
            pending.append((value, path, source))

    enqueue((value, f"resources[{index}]", "") for index, value in enumerate(prepared.resources))
    enqueue(((prepared.buffers, "buffers", ""),))
    while pending:
        value, path, source = pending.popleft()
        if isinstance(value, BoundForward):
            # Runtime operands belong to the bound owner; compiler IR and guards describe them again.
            enqueue(((value.owner, f"{path}.owner", source),))
            continue
        if isinstance(value, torch.Tensor):
            if (value.device.type == "cuda" and value.device.index == prepared.device
                    and value.untyped_storage().device.type != "meta"):
                if not value.numel():
                    continue
                start = value.data_ptr()
                end = start + value.element_size() * (1 + sum(
                    (size - 1) * stride for size, stride in zip(value.shape, value.stride())))
                previous = bisect_right(covered, start, key=lambda span: span[0]) - 1
                if previous >= 0 and covered[previous][1] >= end:
                    continue
                for start, end in tensor_ranges(value):
                    first = max(0, bisect_left(covered, start, key=lambda span: span[0]) - 1)
                    stop = bisect_right(covered, end, key=lambda span: span[0])
                    overlaps = covered[first:stop]
                    result.extend(Claim(left, right, "", path, source, tensor=True)
                                  for left, right in subtract_ranges([(start, end)], overlaps))
                    covered[first:stop] = merge_ranges([*overlaps, (start, end)])
            continue
        if isinstance(value, (ScratchArena, GraphBufferPool)):
            category = "Operator scratch" if isinstance(value, ScratchArena) else "Graph scratch pools"
            tensors = [value.tensor] if isinstance(value, ScratchArena) else [
                chunk for chunks in value.chunks.values() for chunk in chunks]
            result.extend(Claim(t.data_ptr(), t.data_ptr() + t.nbytes, category,
                                f"{path}<{type(value).__name__}>", _source(type(value)), 80)
                          for t in tensors if t.nbytes)
            # Chunk tensors would hide the fields using their individual views.
            continue
        if isinstance(value, Mapping):
            enqueue((child, f"{path}[{key!r}]", source) for key, child in value.items())
        elif isinstance(value, (tuple, list)):
            enqueue((child, f"{path}[{index}]", source) for index, child in enumerate(value))
        elif isinstance(value, MethodType):
            enqueue(((value.__self__, path, source),))
        elif isinstance(value, FunctionType):
            source = f"{value.__code__.co_filename}:{value.__code__.co_firstlineno}"
            enqueue((child, f"{path}.defaults[{index}]", source)
                    for index, child in enumerate(value.__defaults__ or ()))
            enqueue(((value.__kwdefaults__, f"{path}.defaults", source),))
            for name, cell in zip(value.__code__.co_freevars, value.__closure__ or ()):
                try:
                    child = cell.cell_contents
                except ValueError:
                    continue
                enqueue(((child, f"{path}.{name}", source),))
        else:
            cls = type(value)
            source = _source(cls) or source
            path = f"{path}<{cls.__name__}>"
            if is_dataclass(value):
                children = ((f.name, getattr(value, f.name)) for f in fields(value))
            else:
                children = vars(value).items() if hasattr(value, "__dict__") else ()
            enqueue((child, f"{path}.{name}", source) for name, child in children)
    return result


def attribute(ranges, claims):
    """Partition a union of physical ranges using categorical and tensor ownership claims."""
    ranges = merge_ranges(ranges)
    events = []
    for index, claim in enumerate(claims):
        if claim.end > claim.start:
            events.extend(((claim.start, 1, index), (claim.end, -1, index)))
    # Range markers let one sweep clip virtual tensor extents to physical backing.
    for start, end in ranges:
        events.extend(((start, 2, -1), (end, -2, -1)))
    events.sort()
    active, categories, owners = set(), [], []
    inside, previous, rows = 0, None, defaultdict(int)
    for address, kind, index in events:
        if inside and previous is not None and address > previous:
            for heap in (categories, owners):
                while heap and heap[0][-1] not in active:
                    heapq.heappop(heap)
            classification = claims[categories[0][-1]] if categories else None
            owner = claims[owners[0][-1]] if owners else classification
            category = classification.category if classification else (
                "Other owned tensors" if owner else "Unknown storage")
            # Allocator holes and metadata describe storage, not adjacent tensor owners.
            if classification and classification.priority >= 100:
                owner = classification
            rows[category, owner.owner if owner else "unattributed", owner.source if owner else ""] += address - previous
        if abs(kind) == 2:
            inside += kind // 2
        elif kind == 1:
            active.add(index)
            claim = claims[index]
            if claim.category:
                heapq.heappush(categories, (-claim.priority, index))
            if claim.tensor:
                heapq.heappush(owners, (index,))
        else:
            active.remove(index)
        previous = address
    return [dict(category=category, owner=owner, source=source, nbytes=size)
            for (category, owner, source), size in sorted(rows.items())]


def _size(ranges):
    return sum(end - start for start, end in merge_ranges(ranges))


def collect_inventory(prepared, storage, package, *, external=()):
    """Join Heliopause's physical snapshot to the graph-reachable package and its owners.

    This is a preparation snapshot, not a prediction of serving cache size or
    driver overhead after loading. All byte counts exclude imported peer mappings
    and mapped host storage. Dynamic caches count only their backed prefixes.
    """
    import flare_heliopause
    import torch

    native = flare_heliopause.memory_inventory()
    claims = owner_claims(prepared)
    claims.extend(Claim(start, end, "Weights / numeric constants", "checkpoint tensors", priority=30)
                  for start, end in storage.numeric_storage.ranges())
    claims.extend(Claim(start, start + size, "Disposable storage", "export_zero_ranges", priority=10)
                  for start, size in prepared.export_zero_ranges)
    for name, tensor in prepared.buffers.items():
        claims.extend(Claim(start, end, "Graph I/O / persistent state", f"buffers[{name!r}]", priority=40)
                      for start, end in tensor_ranges(tensor))
    segments = [s for s in torch.cuda.memory_snapshot() if s["device"] == storage.device]
    for segment in segments:
        graph_pool = tuple(segment["segment_pool_id"]) == tuple(storage.graph_pool.id)
        for block in segment["blocks"]:
            start, size = block["address"], block["size"]
            if graph_pool:
                claims.append(Claim(start, start + size, "Graph capture pool", "shared CUDA graph pool",
                                    "ignition/preparation/capture.py", 70))
                # Capture pools can retain graph-live blocks marked inactive by PyTorch.
                continue
            live = 0 if block["state"] == "inactive" else block["requested_size"]
            if live < size:
                claims.append(Claim(start + live, start + size, "Allocator slack", "inactive blocks / alignment", priority=100))
    dynamic = {region.address: region for region in storage.dynamic_regions}
    for region in dynamic.values():
        claims.append(Claim(region.address, region.address + region.nbytes,
                            "Preparation cache backing", region.name,
                            "ignition/preparation/package_storage.py:PackageStorage.device_allocate", 110))
    peers = {} if storage.peer_storage is None else storage.peer_storage.owned
    for address, size in peers.items():
        if address not in dynamic:
            claims.append(Claim(address, address + size, "TP communication", "PeerStorage.owned",
                                "ignition/preparation/peer_storage.py", 110))
    for address, size, owner in package.module_globals:
        claims.append(Claim(address, address + size, "Kernel globals", owner, priority=110))
    fixed = [(a.address, a.address + a.nbytes) for a in package.allocations
             if a.memory == "device" and a.sizing is None]
    cache = [(r.address, r.address + r.nbytes) for r in dynamic.values()]
    globals = [(address, address + size) for address, size, _ in package.module_globals]
    physical = merge_ranges([
        *((address, address + size) for address, size in native["mappings"]),
        *((address, address + size) for address, size in external),
        *((s["address"], s["address"] + s["total_size"]) for s in segments),
        *cache, *globals,
    ])
    exported = merge_ranges([*fixed, *globals])
    package_padding = subtract_ranges(exported, physical)
    retained = subtract_ranges(exported, package_padding)
    preparation_only = subtract_ranges(physical, retained)
    used = native["total_bytes"] - native["free_bytes"]
    known = _size(physical)
    return dict(rank=storage.rank, device=storage.device,
                total_bytes=native["total_bytes"], free_bytes=native["free_bytes"],
                device_used_bytes=used, tracked_physical_bytes=known,
                device_unattributed_bytes=used - known,
                torch_allocated_bytes=torch.cuda.memory_allocated(storage.device),
                torch_reserved_bytes=torch.cuda.memory_reserved(storage.device),
                fixed_package_bytes=_size(fixed), kernel_global_bytes=_size(globals),
                package_padding_bytes=_size(package_padding),
                cache_backing_bytes=_size(cache),
                mapped_host_bytes=sum(r.extent for r in storage.host_regions if not r.closed),
                imported_peer_bytes=(0 if storage.peer_storage is None else sum(
                    size for parts in storage.peer_storage.imported_mappings.values() for _, size in parts)),
                entries=attribute(retained, claims), preparation_entries=attribute(preparation_only, claims),
                graphs=len(prepared.graphs), **package.export_counts)


def format_inventory(reports):
    """Format rank-local accounting and the largest source groups in each category."""
    def display_size(size):
        if abs(size) < 1024:
            return f"{size:,} B"
        if abs(size) < 1 << 20:
            return f"{size / 1024:,.2f} KiB"
        return f"{size / (1 << 20):,.2f} MiB"

    lines = ["VRAM inventory at export (physical bytes; aliased storage counted once)"]
    for report in reports:
        lines.append(f"Rank {report['rank']} / CUDA {report['device']}: "
                     f"{display_size(report['device_used_bytes'])} used, {display_size(report['free_bytes'])} free")
        lines.append(f"  Package fixed mappings: {display_size(report['fixed_package_bytes'])}; "
                     f"kernel globals: {display_size(report['kernel_global_bytes'])}")
        if report['package_padding_bytes']:
            lines.append(f"  Additional mapping alignment at load: {display_size(report['package_padding_bytes'])}")
        for title, field in (("Retained storage", "entries"), ("Preparation-only storage", "preparation_entries")):
            categories = defaultdict(list)
            for row in report[field]:
                categories[row["category"]].append(row)
            lines.append(f"  {title}: {display_size(sum(r['nbytes'] for r in report[field]))}")
            for category, rows in sorted(categories.items(), key=lambda pair: -sum(r['nbytes'] for r in pair[1])):
                lines.append(f"    {category}: {display_size(sum(r['nbytes'] for r in rows))}")
                groups = defaultdict(int)
                for row in rows:
                    # Layer and variant repetitions share a source without losing field names.
                    owner = re.sub(r"(?<!resources)\[\d+\]", "[*]", row['owner'])
                    owner = re.sub(r"(?<=\.)\d+(?=\.)", "*", owner)
                    groups[owner, row['source']] += row['nbytes']
                ordered = sorted(groups.items(), key=lambda pair: -pair[1])
                for (owner, source), size in ordered[:5]:
                    lines.append(f"      {display_size(size)}  {owner}" + (f" — {source}" if source else ""))
                if len(ordered) > 5:
                    lines.append(f"      {display_size(sum(size for _, size in ordered[5:]))}  "
                                 f"in {len(ordered) - 5} other source groups")
        lines.append(f"  PyTorch allocated / reserved: {display_size(report['torch_allocated_bytes'])} / "
                     f"{display_size(report['torch_reserved_bytes'])} (overlaps storage above)")
        lines.append(f"  Cache physically backed for preparation: {display_size(report['cache_backing_bytes'])}; "
                     "serving capacity is sized at load")
        lines.append(f"  Mapped host: {display_size(report['mapped_host_bytes'])}; "
                     f"imported peers: {display_size(report['imported_peer_bytes'])} (excluded from local VRAM)")
        lines.append(f"  Graphs: {report['graphs']}; kernels: {report['kernels']} in {report['modules']} modules "
                     f"({display_size(report['cubin_bytes'])} cubin data, not resident VRAM)")
        lines.append(f"  Device usage minus tracked physical storage: {display_size(report['device_unattributed_bytes'])}")
    lines.append("Graph executable overhead, resident kernel code, untracked driver/library allocations, and other "
                 "processes are included in device usage but cannot be separated by this snapshot.")
    lines.append("Shared scratch lists representative tensor owners; source lines identify owning classes or closures.")
    lines.append("Scratch totals count reserved storage, including chunk padding; they do not measure a peak of live values.")
    return "\n".join(lines)


def print_inventory(reports, *, console=None):
    from rich.console import Console
    import sys

    (console or Console(file=sys.stderr)).print(format_inventory(reports), markup=False, highlight=False)
