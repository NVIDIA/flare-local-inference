# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Export CUDA graph resources with explicit allocation-relative bindings."""

from __future__ import annotations

import ctypes as C
from dataclasses import asdict, dataclass
import os
import json
from pathlib import Path


@dataclass(frozen=True)
class Reference:
    allocation: str
    offset: int = 0


@dataclass(frozen=True)
class Relocation:
    offset: int
    target: Reference
    kind: str = "pointer"


@dataclass(frozen=True)
class Argument:
    """Parameter bytes and explicit bindings needed when allocations relocate."""

    size: int
    relocations: tuple[Relocation, ...] = ()
    numeric: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class FileSource:
    """Immutable tensor bytes appended directly from a checkpoint file."""

    path: Path
    offset: int
    nbytes: int

    def __post_init__(self):
        from .checkpoint_source import source_size

        if self.offset < 0 or self.nbytes <= 0 or self.offset + self.nbytes > source_size(self.path):
            raise ValueError("tensor source must name a nonempty range within its checkpoint file")


@dataclass(frozen=True)
class Allocation:
    name: str
    address: int
    nbytes: int
    alignment: int = 256
    initialize: str = "snapshot"
    relocations: tuple[Relocation, ...] = ()
    preserve_address: bool = False
    memory: str = "device"
    snapshot_spans: tuple[tuple[int, int], ...] | None = None
    sizing: dict | None = None


@dataclass(frozen=True)
class Kernel:
    function: int
    module: str
    arguments: tuple[Argument, ...]


class _Cuda:
    def __init__(self):
        self.lib = C.WinDLL("nvcuda.dll", winmode=0x800) if os.name == "nt" else C.CDLL("libcuda.so.1")

    def call(self, name, *args):
        function = getattr(self.lib, name)
        function.restype = C.c_int
        status = function(*args)
        self.check(name, status)

    def check(self, name, status):
        if status:
            message = C.c_char_p()
            self.lib.cuGetErrorString(C.c_int(status), C.byref(message))
            raise RuntimeError(f"{name}: CUDA {status}: {message.value!r}")



class RuntimePackage:
    """Metadata adapter retaining captured graph and allocation owners during native export."""

    def __init__(self, *, allocations, modules, kernels, owners=(), reservations=(),
                 events=None, transports=(), device_address_floor=0, recorder=None):
        self.allocations = tuple(allocations)
        self.modules, self.kernels = modules, tuple(kernels)
        self.owners, self.reservations = tuple(owners), tuple(reservations)
        self.events, self.transports = dict(events or {}), tuple(transports)
        self.device_address_floor = device_address_floor
        self.recorder = recorder
        self.export_counts = dict(kernels=len(self.kernels), modules=len(self.modules),
            cubin_bytes=sum(len(image) for image in self.modules.values() if isinstance(image, bytes)))

    def write(self, path, graphs, *, execution=None, server=None, progress=None, package_data=None,
              publisher=None, sources=False, rebuild_weights=None, zero_ranges=()):
        import flare_heliopause
        from .package_snapshot import _OPTIONS
        from .rebuild_manifest import rebuild_weights_metadata, weight_payload_request

        if any(isinstance(module, bytes) for module in self.modules.values()):
            raise ValueError("module images must be native recorder handles or cubin file paths")
        context, _ = flare_heliopause.current_context()
        from ignition.components.weights import record_final_fp4_expert_provenance_all
        record_final_fp4_expert_provenance_all()
        # Absent cache configuration stays absent: the native exporter owns
        # default destination and cache selection.
        cache = _OPTIONS.get()
        request = dict(path=str(path), context=context, graphs=_graphs(graphs),
            allocations=[asdict(a) for a in self.allocations], kernels=[asdict(k) for k in self.kernels],
            modules={name: str(path) for name,path in self.modules.items()},
            roots=[a.name for a in self.allocations],
            cupti_path=_cupti_path(), publisher=_publisher_metadata(publisher),
            reservations=[(r["address"], r["nbytes"]) for r in self.reservations], events=self.events,
            transports=self.transports, device_address_floor=self.device_address_floor, execution=execution,
            server=server, package_data=None if package_data is None else package_data.export_metadata(),
            rebuild_weights=None if rebuild_weights is None else rebuild_weights_metadata(rebuild_weights),
            weight_payload=weight_payload_request(rebuild_weights),
            zero_ranges=list(zero_ranges), sources=sources,
            options=_cache_options(cache))
        self.source_path = Path(path)
        self._capture(request, (self, self.owners, graphs), self.recorder)

    def _capture(self, request, owners, recorder):
        import flare_heliopause

        self.native_output = flare_heliopause.export_capture(json.dumps(request), owners, recorder)
        output = json.loads(self.native_output.summary())
        self.allocations = tuple(Allocation(**a) for a in output["allocations"])
        self.module_globals = output["module_globals"]
        self.external = output["external"]
        self.export_counts = output["export_counts"]


def _graphs(graphs):
    return [dict(name=name, graph=int(graph.raw_cuda_graph()), executable=int(graph.raw_cuda_graph_exec()))
            for name, graph in graphs.items()]


def _cache_options(cache, options=None):
    reuse = None if cache is None else cache.reuse_weights
    if reuse is not None:
        reuse = dict(reuse, rebuild_manifest=None if options is None else options.get('rebuild_manifest'))
    return dict(cache=None if cache is None else str(cache.cache), no_weights=False if cache is None else cache.no_weights,
                reuse_weights=reuse)


def _cupti_path():
    from .module_images import cupti_library
    return str(cupti_library())


def _publisher_metadata(publisher):
    return None if publisher is None else publisher.native_request


def export_model(prepared, images, storage, path, *, options, progress=None, publisher=None, bundle_only=False,
                 sources=False):
    """Hand captured handles and preparation metadata to the native exporter."""
    import flare_heliopause
    from .tensor_metadata import live_metadata
    from .package_snapshot import _OPTIONS
    from .rebuild_manifest import rebuild_weights_metadata, weight_payload_request
    from .memory_inventory import collect_inventory, print_inventory

    if storage.error is not None:
        raise RuntimeError("native package allocator failed") from storage.error
    if prepared._consumed:
        raise RuntimeError("runtime package export requires an unused prepared model")
    context, _ = flare_heliopause.current_context()
    flare_heliopause.export_synchronize(context, (prepared, storage, images))
    dynamic = tuple(storage.dynamic_regions)
    zero_ranges = [(address, size) for address, size in prepared.export_zero_ranges
        if not any(region.address <= address < region.address + region.extent for region in dynamic)]
    zero_ranges.extend((segment["address"], segment["total_size"])
                       for segment in storage.graph_pool.snapshot(include_traces=False))
    inactive = storage.unused_ranges()
    zero_ranges.extend((start, end - start) for start, end in inactive)
    transports = [runtime.native_description() for runtime in storage.transports]
    for runtime in storage.transports:
        runtime.close()
    for transport in transports:
        zero_ranges.extend((transport[part]["address"], transport[part]["nbytes"])
                           for part in ("region", "counters"))
    allocations = [Allocation(f"device-{index}", mapping.address, mapping.nbytes,
        alignment=storage.granularity, preserve_address=True)
        for index, mapping in enumerate(sorted(storage.mappings.values(), key=lambda m: m.address))]
    imports = {} if storage.peer_storage is None else storage.peer_storage.imported
    allocations.extend(Allocation(f"peer-{index}", address, size, alignment=storage.granularity,
        initialize="zero", preserve_address=True, memory="peer")
        for index, (address, size) in enumerate(sorted(imports.items())))
    allocations.extend(Allocation(f"host-{index}", region.address, region.extent, alignment=4096,
        preserve_address=True, memory="mapped_host") for index, region in enumerate(storage.host_regions)
        if not region.closed)
    allocations.extend(Allocation(region.name, region.address, region.extent, alignment=storage.granularity,
        initialize="zero", preserve_address=True, sizing=dict(capacity=region.group,
        unit_bytes=region.unit_bytes, maximum=region.maximum)) for region in dynamic)
    numeric_bounds = [*allocations, *(Allocation(f"recorded-{address}", address, size)
        for address, (size, kind) in images.allocations.items() if kind == "device")]
    numeric, parameter_layouts = live_metadata(
        numeric_bounds, storage.numeric_storage, prepared.buffers, zero_ranges=zero_ranges)
    # Warmup/binding has finalized expert storage by export time.
    from ignition.components.weights import record_final_fp4_expert_provenance_all
    record_final_fp4_expert_provenance_all()
    context, _ = flare_heliopause.current_context()
    # Absent cache configuration stays absent; the native exporter decides.
    cache = _OPTIONS.get()
    request = dict(path=str(path), context=context, rank=storage.rank, granularity=storage.granularity,
        world_size=storage.world, local_ranks=storage.local_ranks,
        publisher=_publisher_metadata(publisher),
        peer_addresses=[] if storage.peer_storage is None else list(storage.peer_storage.owned),
        peer_mappings=[] if storage.peer_storage is None else
            [part for parts in storage.peer_storage.imported_mappings.values() for part in parts],
        graphs=_graphs(prepared.graphs), allocations=[asdict(a) for a in allocations], discover_allocations=True,
        events=prepared.export_events, numeric=numeric, inactive=inactive,
        parameter_layouts=parameter_layouts, zero_ranges=zero_ranges, transports=transports,
        device_address_floor=storage.address_floor,
        execution=prepared.manifest(all_variants=True), server=None if bundle_only else {**prepared.server_config, **options},
        package_data=None if bundle_only or prepared.package_data is None else prepared.package_data.export_metadata(),
        rebuild_weights=None if getattr(prepared, "rebuild_weights", None) is None else
            rebuild_weights_metadata(prepared.rebuild_weights),
        weight_payload=weight_payload_request(getattr(prepared, "rebuild_weights", None)),
        options=_cache_options(cache, options), sources=sources)
    package = RuntimePackage(allocations=allocations, modules={}, kernels=(), owners=(prepared, storage, images))
    package.source_path = Path(path)
    package._capture(request, package.owners, images.native)
    package.memory_inventory = collect_inventory(prepared, storage, package, external=package.external)
    if publisher is not None:
        print_inventory([package.memory_inventory], console=getattr(progress, "console", None))
    return package
