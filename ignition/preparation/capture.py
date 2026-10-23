"""Compile prepared model forwards and capture one native executable per shape."""

from contextlib import nullcontext
from time import monotonic

import torch

from ignition.progress import graph_catalog, phase

from .aot import ForwardCache
from .compilation import _defer_full_collection


@_defer_full_collection()
def capture_forwards(functions, signatures, *, session, device, stream, pool,
                     collective=None, coordinator=None, compile_decoder=True, materialize=None,
                     progress=None, label="Prepare serving graphs", cache_root=None, retain_graphs=False,
                     warmup_storage=None, capture_names=None,
                     compile_kinds=("prefill", "scored_prefill", "decode", "resume", "speculative", "mixed", "vision")):
    """Retain compiled callables and capture with an optional b12x IPC runtime.

    Persistent state and graph outputs must live in caller-owned storage.
    Only graph-local intermediates may use the shared graph memory pool;
    serving replays variants serially and may choose them in any order.
    """
    if (progress is not None and not getattr(session, "_graph_catalog_reported", False)
            and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)):
        graph_catalog(progress, (signatures[name] for name in functions))
        session._graph_catalog_reported = True
    names = tuple(functions) if capture_names is None else tuple(capture_names)
    if len(set(names)) != len(names) or not set(names) <= functions.keys():
        raise ValueError("capture selection must name distinct prepared forwards")
    graphs, compiled = {}, {}
    warmup = nullcontext if warmup_storage is None else warmup_storage.transient
    if compile_decoder:
        cache = ForwardCache(namespace=session.namespace.to_dict(), device=device, root=cache_root,
                             headroom_bytes=getattr(session, "headroom_bytes", 0),
                             compile_workers=getattr(session, "requested_compile_workers", None))
        forwards = {name: function for name, function in functions.items()
                    if name in names and signatures[name].kind in compile_kinds}
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            compiled = cache.prepare_many(forwards, signatures, progress=progress,
                                          coordinator=coordinator, materialize=materialize, warmup=warmup)
        stream.synchronize()
    elif materialize is not None:
        materialize()
    with phase(progress, label, total=len(names)) as stage:
        for name in names:
            function = functions[name]
            started = monotonic()
            stream.wait_stream(torch.cuda.current_stream(device))
            with warmup(), torch.cuda.stream(stream):
                if name in compiled:
                    function = compiled[name]
                else:
                    function()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True) if retain_graphs else torch.cuda.CUDAGraph()
            communication = nullcontext() if collective is None else collective.capture(stream=stream)
            def memory_detail():
                free, total = torch.cuda.mem_get_info(device)
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                return (f"{free / (1 << 20):.0f} MiB free, "
                        f"{allocated / (1 << 20):.0f} MiB allocated, "
                        f"{reserved / (1 << 20):.0f} MiB reserved, "
                        f"{(total - free - reserved) / (1 << 20):.0f} MiB outside PyTorch")

            try:
                if progress is not None:
                    progress(f"capturing {name}: {memory_detail()}")
                with communication, session.capture(), torch.cuda.graph(graph, pool=pool, stream=stream):
                    function()
                if retain_graphs:
                    if progress is not None:
                        progress(f"instantiating {name}: {memory_detail()}")
                    graph.instantiate()
            except Exception as error:
                error.add_note(f"Graph {name!r} on {device}: {memory_detail()}")
                raise
            graphs[name] = graph
            stage.update(advance=1)
            if progress is not None:
                cached = " (compiled cache hit)" if name in compiled and compiled[name].cache_hit else ""
                progress(f"captured {name} in {monotonic() - started:.2f}s{cached}")
    torch.cuda.synchronize(device)
    return graphs, compiled
