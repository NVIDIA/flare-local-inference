"""Prepared-model ownership through package export and diagnostic execution."""

import json
from threading import get_ident
from types import MappingProxyType

import torch

from .execution import PreparedExecution


class PreparedModel(PreparedExecution):
    """Own frozen resources until export or diagnostic execution has finished.

    ``forwards`` maps each graph name to the callable its graph captured, compiled
    where preparation compiled it. Only Ignition's diagnostic eager execution calls
    them; it is None for recipes that do not retain them.
    """

    def __init__(self, *, server_config, memory_report, session, close_resources=(), package_data=None,
                 forwards=None, embedding_catalog=None, rebuild_weights=None, **kwargs):
        super().__init__(**kwargs)
        if forwards is not None and set(forwards) != set(self.graphs):
            raise ValueError("retained forwards must cover exactly the captured graphs")
        self.server_config = server_config
        self.memory_report = memory_report
        self.session = session
        self.package_data = package_data
        self.rebuild_weights = rebuild_weights
        self.forwards = None if forwards is None else MappingProxyType(dict(forwards))
        self.embedding_catalog = embedding_catalog
        self.embedding_tables = self._embedding_binding = None
        self._close_resources = tuple(close_resources)
        self._rank_drained = False
        self.runtime = None
        self.closed = False
        self._startup_undrained = False
        self._preparation_thread = get_ident()

    def manifest(self, *, all_variants=False):
        """Expose the serving width to diagnostics and the complete catalog to export."""
        manifest = super().manifest(all_variants=all_variants)
        fixed = self.server_config.get("speculative_fixed_width")
        if fixed is not None and not all_variants:
            manifest["graphs"] = [graph for graph in manifest["graphs"]
                if graph["kind"] not in ("speculative", "verification") or graph["token_capacity"] in (1, fixed)]
        if self._embedding_binding is not None and not all_variants:
            manifest["engram"] = self._embedding_binding
        return manifest

    def prepare_embedding_tables(self, *, descriptors=None):
        """Attach native RAM delivery before handoff; peers map the root's sealed backing."""
        if self.embedding_catalog is None:
            if descriptors is not None:
                raise ValueError("shared embedding descriptors require a table catalog")
            return None
        if self._consumed or self.closed or self.embedding_tables is not None:
            raise RuntimeError("embedding tables require an unconsumed model without an existing attachment")
        import flare_heliopause

        if not hasattr(flare_heliopause, "EmbeddingTables"):
            raise RuntimeError("resident embedding tables require rebuilding the installed Heliopause extension")
        tables = flare_heliopause.EmbeddingTables(self.embedding_catalog, descriptors=descriptors)
        binding = tables.bind(self.manifest(), self.export_events)
        self.embedding_tables, self._embedding_binding = tables, binding
        return tables

    def start_server(self, *, bind: str = "127.0.0.1:8000", template_defaults=None, peers=None, options=None):
        """Start in-process HTTP diagnostics with a qualification build of heliopause."""
        import flare_heliopause

        if not hasattr(flare_heliopause, "Server"):
            raise RuntimeError("in-process HTTP diagnostics require heliopause's qualification Cargo feature; "
                               "use ignition export followed by flare run for package-based serving")
        with self._handoff_lock:
            if self._consumed or self.closed:
                raise RuntimeError("prepared model has already been consumed or closed")
            if self.embedding_catalog is not None and self.embedding_tables is None:
                if peers is not None:
                    raise RuntimeError("tensor-parallel embedding tables must be shared before native handoff")
                self.prepare_embedding_tables()
            config = {**self.server_config, **(options or {}), "bind": bind,
                      "template_defaults": {} if template_defaults is None else template_defaults}
            manifest = self.manifest()
            torch.cuda.synchronize(self.device)
            try:
                arguments = {} if peers is None else {"peer_fds": [peer.fileno() for peer in peers]}
                server = flare_heliopause.Server(self, manifest, json.dumps(config), **arguments)
            except BaseException as error:
                if getattr(error, "resources_retained", False):
                    self._startup_undrained = True
                    self._consumed = True
                raise
            self.runtime = server
            self._consumed = True
            return server

    def serve_rank(self, socket, *, rank, world):
        super().serve_rank(socket, rank=rank, world=world)
        self._rank_drained = True

    def close(self):
        with self._handoff_lock:
            self._close_locked()

    def _close_locked(self):
        if self.closed:
            return
        if get_ident() != self._preparation_thread:
            raise RuntimeError("prepared model cleanup must run on its preparation thread")
        if self._startup_undrained:
            raise RuntimeError("native startup could not confirm stream drain; prepared resources remain retained")
        shutdown_error = None
        if self.runtime is not None:
            try:
                self.runtime.close()
            except BaseException as error:
                if not self.runtime.drained:
                    raise
                shutdown_error = error
            if not self.runtime.drained:
                raise RuntimeError("native streams did not drain; prepared resources remain retained")
        elif self._consumed and not self._rank_drained:
            raise RuntimeError("externally started executor must retain this model until its own shutdown")
        torch.cuda.synchronize(self.device)
        for graph in self.graphs.values():
            graph.reset()
        self.session.close()
        for resource in self._close_resources:
            resource.close()
        self.rebuild_weights = None
        self._resources = ()
        self._buffers = MappingProxyType({})
        self._graphs = MappingProxyType({})
        self.forwards = None
        self.session = None
        self._close_resources = ()
        self.runtime = None
        self._embedding_binding = self.embedding_tables = None
        self.closed = True
        if shutdown_error is not None:
            raise shutdown_error

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.close()
        except BaseException as error:
            if value is None:
                raise
            value.add_note(f"prepared model cleanup failed: {error!r}")

