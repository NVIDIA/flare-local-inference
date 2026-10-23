"""Spawn preparation ranks and transfer their lifetime supervision to Rust."""

import contextlib
import os
import signal
import socket
import traceback
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import torch.distributed as dist
import torch.multiprocessing as mp

from ignition.progress import phase
from .tuning_replicas import TuningComplete


# Peers wait here while rank zero compiles the complete serving catalog.
_PREPARATION_TIMEOUT = timedelta(hours=2)


def _use_loopback():
    # Ranks on one host connect over loopback. The hostname can resolve to an address the network
    # later withdraws, such as an expiring IPv6 prefix, which stalls every established connection.
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"


def minimum_across_ranks(value, group):
    """The smallest integer any rank reports, for sizing storage every rank must allocate."""
    if group is None:
        return value
    import torch

    result = torch.tensor(value, dtype=torch.int64)
    dist.all_reduce(result, op=dist.ReduceOp.MIN, group=group)
    return result.item()


def _peer(parent, snapshot, config, rank, rendezvous, connection, control, prepare, packaged, recording):
    import flare_heliopause

    flare_heliopause.bind_parent_lifetime(parent)
    # Rank zero coordinates shutdown; a terminal interrupt must not end a peer mid-drain.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        _use_loopback()
        dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=len(config.devices),
                                timeout=_PREPARATION_TIMEOUT)
        recorder = storage = None
        if packaged:
            from .module_images import ModuleImages
            from .package_storage import PackageStorage
            import torch

            # Only an export needs module images; without one, a profiler may own CUPTI.
            recorder = ModuleImages() if recording else None
            with recorder if recording else contextlib.nullcontext():
                torch.cuda.set_device(config.devices[rank])
                storage = PackageStorage(rank=rank, world=len(config.devices), device=config.devices[rank])
                with storage:
                    model = prepare(snapshot, config, rank=rank, group=dist.group.WORLD, package_storage=storage)
        else:
            model = prepare(snapshot, config, rank=rank, group=dist.group.WORLD)
        model.manifest()
        control.send(model.memory_report)
        command = control.recv()
        if isinstance(command, tuple) and command[0] == "embedding_tables":
            from multiprocessing.reduction import recv_handle

            _, catalog, count = command
            if count != 2 * len(catalog["tables"]):
                raise ValueError("shared embedding descriptor count differs from its catalog")
            descriptors = []
            try:
                for _ in range(count):
                    descriptors.append(recv_handle(control))
                # Equal catalogs share physical RAM. Disjoint table partitions load their
                # own checkpoint spans and have independent delivery mailboxes.
                model.prepare_embedding_tables(descriptors=descriptors if catalog == model.embedding_catalog else None)
            finally:
                for descriptor in descriptors:
                    os.close(descriptor)
            control.send("embedding_tables_ready")
            command = control.recv()
        if isinstance(command, tuple) and command[0] == "export":
            from .runtime_package import export_model

            if recorder is None:
                raise RuntimeError("export requires module images recorded during preparation")

            from contextlib import nullcontext
            from .package_snapshot import export_options
            _, path, options, cache = command
            dist.barrier()
            storage.peer_storage.reset()
            dist.barrier()
            with export_options(cache.cache, no_weights=cache.no_weights) if cache else nullcontext():
                package = export_model(model, recorder, storage, path, options=options, sources=True)
            control.send({"peer_memory": list(storage.peer_storage.owned.items()),
                          "memory_inventory": package.memory_inventory})
            command = control.recv()
        control.close()
        if command == "serve":
            model.serve_rank(connection, rank=rank, world=len(config.devices))
        elif isinstance(command, tuple) and command[0] == "direct":
            from ignition.direct.executor import serve_peer

            serve_peer(model, connection, **command[1])
        elif command != "close":
            raise RuntimeError("invalid prepared rank handoff")
        connection.close()
        model.close()
        if storage is not None:
            for region in storage.host_regions:
                region.close()
        dist.destroy_process_group()
    except TuningComplete:
        control.send("tuning_complete")
        if control.recv() != "close":
            os._exit(1)
        control.close()
        connection.close()
        dist.destroy_process_group()
    except BaseException:  # noqa: BLE001 - Never unwind CUDA resources around a stranded collective.
        traceback.print_exc()
        os._exit(1)


class PreparedRankGroup:
    """Root model and owned peer processes through native coordinated shutdown."""

    def __init__(self, model, processes, connections, controls, supervisor, rendezvous, reports, *, progress=None):
        self.model, self.processes, self.connections, self.controls = model, processes, connections, controls
        self.supervisor, self.rendezvous = supervisor, rendezvous
        self.progress = progress
        self.memory_report = {"world_size": len(processes) + 1, "ranks": [model.memory_report, *reports]}
        self._started = self._closed = False

    def start_server(self, *, bind="127.0.0.1:8000", template_defaults=None, options=None):
        return self._start_native(lambda: self.model.start_server(bind=bind,
            template_defaults=template_defaults, peers=self.connections, options=options))

    def start_executor(self):
        """Attach shared RAM tables and hand every rank to the diagnostic native executor."""
        def start():
            executor = self.model.start_executor(peers=self.connections)
            self.model.runtime = executor
            return executor

        return self._start_native(start)

    def _start_native(self, start):
        if self._started or self._closed:
            raise RuntimeError("prepared rank group has already been consumed or closed")
        try:
            if self.model.embedding_catalog is not None:
                from multiprocessing.reduction import send_handle

                with phase(self.progress, "Load resident embedding tables") as stage:
                    tables = self.model.prepare_embedding_tables()
                    stage.update(f"{tables.nbytes / (1 << 30):.1f} GiB on rank zero; binding {len(self.processes)} peers")
                descriptors = tables.descriptors
                for control, process in zip(self.controls, self.processes, strict=True):
                    control.send(("embedding_tables", self.model.embedding_catalog, len(descriptors)))
                    for descriptor in descriptors:
                        send_handle(control, descriptor, process.pid)
                for control in self.controls:
                    if control.recv() != "embedding_tables_ready":
                        raise RuntimeError("rank did not confirm its resident embedding binding")
            for control in self.controls:
                control.send("serve")
                control.close()
            self._started = True
            server = start()
            for connection in self.connections:
                connection.close()
            return server
        except BaseException:  # noqa: BLE001 - Startup failure can strand peer collectives.
            traceback.print_exc()
            self.supervisor.abort("native rank-group startup failed")

    def start_direct(self, *, eager):
        """Hand every peer to Ignition's diagnostic step loop; returns the sockets that carry its steps.

        Rank zero then sends each step it decides over these sockets, and every
        rank executes the same steps in order. Rank zero sends the loop's stop
        message before closing the group.
        """
        if self._started or self._closed:
            raise RuntimeError("prepared rank group has already been consumed or closed")
        for control in self.controls:
            control.send(("direct", {"eager": eager}))
            control.close()
        self._started = True
        return list(self.connections)

    def close(self):
        if self._closed:
            return
        try:
            self.supervisor.begin_shutdown(60000)
            if not self._started:
                for control in self.controls:
                    control.send("close")
                    control.close()
            self.model.close()
            dist.destroy_process_group()
            self.supervisor.finish_shutdown()
            for process in self.processes:
                process.join()
            for connection in self.connections:
                connection.close()
            self.rendezvous.cleanup()
            self._closed = True
        except BaseException:  # noqa: BLE001 - Failed drain requires process-level reclamation.
            traceback.print_exc()
            self.supervisor.abort("rank-group cleanup could not confirm drain")

    def __enter__(self):
        return self

    def manifest(self):
        return self.model.manifest()

    def __exit__(self, kind, value, traceback):
        self.close()


def prepare_group(snapshot, config, prepare, *, progress=None, package_storage=None, ranks=None, checkpoint=None):
    """Prepare every rank before returning a root-owned native handoff.

    This function owns the process's Gloo group and must run in the main thread.
    Peer failure or unconfirmed CUDA drain terminates this process and its peers.
    `ranks` is rank zero's rendezvous when every rank runs on its own host; its
    peers resolve `checkpoint` locally and, for export, prepare into their own package storage.
    """
    import flare_heliopause

    if ranks is not None:
        return ranks.prepare(snapshot, config, prepare, checkpoint=checkpoint, progress=progress,
                             package_storage=package_storage)

    if dist.is_initialized():
        raise RuntimeError("rank preparation requires exclusive ownership of the control group")
    from .module_images import ModuleImages

    recording = package_storage is not None and ModuleImages.active > 0
    rendezvous = TemporaryDirectory(prefix="flare-ranks-")
    address = Path(rendezvous.name, "gloo").as_uri()
    context = mp.get_context("spawn")
    processes, connections, controls = [], [], []
    try:
        for rank in range(1, len(config.devices)):
            root, peer = socket.socketpair()
            connections.append(root)
            control, child_control = context.Pipe()
            controls.append(control)
            process = context.Process(target=_peer, args=(os.getpid(), snapshot, config, rank,
                address, peer, child_control, prepare, package_storage is not None, recording),
                name=f"flare-rank-{rank}")
            try:
                process.start()
            finally:
                peer.close()
                child_control.close()
            processes.append(process)
        supervisor = flare_heliopause.RankSupervisor([process.pid for process in processes])
    except BaseException:
        for process in processes:
            process.kill()
            process.join()
        for resource in (*connections, *controls):
            resource.close()
        rendezvous.cleanup()
        raise
    try:
        with phase(progress, "Connect tensor-parallel ranks"):
            _use_loopback()
            dist.init_process_group("gloo", init_method=address, rank=0, world_size=len(config.devices),
                                    timeout=_PREPARATION_TIMEOUT)
        model = prepare(snapshot, config, rank=0, group=dist.group.WORLD, progress=progress,
                        **({} if package_storage is None else {"package_storage": package_storage}))
        model.manifest()
        with phase(progress, "Wait for tensor-parallel ranks", total=len(controls)) as stage:
            reports = []
            for control in controls:
                reports.append(control.recv())
                stage.update(advance=1)
        return PreparedRankGroup(model, processes, connections, controls, supervisor, rendezvous, reports,
                                 progress=progress)
    except TuningComplete:
        supervisor.begin_shutdown(60000)
        for control in controls:
            if control.recv() != "tuning_complete":
                supervisor.abort("tuning replica did not confirm completion")
            control.send("close")
            control.close()
        dist.destroy_process_group()
        supervisor.finish_shutdown()
        for process in processes:
            process.join()
        for connection in connections:
            connection.close()
        rendezvous.cleanup()
        raise
    except BaseException:  # noqa: BLE001 - Preparation interruption must terminate every rank.
        traceback.print_exc()
        supervisor.abort("tensor-parallel preparation failed")
