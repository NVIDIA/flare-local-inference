"""Package-owned TP4 reductions across retained graph variants and native replay."""

from contextlib import contextmanager
from copy import deepcopy
import ctypes
from datetime import timedelta
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import weakref

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


_gpu_only = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
    reason="requires four assigned CUDA devices and FLARE_RUN_TP_CUDA_TESTS=1",
)


def _operands(rows, columns, step):
    generator = torch.Generator().manual_seed(4181 + columns + step)
    values = [(torch.randn((rows, columns), generator=generator)
               * 2. ** torch.randint(-12, 13, (rows, columns), generator=generator)).bfloat16()
              for _ in range(4)]
    # Cancellation exposes the FP32 addition order before the final BF16 cast.
    values[2][:rows // 2] = -values[0][:rows // 2]
    values[3][:rows // 2] = -values[1][:rows // 2]
    return values


def _selected_query(operation):
    """Describe reduction arithmetic using the persistent runtime selector."""
    from b12x.comm.pcie._joint_profile import JOINT_FAMILIES, JOINT_MODES, selected_profile

    state = operation.plan.prepared.state
    query = state.query
    call = dict(query.call)
    if call.get("transport") in JOINT_FAMILIES:
        encoded = int(state.selector.item())
        names = {number: name for name, number in JOINT_MODES.items()}
        mode = names.get(encoded >> 16)
        assert mode is not None, encoded
        active = encoded & 0xffff
        assert selected_profile(call["transport"], call["blocks"], mode, active,
                                world_size=query.world_size) == encoded
        call.update(submode=mode, active_blocks=active)
    return dict(surface=query.surface, call=call)


def _expected(values, rank, query):
    surface = query["surface"]
    assert surface in ("OneshotAllReduce.all_reduce", "PCIeTwoShotBF16.all_reduce"), surface
    transport = query["call"]["transport"]
    if surface == "PCIeTwoShotBF16.all_reduce":
        assert transport in ("push", "pull"), transport
        transport = "bulk_" + transport
    else:
        if transport in ("joint_latency", "joint_bandwidth"):
            from b12x.comm.pcie._joint_profile import joint_modes

            mode = query["call"]["submode"]
            assert mode in joint_modes(transport, len(values)), (transport, mode)
            transport = {"full": "remote_push_full", "payload_rsag": "remote_push_rsag"}.get(mode, mode)
        else:
            assert transport in ("pull", "remote_push_full", "remote_push_rsag"), transport
    bulk = transport in ("bulk_push", "bulk_pull", "serial_push", "queued_push")
    if transport in ("remote_push_full", "remote_push_rsag"):
        values = [value.clone() for value in values]
        for value in values:
            bits = value.view(torch.int16)
            bits[bits == 0] = -32768
    cyclic = []
    for first in range(4):
        value = values[first].float()
        if bulk:
            value = torch.zeros_like(value) + value
        for offset in range(1, 4):
            value = value + values[(first + offset) % 4].float()
        cyclic.append(value.bfloat16())
    if bulk or transport == "remote_push_rsag":
        elements = values[0].numel()
        assert elements % 8 == 0
        packs = elements // 8
        shard = ((packs + 3) // 4 if bulk else packs // 4) * 8
        parts = [cyclic[owner].flatten()[min(elements, owner * shard):
                 elements if owner == 3 else min(elements, (owner + 1) * shard)].clone()
                 for owner in range(4)]
        if transport == "remote_push_rsag":
            for owner, part in enumerate(parts):
                if owner != rank:
                    bits = part.view(torch.int16)
                    bits[bits == 0] = -32768
        return torch.cat(parts).view_as(values[0])
    return cyclic[rank]


def test_collective_oracle_preserves_transport_zero_bits_and_rejects_unknown_transport():
    from b12x.comm.pcie._joint_profile import JOINT_FAMILIES, joint_modes

    zeros = [torch.zeros((1, 32), dtype=torch.bfloat16) for _ in range(4)]
    for transport, bits in (("pull", 0), ("remote_push_full", -32768), ("remote_push_rsag", -32768)):
        query = dict(surface="OneshotAllReduce.all_reduce", call=dict(transport=transport))
        assert torch.all(_expected(zeros, 0, query).view(torch.int16) == bits)
    negative = [value.neg() for value in zeros]
    for transport in ("push", "pull"):
        query = dict(surface="PCIeTwoShotBF16.all_reduce", call=dict(transport=transport))
        assert torch.all(_expected(negative, 0, query).view(torch.int16) == 0)
    for family in JOINT_FAMILIES:
        for mode in joint_modes(family, 4):
            query = dict(surface="OneshotAllReduce.all_reduce", call=dict(transport=family, submode=mode))
            bits = -32768 if mode in ("full", "payload_rsag") else 0
            assert torch.all(_expected(zeros, 0, query).view(torch.int16) == bits)
            bits = 0 if mode in ("bulk_push", "bulk_pull", "serial_push", "queued_push") else -32768
            assert torch.all(_expected(negative, 0, query).view(torch.int16) == bits)
        with pytest.raises(AssertionError):
            _expected(zeros, 0, dict(surface="OneshotAllReduce.all_reduce",
                                   call=dict(transport=family, submode="unknown")))
    for transport in ("unknown", None):
        with pytest.raises(AssertionError):
            _expected(zeros, 0, dict(surface="OneshotAllReduce.all_reduce", call=dict(transport=transport)))
    with pytest.raises(AssertionError):
        _expected(zeros, 0, dict(surface="unknown", call={}))
    with pytest.raises(AssertionError):
        _expected(zeros, 0, dict(surface="PCIeTwoShotBF16.all_reduce", call=dict(transport="unknown")))


@pytest.mark.parametrize("elements", (8, 40, 72))
@pytest.mark.parametrize("mode", ("payload_rsag", "bulk_push", "bulk_pull", "serial_push", "queued_push"))
def test_collective_oracle_preserves_pack_owners_and_cancellation_order(elements, mode):
    values = [torch.full((elements,), value, dtype=torch.bfloat16)
              for value in (1e8, 1, -1e8, 1)]
    query = dict(surface="OneshotAllReduce.all_reduce",
                 call=dict(transport="joint_bandwidth", submode=mode))
    packs = elements // 8
    shard = packs // 4 if mode == "payload_rsag" else (packs + 3) // 4
    owners = [min(pack // shard, 3) if shard else 3 for pack in range(packs)]
    for rank in range(4):
        expected = torch.tensor([1 if owner % 2 == 0 else 0 for owner in owners], dtype=torch.bfloat16)
        if mode == "payload_rsag":
            for index, owner in enumerate(owners):
                if owner != rank and expected[index] == 0:
                    expected.view(torch.int16)[index] = -32768
        assert torch.equal(_expected(values, rank, query).view(torch.int16),
                           expected.repeat_interleave(8).view(torch.int16))


@pytest.mark.parametrize("family", ("joint_latency", "joint_bandwidth"))
def test_collective_oracle_reads_the_device_selector_instead_of_prepared_submode(family):
    from b12x.comm.pcie._joint_profile import JOINT_FAMILIES, JOINT_MODES

    call = dict(transport=family, threads=256, blocks=64,
                submode=JOINT_FAMILIES[family][0], active_blocks=16)
    selector = torch.tensor([(JOINT_MODES["serial_push"] << 16) | 1], dtype=torch.uint32)
    state = SimpleNamespace(query=SimpleNamespace(surface="OneshotAllReduce.all_reduce",
                            call=call, world_size=4), selector=selector)
    operation = SimpleNamespace(plan=SimpleNamespace(prepared=SimpleNamespace(state=state)))
    selected = _selected_query(operation)["call"]
    assert selected["submode"] == "serial_push" and selected["active_blocks"] == 1
    selector.fill_((JOINT_MODES["pull"] << 16) | 8)
    selected = _selected_query(operation)["call"]
    assert selected["submode"] == "pull" and selected["active_blocks"] == 8
    assert call["submode"] == JOINT_FAMILIES[family][0] and call["active_blocks"] == 16
    selector.zero_()
    with pytest.raises(ValueError, match="CTA"):
        _selected_query(operation)


def _writer_ladder(ladder):
    """Exercise both writer protocols without depending on a timing winner."""
    from b12x.comm.pcie._joint_profile import JointMode
    from ignition.preparation.collective_ladder import validate

    ladder = deepcopy(ladder)
    ladder["name"] = "tp4-writer-protocol-coverage"
    boundary = 64 << 10
    assert any(bucket["max_bytes"] == boundary for bucket in ladder["buckets"])
    for bucket in ladder["buckets"]:
        if bucket["max_bytes"] <= boundary:
            bucket["profiles"] = {JointMode.SERIAL_PUSH.name.lower(): [1]}
        else:
            blocks = bucket["blocks"]
            bucket["profiles"] = {JointMode.QUEUED_PUSH.name.lower(): sorted({min(count, blocks) for count in (8, 32, blocks)})}
    return validate(ladder)


def _writer_observation(state, mode, generation=None):
    """Read local protocol storage after native completion, outside allocation checks."""
    from b12x.comm.pcie._joint_cute import _EPOCH_OFFSET
    from b12x.comm.pcie._joint_profile import (
        JointMode, JOINT_MAX_RANKS, JOINT_QUEUE_SLOT0_OFFSET, JOINT_QUEUE_SLOT_STRIDE,
    )

    stream = torch.cuda.current_stream(state.runtime.device)
    base = state.workspace.shared.local_ptr
    def read(dtype, count, offset):
        values = (dtype * count)()
        state.runtime._ipc.cudaMemcpyAsync(ctypes.addressof(values), base + offset,
            ctypes.sizeof(values), stream.cuda_stream)
        stream.synchronize()
        return tuple(values)
    result = {"epoch": read(ctypes.c_uint32, 2, _EPOCH_OFFSET)}
    if mode == JointMode.SERIAL_PUSH:
        result["barriers"] = read(ctypes.c_uint32, JOINT_MAX_RANKS, 0)
    elif mode == JointMode.QUEUED_PUSH:
        if generation is not None:
            offset = JOINT_QUEUE_SLOT0_OFFSET + (generation % 2) * JOINT_QUEUE_SLOT_STRIDE
            result["queue_token"] = read(ctypes.c_uint64, 1, offset)[0]
    else:
        raise AssertionError(f"native writer observation received {mode!r}")
    return result


def _assert_writer_progress(mode, rank, world, before, after):
    from b12x.comm.pcie._joint_profile import JointMode, JOINT_QUEUE_FIRST_PHASE

    generation, arrivals = before["epoch"]
    assert arrivals == 0, before
    assert after["epoch"] == (ctypes.c_uint32(generation + 1).value, 0), (before, after)
    if mode == JointMode.SERIAL_PUSH:
        for peer, (start, end) in enumerate(zip(before["barriers"], after["barriers"], strict=True)):
            expected = 2 if peer < world and peer != rank else 0
            assert ctypes.c_uint32(end - start).value == expected, (peer, start, end, expected)
    elif mode == JointMode.QUEUED_PUSH:
        epoch_bits = ctypes.sizeof(ctypes.c_uint32) * 8
        assert after["queue_token"] == ((generation << epoch_bits) | (JOINT_QUEUE_FIRST_PHASE + 2 * world)), (before, after)
    else:
        raise AssertionError(f"native writer progress received {mode!r}")


def _rank(rank, rendezvous, directory, widths, collective_tuning, writer_protocols=False):
    from ignition.components.collectives import PreparedAllReduce
    from ignition.preparation.collective_ladder import load
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.runtime_package import export_model
    from ignition.preparation.startup_tuning import capture_consumers
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(minutes=5))
    max_rows = max((2 << 20) // (width * 2) + 1 for width in widths)
    runtime = worker = None
    graphs, tuners = {}, {}
    ladder = (load(Path(__file__).resolve().parents[2] / "ignition/preparation/ladders/gb202_pcie_tp4.json")
              if collective_tuning == "runtime" else None)
    if writer_protocols:
        assert collective_tuning == "runtime"
        ladder = _writer_ladder(ladder)

    @contextmanager
    def execution_lifetime():
        """Drain native work before releasing either graph catalog."""
        nonlocal runtime, worker
        try:
            yield
        finally:
            error = None
            if worker is not None:
                try:
                    worker.close()
                except BaseException as failure:
                    error = failure
                if not worker.drained:
                    # Process exit must reclaim owners that CUDA may still access.
                    print(f"native collective test executor did not drain: {error!r}",
                          file=sys.stderr, flush=True)
                    os._exit(1)
                worker = None
            torch.cuda.synchronize(device)
            for graph in (*graphs.values(), *(tuner.graph for tuner in tuners.values())):
                graph.reset()
            graphs.clear()
            tuners.clear()
            if runtime is not None:
                runtime.close()
                runtime = None
            if error is not None:
                active = sys.exception()
                if active is None:
                    raise error
                active.add_note(f"native collective cleanup failed after drain: {error!r}")

    try:
        with ModuleImages() as images, PackageStorage(rank=rank, world=4, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=max_rows,
                hidden_size=max(widths), ipc=storage.peer_storage, collective_tuning=collective_tuning)
            with RecordedPreparationSession(device=device, checkpoint="tp4-collectives",
                    workload_schema="bf16-hidden-reduction", numerical_policy="fp32-single-rounding",
                    coordinator=coordinator, benchmark_storage=storage, collective_tuning=collective_tuning,
                    collective_ladder=ladder) as session, execution_lifetime():
                # Hidden widths share one runtime, which
                # transports flat BF16 packs independently of tensor row geometry.
                sources = {width: torch.empty((max_rows, width), dtype=torch.bfloat16, device=device)
                           for width in widths}
                outputs, operations = {}, {}
                for width, source in sources.items():
                    upper = (2 << 20) // (width * 2)
                    row_counts = (1, 8, 12, 13, 32, 33, 128, upper, upper + 1)
                    for rows in row_counts:
                        key = width, rows
                        outputs[key] = torch.empty_like(source[:rows])
                        operations[key] = PreparedAllReduce(source[:rows], outputs[key], runtime=runtime,
                            session=session, coordinator=coordinator, name=f"reduce-{width}-{rows}")
                session.tune(show_progress=False)
                stream = torch.cuda.Stream(device=device)
                for epoch in range(2):
                    for (width, rows), operation in operations.items():
                        sources[width].fill_(rank + 1)
                        operation(sources[width][:rows])
                        torch.cuda.synchronize(device)
                        dist.barrier()
                        graph = torch.cuda.CUDAGraph(keep_graph=True)
                        stream.wait_stream(torch.cuda.current_stream(device))
                        with capture_consumers(session, graph), runtime.capture(stream=stream), session.capture(), torch.cuda.graph(
                                graph, stream=stream, pool=storage.graph_pool.id):
                            operation(sources[width][:rows])
                        graph.instantiate()
                        graphs[f"reduce-{width}-{rows}-{epoch}"] = graph
                registry = getattr(session, "_startup_collectives", None)
                tuning_buffers, bindings, startup_owners = {}, {}, ()
                if collective_tuning == "runtime":
                    assert registry is not None and registry.groups
                    tuning_buffers, tuners, bindings, startup_owners = registry.catalog(graphs)
                    assert len(tuners) == 1 and set(bindings) == set(graphs)
                    assert {field for tuner in tuners.values() for field in tuner.outputs} == {
                        binding.buffer for values in bindings.values() for binding in values}
                else:
                    assert registry is None
                session.freeze()

                # Both transports must use the same explicit VMM ownership path
                # that the runtime-package exporter uses for peer resources.
                peer = storage.peer_storage
                assert len(peer.owned) >= 2 and len(peer.imported) >= 6
                for address, size in peer.owned.items():
                    assert address in storage.mappings
                    assert storage.mappings[address].nbytes == size
                for parts in peer.imported_mappings.values():
                    for address, size in parts:
                        assert images.allocations[address] == (size, "device")
                owners = [None] * 4
                dist.all_gather_object(owners, dict(peer.owned))
                other_owners = {address: size for index, owned in enumerate(owners) if index != rank
                                for address, size in owned.items()}
                assert dict(peer.imported) == other_owners

                buffers = {"frame_index": torch.zeros(1, dtype=torch.int32, device=device),
                    "submission_sequence": torch.zeros(1, dtype=torch.int64, device=device),
                    "completion_records": torch.zeros((2, 1, 24), dtype=torch.int64, device=device)}
                buffers.update({f"source-{width}": value for width, value in sources.items()})
                buffers.update({f"output-{width}-{rows}": value
                                for (width, rows), value in outputs.items()})
                buffers.update(tuning_buffers)
                bundle = PreparedExecution(device=rank, graphs=graphs, buffers=buffers,
                    signatures={name: GraphSignature("decode", 1, 1) for name in graphs}, max_active=1,
                    tuning_graphs=tuners, tuned_bindings=bindings,
                    resources=(runtime, session, stream, storage, operations, sources, outputs, *startup_owners))
                bundle.server_config = {}
                bundle.package_data = None
                torch.cuda.synchronize(device)
                dist.barrier()
                peer.reset()
                torch.cuda.synchronize(device)
                dist.barrier()
                # Stage a tiny unsigned rank bundle, not a model checkpoint.
                package = export_model(bundle, images, storage,
                    directory / f"rank-{rank}.json", options={}, bundle_only=True, sources=True)
                assert package.export_counts["modules"] > 0
                exported = {(allocation.address, allocation.nbytes) for allocation in package.allocations
                            if allocation.memory == "peer"}
                expected_peers = set(peer.imported.items())
                required, unused = expected_peers, set()
                if collective_tuning == "runtime":
                    def imports(shared_buffers):
                        return {(pointer, peer.imported[pointer])
                                for shared in shared_buffers for pointer in shared.remote_ptrs}

                    states = [operation.plan.prepared.state for operation in operations.values()]
                    channels = {id(state.runtime): state.runtime for state in states}
                    required = imports([state.workspace.shared for state in states])
                    required.update(imports(owner.shared for owner in startup_owners))
                    roots = list(buffers.values())
                    for state in states:
                        roots.extend((state.workspace.peer_table, state.workspace.signal_table))
                    for owner in startup_owners:
                        roots.extend((owner.peers, owner.samples, owner.profiles, owner.mismatch,
                            owner.invalid, owner.eviction, owner.eviction_sum, owner.inputs.source,
                            owner.inputs.out, *owner.inputs.reference.seeds, *owner.inputs.reference.expected))
                    available = imports(shared for channel in channels.values() for shared in channel._owned_buffers)
                    expected_peers = required.copy()
                    for channel in channels.values():
                        # A captured tensor roots its complete allocator backing. A
                        # coallocated oneshot pointer table retains its imports even
                        # though joint kernels use their workspace's pointer tables.
                        address = channel.rank_data.data_ptr()
                        backing = next(mapping for mapping in storage.mappings.values()
                                       if mapping.address <= address < mapping.address + mapping.nbytes)
                        if any(tensor is not None and backing.address <= tensor.data_ptr() < backing.address + backing.nbytes
                               for tensor in roots):
                            pointers = (*channel.signal_ptrs, *(pointer for slot in channel._eager_ptrs for pointer in slot))
                            expected_peers.update((address, nbytes) for pointer in pointers
                                for address, nbytes in available if address <= pointer < address + nbytes)
                    twoshot = getattr(runtime, "_bf16_twoshot", None)
                    # The separately allocated two-shot channel has no captured consumer.
                    unused_twoshot = set() if twoshot is None else imports(twoshot._owned_buffers)
                    registered = available | unused_twoshot
                    assert registered == set(peer.imported.items()), dict(
                        imported_count=len(peer.imported), registered_count=len(registered),
                        missing=sorted(set(peer.imported.items()) - registered),
                        extra=sorted(registered - set(peer.imported.items())))
                    assert expected_peers <= available and not expected_peers.intersection(unused_twoshot), dict(
                        required_count=len(required), expected_count=len(expected_peers), available_count=len(available),
                        unregistered=sorted(expected_peers - available),
                        used_twoshot=sorted(expected_peers.intersection(unused_twoshot)))
                    unused = (available - expected_peers) | unused_twoshot
                difference = dict(imported_count=len(peer.imported), required_count=len(required),
                    expected_count=len(expected_peers), exported_count=len(exported), unused_count=len(unused),
                    missing=sorted(expected_peers - exported), extra=sorted(exported - expected_peers),
                    retained_unused=sorted(exported.intersection(unused)))
                assert not exported.intersection(unused), difference
                assert exported == expected_peers, difference
                del package
                selectors = {name: buffers[name] for tuner in tuners.values() for name in tuner.outputs}
                statuses = {name: value for name, value in tuning_buffers.items() if name.endswith("-status")}
                for selector in selectors.values():
                    selector.zero_()
                for status in statuses.values():
                    status.fill_(99)
                torch.cuda.synchronize(device)
                dist.barrier()
                bundle_ref = weakref.ref(bundle)
                startup_refs = tuple(weakref.ref(owner) for owner in startup_owners)
                worker = bundle.start_executor()
                del bundle, startup_owners, registry
                selected = {name: int(selector.item()) for name, selector in selectors.items()}
                assert coordinator.same_on_every_rank(tuple(selected.items()))
                assert all(status.item() == 0 for status in statuses.values())
                queries = {key: _selected_query(operation) for key, operation in operations.items()}
                if writer_protocols:
                    from b12x.comm.pcie._joint_profile import JointMode, JOINT_MODES

                    writers = {JointMode.SERIAL_PUSH, JointMode.QUEUED_PUSH}
                    modes = {key: JointMode(JOINT_MODES[query["call"]["submode"]]) for key, query in queries.items()}
                    assert set(modes.values()) == writers
                names = list(graphs)
                # Retain both capture epochs while interleaving all capacities.
                for step in range(3):
                    operands = {width: _operands(max_rows, width, step) for width in widths}
                    for index in (list(range(len(names))) if step % 2 == 0
                                  else list(reversed(range(len(names))))):
                        _, width, rows, _ = names[index].split("-")
                        width, rows = int(width), int(rows)
                        values = operands[width]
                        sources[width].copy_(values[rank])
                        for output in outputs.values():
                            output.fill_(float("nan"))
                        expected_source = sources[width].clone()
                        torch.cuda.synchronize(device)
                        dist.barrier()
                        if writer_protocols:
                            state = operations[width, rows].plan.prepared.state
                            mode = modes[width, rows]
                            protocol_before = _writer_observation(state, mode)
                        before = torch.cuda.memory_stats(device)
                        ticket = worker.submit(index, [])
                        assert ticket is not None
                        deadline = time.monotonic() + 60
                        completion = worker.poll()
                        while completion is None and time.monotonic() < deadline:
                            time.sleep(.001)
                            completion = worker.poll()
                        assert completion is not None and completion.fence == ticket
                        after = torch.cuda.memory_stats(device)
                        for key in ("allocation.all.allocated", "allocation.all.freed",
                                    "segment.all.allocated", "segment.all.freed"):
                            assert before[key] == after[key]
                        if writer_protocols:
                            protocol_after = _writer_observation(state, mode, protocol_before["epoch"][0])
                            _assert_writer_progress(mode, rank, state.query.world_size, protocol_before, protocol_after)
                        assert bundle_ref() is not None and all(reference() is not None for reference in startup_refs)
                        assert {name: int(selector.item()) for name, selector in selectors.items()} == selected
                        assert all(status.item() == 0 for status in statuses.values())
                        assert torch.equal(sources[width].view(torch.uint8), expected_source.view(torch.uint8))
                        expected = _expected([value[:rows] for value in values], rank, queries[width, rows])
                        assert torch.equal(outputs[width, rows].cpu().view(torch.uint8), expected.view(torch.uint8))
                dist.barrier()
            assert bundle_ref() is None
            assert not peer.owned and not peer.imported and not peer.imported_mappings
    finally:
        if worker is not None:
            worker.close()
        for graph in (*graphs.values(), *(tuner.graph for tuner in tuners.values())):
            graph.reset()
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("widths", [(4096, 5120), (2560,)], ids=["4096-5120", "qwen-2560"])
@pytest.mark.parametrize("collective_tuning", ("static", "runtime"))
@_gpu_only
def test_package_owned_tp4_collective_graph_variants(tmp_path, widths, collective_tuning):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign four CUDA devices explicitly"
    assert torch.cuda.device_count() == 4
    mp.spawn(_rank, args=((tmp_path / "gloo").as_uri(), tmp_path, widths, collective_tuning), nprocs=4, join=True)


@pytest.mark.parametrize("widths", [(4096, 5120), (2560,)], ids=["4096-5120", "qwen-2560"])
@_gpu_only
def test_package_owned_tp4_writer_graph_variants(tmp_path, widths):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign four CUDA devices explicitly"
    assert torch.cuda.device_count() == 4
    mp.spawn(_rank, args=((tmp_path / "gloo").as_uri(), tmp_path, widths, "runtime", True), nprocs=4, join=True)


def test_writer_ladder_keeps_geometry_and_covers_both_protocols():
    from b12x.comm.pcie._joint_profile import JointMode
    from ignition.preparation.collective_ladder import load, select

    original = load(Path(__file__).resolve().parents[2] / "ignition/preparation/ladders/gb202_pcie_tp4.json")
    ladder = _writer_ladder(original)
    for before, after in zip(original["buckets"], ladder["buckets"], strict=True):
        assert {key: value for key, value in before.items() if key != "profiles"} == {
            key: value for key, value in after.items() if key != "profiles"}
    conditions = dict(world_size=4, capability=(12, 0), inplace=False)
    assert select(ladder, 12 * 2560 * 2, **conditions)["profiles"] == {JointMode.SERIAL_PUSH.name.lower(): [1]}
    assert set(select(ladder, 13 * 2560 * 2, **conditions)["profiles"]) == {JointMode.QUEUED_PUSH.name.lower()}
    assert original["name"] != ladder["name"]
    for blocks, expected in ((1, [1]), (4, [4]), (16, [8, 16]), (32, [8, 32])):
        limited = deepcopy(original)
        for bucket in limited["buckets"]:
            bucket["blocks"] = blocks
        selected = select(_writer_ladder(limited), 13 * 2560 * 2, **conditions)
        assert selected["profiles"] == {JointMode.QUEUED_PUSH.name.lower(): expected}


def test_writer_progress_rejects_self_barriers_and_stale_queue_completion():
    from b12x.comm.pcie._joint_profile import JointMode, JOINT_MAX_RANKS, JOINT_QUEUE_FIRST_PHASE

    before = dict(epoch=(17, 0), barriers=(100, 200, 300, 400, *(0 for _ in range(JOINT_MAX_RANKS - 4))))
    after = dict(epoch=(18, 0), barriers=(100, 202, 302, 402, *(0 for _ in range(JOINT_MAX_RANKS - 4))))
    _assert_writer_progress(JointMode.SERIAL_PUSH, 0, 4, before, after)
    with pytest.raises(AssertionError):
        _assert_writer_progress(JointMode.SERIAL_PUSH, 0, 4, before,
            after | {"barriers": (102, *after["barriers"][1:])})
    with pytest.raises(AssertionError):
        _assert_writer_progress(JointMode.SERIAL_PUSH, 0, 4, before, after | {"barriers": before["barriers"]})
    token = (17 << (ctypes.sizeof(ctypes.c_uint32) * 8)) | (JOINT_QUEUE_FIRST_PHASE + 8)
    _assert_writer_progress(JointMode.QUEUED_PUSH, 0, 4, before, dict(epoch=(18, 0), queue_token=token))
    for invalid in (token - 1, token - (1 << (ctypes.sizeof(ctypes.c_uint32) * 8))):
        with pytest.raises(AssertionError):
            _assert_writer_progress(JointMode.QUEUED_PUSH, 0, 4, before, dict(epoch=(18, 0), queue_token=invalid))


@_gpu_only
def test_exported_tp4_collectives_replay_after_preparation_exit(tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign four CUDA devices explicitly"
    root = Path(__file__).resolve().parents[2]
    target = Path(os.environ.get("CARGO_TARGET_DIR", root / "helios" / "target"))
    subprocess.run(["cargo", "build", "-j", "4", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package,test-tools", "--bin", "test_fixed_runtime_package"],
        cwd=root, check=True, timeout=600)
    result = subprocess.run([sys.executable, str(root / "helios/tests/collective_package_fixture.py"),
        str(tmp_path)], cwd=root, text=True, capture_output=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all four preparation ranks exited" in result.stdout
    print(result.stdout, end="")
    # subprocess.run waits for the Python exporter; the Rust reader independently
    # rejects any still-live recorded preparation PID before initializing CUDA.
    result = subprocess.run([str(target / "debug/test_fixed_runtime_package"), str(tmp_path), "--tp4-collective"],
        cwd=root, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "native TP4 collective package replay passed after Python exit" in result.stdout
