"""Captured GLM admission and immutable continuation agreement on TP2/TP4."""

import os
from datetime import timedelta

import pytest
import torch

from tests.components.completion_support import completion_outputs
import torch.distributed as dist
import torch.multiprocessing as mp


def _run(rank, world, rendezvous=None):
    from b12x.preparation import PreparationSession

    from ignition.components.maintenance import PreparedMaintenance as LocalMaintenance
    from ignition.components.sampling import (
        AdmissionBuffers,
        DeviceRequestState,
        RequestAdmission,
    )
    from ignition.components.state_transfer import PreparedStateTransfer, StateCommands
    from ignition.preparation.collectives import (
        CollectivePreparation,
        create_all_reduce_runtime,
    )
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.glm53.batch import BatchInputs
    from ignition.recipes.glm53.maintenance import PreparedMaintenance
    from ignition.recipes.glm53.pool import ContinuationPool

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    runtime, coordinator, worker, graphs = None, None, None, {}
    if world > 1:
        dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world, timeout=timedelta(minutes=3))
        coordinator = CollectivePreparation(dist.group.WORLD)
        runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=1, hidden_size=8)
    try:
        pool = ContinuationPool.allocate(("linear_attention", "indexed_attention"), max(world, 2), 8, 5, device)
        requests = DeviceRequestState.allocate(device, vocab_size=154880)
        inputs, commands = BatchInputs.allocate(1, 3, device), StateCommands.allocate(device)
        inputs.request_state_rows[:, 0] = torch.tensor([6, 3, 1, 4], device=device)
        admission = AdmissionBuffers.allocate(device, vocab_size=154880)
        records = torch.zeros((2, 4, 8), dtype=torch.int64, device=device)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.ones(1, dtype=torch.int64, device=device)
        operations = {}
        preparation = RecordedPreparationSession(device=device, checkpoint="glm-continuation-maintenance",
            workload_schema="glm-maintenance-v1", numerical_policy="default", coordinator=coordinator) if world > 1 else PreparationSession(device=device)
        with preparation as session:
            for mode in ("init", "restore", "snapshot"):
                if world > 1:
                    operations[mode] = PreparedMaintenance(mode, requests, pool, commands, inputs, completion_outputs(records), frame, sequence,
                        runtime=runtime, coordinator=coordinator, session=session, name=mode, admission_buffers=admission)
                else:
                    transfer = PreparedStateTransfer(pool, commands, inputs.page_tables,
                        pages=pool.partial_pages(), page_tokens=256, state_rows=inputs.request_state_rows, requests=requests)
                    slots, epochs, count = (inputs.row_slot_ids, inputs.row_generations, inputs.live_rows) if mode == "snapshot" else (
                        admission.admission_slots, admission.admission_generations, admission.admission_count)
                    local = LocalMaintenance(mode, requests, transfer, slots, epochs, count, completion_outputs(records), frame, sequence,
                                              world_size=1, admission_buffers=admission)
                    local.prepare()
                    def call(local=local):
                        local.validation_sum.copy_(local.validate())
                        local.apply()
                    operations[mode] = call
            if world > 1:
                session.tune()
            stream = torch.cuda.Stream(device=device)
            from contextlib import nullcontext
            for mode, operation in operations.items():
                torch.cuda.synchronize(device)
                stream.wait_stream(torch.cuda.current_stream(device))
                graph = torch.cuda.CUDAGraph()
                with (runtime.capture(stream=stream) if runtime else nullcontext()), session.capture(), torch.cuda.graph(graph, stream=stream):
                    operation()
                graphs[mode] = graph
            session.freeze()
            bundle = PreparedGraphBundle(device=rank, graphs=graphs, buffers={"state": pool.backing, "records": records},
                resources=(pool, requests, inputs, admission, commands, records, frame, sequence, operations, runtime, session, stream))
            worker = bundle.start_worker()

            def replay(mode):
                torch.cuda.synchronize(device)
                if world > 1:
                    dist.barrier()
                before = torch.cuda.memory_stats(device)
                ticket = worker.submit(bundle.graph_indices[mode])
                result = worker.poll(timeout_ms=120000)
                assert result is not None and result.ticket == ticket and result.error is None
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key], key

            def command(slot, source, destination, endpoint, operation=0):
                commands.request_slots[0] = slot
                commands.num_ops.fill_(1)
                commands.source_slots[0], commands.destination_slots[0] = source, destination
                commands.expected_consumed[0], commands.operation[0] = endpoint, operation

            inputs.page_tables.copy_(torch.tensor([[0, 1, 2], [0, 3, 2], [0, 4, 2], [0, 2, 1]], dtype=torch.int32))
            admission.admission_count.fill_(1)
            admission.stage(0, 0, RequestAdmission(1, 13, max_new_tokens=8), [5, 7])
            command(0, -1, 6, 0, 1)
            replay("init")
            assert not commands.errors.count_nonzero()
            assert requests.generation.tolist() == [1, 0, 0, 0]
            assert requests.histogram[0].sum() == 2 and not pool.backing.count_nonzero()

            pool.backing[6].fill_(0x30 + rank)
            requests.consumed_tokens[0] = 257
            for _, cache in pool.partial_pages():
                cache.fill_(0x70 + rank)
            inputs.row_slot_ids[0], inputs.row_generations[0] = 0, 1
            inputs.live_rows.fill_(1)
            command(0, 6, 2, 257)
            replay("snapshot")
            assert not commands.errors.count_nonzero()
            immutable = pool.backing[2].clone()
            assert records[0, 0, 2] == 257 and records[0, 0, 5] == 1
            admission.stage(0, 1, RequestAdmission(1, 99, max_new_tokens=8), [5] * 257)
            command(1, 2, 3, 257)
            replay("restore")
            assert not commands.errors.count_nonzero()
            assert torch.equal(pool.backing[3], immutable)
            assert requests.consumed_tokens[1] == 257 and requests.seed[1] == 99
            assert requests.histogram[1, 5] == 257 and requests.rng_counter[1] == 0
            for name, cache in pool.partial_pages():
                assert torch.equal(cache[3], pool.region(name)[2].reshape(-1))

            for fault in ("epoch", "endpoint", "page", "admission", "count"):
                admission.stage(0, 2, RequestAdmission(1, 37, max_new_tokens=8), [5] * 257)
                inputs.row_slot_ids[0], inputs.row_generations[0] = 0, 1
                mode = "snapshot" if fault in ("epoch", "count") else "restore"
                command(0, 6, 7, 257) if mode == "snapshot" else command(2, 2, 1, 257)
                if rank == world - 1:
                    if fault == "epoch":
                        requests.generation[0] = 2
                    elif fault == "endpoint":
                        pool.consumed_tokens[2] = 258
                    elif fault == "page":
                        inputs.page_tables[2, 1] = -1
                    elif fault == "admission":
                        admission.admission_parameters[0, 0] = -1
                    else:
                        commands.num_ops.fill_(0)
                before = pool.backing.clone()
                before_requests = {name: getattr(requests, name).clone() for name in requests.__dataclass_fields__}
                caches = {name: cache.clone() for name, cache in pool.partial_pages()}
                replay(mode)
                assert commands.errors[0] == 8
                record = records[0, 0 if mode == "snapshot" else 2]
                assert record[2:4].tolist() == [0, 0] and record[5:7].tolist() == [3, 72]
                assert torch.equal(pool.backing, before)
                for name, value in before_requests.items():
                    assert torch.equal(getattr(requests, name), value), name
                for name, cache in pool.partial_pages():
                    assert torch.equal(cache, caches[name]), name
                requests.generation[0] = 1
                pool.consumed_tokens[2] = 257
                inputs.page_tables[2, 1] = 4
            assert torch.equal(pool.backing[2], immutable)
            worker.close()
            worker = None
    finally:
        if worker is not None:
            worker.close()
        for graph in graphs.values():
            graph.reset()
        if runtime is not None:
            runtime.close()
            dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an explicitly assigned CUDA device")
def test_admission_and_continuation_native_replay():
    _run(0, 1)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1", reason="requires four assigned CUDA devices")
@pytest.mark.parametrize("world", [2, 4])
def test_all_ranks_reject_before_continuation_mutation(world, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") and torch.cuda.device_count() == 4
    mp.spawn(_run, args=(world, (tmp_path / "rendezvous").as_uri()), nprocs=world, join=True)
