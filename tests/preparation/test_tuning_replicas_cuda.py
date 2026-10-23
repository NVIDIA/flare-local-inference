"""Candidate sharding and graph replay through the shared b12x preparation path."""

import multiprocessing
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires two explicitly selected CUDA GPUs")


def _worker(directory, replica, connection):
    import torch
    from ignition.components.linear import UnquantizedProjection
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.preparation.tuning_replicas import TuningComplete, TuningReplica

    torch.cuda.set_device(replica)
    device = torch.device("cuda", replica)
    generator = torch.Generator().manual_seed(47)
    source = torch.randn((4, 768), generator=generator, dtype=torch.bfloat16).to(device)
    weight = torch.randn((512, 768), generator=generator, dtype=torch.bfloat16).to(device)
    output = torch.empty((4, 512), device=device, dtype=torch.bfloat16)
    role = TuningReplica(str(Path(directory) / "gloo"), replica, 2, 1)
    session = RecordedPreparationSession(device=device, checkpoint="replica-cuda-test",
        workload_schema="projection", numerical_policy="bf16", compile_workers=4,
        namespace={"case": directory}, tuning_replica=role)
    projection = UnquantizedProjection(source, weight, output, session=session, name="projection")
    try:
        session.tune()
    except TuningComplete:
        assert replica == 1
    else:
        assert replica == 0
        expected = source.float() @ weight.float().T
        projection(source)
        torch.testing.assert_close(output.float(), expected, rtol=0.01, atol=0.25)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            projection(source)
        for _ in range(4):
            output.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(output.float(), expected, rtol=0.01, atol=0.25)
    report = session.tuning_report
    connection.send({key: report[key] for key in ("compile_workers", "benchmarked_candidates",
                     "parent_cute_compilations", "parent_triton_compilations", "selections")})
    connection.close()
    session.close()


def test_two_gpus_share_candidates_and_replay_the_selected_plan(tmp_path):
    context = multiprocessing.get_context("spawn")
    processes, connections = [], []
    try:
        for replica in range(2):
            root, peer = context.Pipe(duplex=False)
            process = context.Process(target=_worker, args=(str(tmp_path), replica, peer))
            process.start()
            peer.close()
            processes.append(process)
            connections.append(root)
        for process in processes:
            process.join(timeout=180)
        assert [process.exitcode for process in processes] == [0, 0]
        reports = [connection.recv() for connection in connections]
        assert all(report["compile_workers"] == 2 for report in reports)
        assert all(report["benchmarked_candidates"] > 0 for report in reports)
        assert all(report["parent_cute_compilations"] == report["parent_triton_compilations"] == 0
                   for report in reports)
        assert reports[0]["selections"][0]["config"] == reports[1]["selections"][0]["config"]
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
        for connection in connections:
            connection.close()


def _tp_worker(directory, member, connection, replicas=2, tensor_parallel=2, checkpoint_stage=None):
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    from ignition.components.collectives import PreparedAllReduce
    from ignition.components.linear import UnquantizedProjection
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.preparation.tuning_replicas import TuningComplete, TuningReplica
    from b12x.preparation.session import PreparationJob

    replica, rank = divmod(member, tensor_parallel)
    torch.cuda.set_device(member)
    device = torch.device('cuda', member)
    rendezvous = f'tp-{replica}' if checkpoint_stage is None else f'tp-{replica}-{checkpoint_stage}'
    dist.init_process_group('gloo', init_method=(Path(directory) / rendezvous).as_uri(),
                            rank=rank, world_size=tensor_parallel, timeout=timedelta(seconds=90))
    coordinator = CollectivePreparation(dist.group.WORLD)
    runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=4, hidden_size=768)
    role = (TuningReplica(str(Path(directory) / 'replicas'), replica, replicas, tensor_parallel)
            if replicas > 1 else None)
    session = RecordedPreparationSession(device=device, checkpoint='replica-tp-test',
        workload_schema='projection-reduction', numerical_policy='bf16', compile_workers=4,
        namespace={'case': directory}, tuning_replica=role, coordinator=coordinator)
    if checkpoint_stage is not None:
        session.cache_dir = Path(directory) / 'selections'
    if checkpoint_stage == 'interrupt':
        import ignition.preparation.collectives as collectives

        def interrupt_consolidation(contributions):
            # Every rank has completed its local race; no final winner is installed.
            os._exit(23)

        collectives._winners = interrupt_consolidation
    source = torch.full((4, 768), rank + 1, device=device, dtype=torch.bfloat16)
    output = torch.empty_like(source)
    projection = None
    if rank == 0 or replicas == 1:
        weight = torch.ones((512, 768), device=device, dtype=torch.bfloat16)
        projected = torch.empty((4, 512), device=device, dtype=torch.bfloat16)
        projection = UnquantizedProjection(source, weight, projected, session=session, name='projection')
    reduction = PreparedAllReduce(source, output, runtime=runtime, session=session,
                                  coordinator=coordinator, name='reduce')
    candidates = []
    trial = PreparationJob._trial

    def record_trial(job, obligation, index, assignment, config):
        candidates.append((obligation.request.name, index))
        return (yield from trial(job, obligation, index, assignment, config))

    PreparationJob._trial = record_trial
    try:
        session.tune()
    except TuningComplete:
        assert replica == 1
    else:
        assert replica == 0
        reduction(source)
        expected_sum = tensor_parallel * (tensor_parallel + 1) // 2
        torch.testing.assert_close(output, torch.full_like(output, expected_sum), rtol=0, atol=0)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
            reduction(source)
            if projection is not None:
                projection(source)
        for _ in range(4):
            output.fill_(float('nan'))
            dist.barrier()
            graph.replay()
            torch.testing.assert_close(output, torch.full_like(output, expected_sum), rtol=0, atol=0)
            if projection is not None:
                torch.testing.assert_close(projected, torch.full_like(projected, 768 * (rank + 1)), rtol=0, atol=0)
    connection.send({'selections': session.tuning_report['selections'], 'candidates': candidates,
                     'restored': session.tuning_report['restored_candidates'],
                     'measured': session.tuning_report['benchmarked_candidates'],
                     'shard': (session._tuning_rank, session._tuning_ranks)})
    connection.close()
    dist.barrier()
    session.close()
    runtime.close()
    dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get('FLARE_RUN_TP_CUDA_TESTS') != '1',
                    reason='requires four explicitly selected CUDA GPUs')
def test_tp_replicas_preserve_uneven_rank_work_and_collectives(tmp_path):
    context = multiprocessing.get_context('spawn')
    processes, connections = [], []
    try:
        for member in range(4):
            root, peer = context.Pipe(duplex=False)
            process = context.Process(target=_tp_worker, args=(str(tmp_path), member, peer))
            process.start()
            peer.close()
            processes.append(process)
            connections.append(root)
        for process in processes:
            process.join(timeout=180)
        assert [process.exitcode for process in processes] == [0] * 4
        selections = [connection.recv()['selections'] for connection in connections]
        for rank in range(2):
            assert [(item['name'], item['config']) for item in selections[rank]] == [
                (item['name'], item['config']) for item in selections[rank + 2]]
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
        for connection in connections:
            connection.close()


@pytest.mark.skipif(os.environ.get('FLARE_RUN_TP_CUDA_TESTS') != '1',
                    reason='requires four explicitly selected CUDA GPUs')
def test_local_tp_ranks_measure_disjoint_candidates_and_replay_collectives(tmp_path):
    context = multiprocessing.get_context('spawn')
    processes, connections = [], []
    try:
        for rank in range(4):
            root, peer = context.Pipe(duplex=False)
            process = context.Process(target=_tp_worker, args=(str(tmp_path), rank, peer, 1, 4))
            process.start()
            peer.close()
            processes.append(process)
            connections.append(root)
        for process in processes:
            process.join(timeout=180)
        assert [process.exitcode for process in processes] == [0] * 4
        reports = [connection.recv() for connection in connections]
        all_candidates = [item for report in reports for item in report['candidates']]
        assert len(set(all_candidates)) == len(all_candidates)
        selections = []
        for rank, report in enumerate(reports):
            assert report['shard'] == (rank, (0, 1, 2, 3))
            assert report['candidates']
            assert all(index % 4 == rank for _, index in report['candidates'])
            selected = next(item for item in report['selections'] if item['name'] == 'projection')
            assert selected['source'] == 'tuned'
            assert selected['coverage']['measured_count'] == len(all_candidates)
            assert selected['coverage']['effective_count'] == len(all_candidates)
            selections.append(selected['config'])
        assert all(config == selections[0] for config in selections)
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
        for connection in connections:
            connection.close()


@pytest.mark.skipif(os.environ.get('FLARE_RUN_TP_CUDA_TESTS') != '1',
                    reason='requires four explicitly selected CUDA GPUs')
def test_local_tp_restart_restores_races_before_consolidation(tmp_path):
    import json
    import sqlite3

    context = multiprocessing.get_context('spawn')
    for stage in ('interrupt', 'resume'):
        processes, connections = [], []
        try:
            for rank in range(4):
                root, peer = context.Pipe(duplex=False)
                process = context.Process(target=_tp_worker,
                    args=(str(tmp_path), rank, peer, 1, 4, stage))
                process.start()
                peer.close()
                processes.append(process)
                connections.append(root)
            for process in processes:
                process.join(timeout=180)
            assert [process.exitcode for process in processes] == [23 if stage == 'interrupt' else 0] * 4
            if stage == 'interrupt':
                files = list((tmp_path / 'selections').glob('*.progress.sqlite3'))
                assert len(files) == 1
                with sqlite3.connect(files[0]) as connection:
                    rows = connection.execute('SELECT payload FROM races_v1').fetchall()
                assert len(rows) == 4
                measured = sum(len(json.loads(row[0])['measurements']) for row in rows)
                assert not list((tmp_path / 'selections').glob('*.json'))
            else:
                reports = [connection.recv() for connection in connections]
                assert sum(report['restored'] for report in reports) == measured
                assert all(report['restored'] > 0 and report['measured'] == 0
                           and not report['candidates'] for report in reports)
                selected = [next(item for item in report['selections'] if item['name'] == 'projection')
                            for report in reports]
                assert all(item['config'] == selected[0]['config'] for item in selected)
                assert selected[0]['coverage']['measured_count'] == measured
        finally:
            for process in processes:
                if process.is_alive():
                    process.kill()
                process.join()
            for connection in connections:
                connection.close()


def _partial_race_worker(directory, interrupt, connection):
    import torch
    from b12x.preparation import autotune_effort
    from b12x.preparation.session import PreparationJob
    from ignition.components.linear import UnquantizedProjection
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(0)
    torch.manual_seed(415125)
    source = torch.randn((224, 5376), device='cuda', dtype=torch.bfloat16)
    weight = (torch.randn((1, 5376), device='cuda') / 5376**0.5).bfloat16()
    output = torch.empty((224, 1), device='cuda', dtype=torch.float32)
    measure, trial = PreparationJob._measure, PreparationJob._trial
    candidates = []

    def interrupt_batch(job, trials, **kwargs):
        if interrupt and job._batch_index == 2:
            os._exit(23)
        return (yield from measure(job, trials, **kwargs))

    def record_trial(job, obligation, index, assignment, config):
        candidates.append(index)
        return (yield from trial(job, obligation, index, assignment, config))

    PreparationJob._measure, PreparationJob._trial = interrupt_batch, record_trial
    with autotune_effort('high'), RecordedPreparationSession(device='cuda:0',
            checkpoint='confidence-tuning-restart', workload_schema='confidence', numerical_policy='fp32',
            namespace={'case': directory}, compile_workers=2, autotune_effort='high') as session:
        session.cache_dir = Path(directory) / 'selections'
        session.race_batch = 2
        projection = UnquantizedProjection(source, weight, output, session=session,
                                            name='split.proposal32x8.confidence')
        session.tune()
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture(), torch.cuda.graph(graph):
                projection(source)
            source.mul_(0.5)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output.double(), source.double() @ weight.double().T,
                                       rtol=1e-6, atol=1e-6)
        finally:
            graph.reset()
        connection.send({'candidates': candidates, 'report': session.tuning_report})
    connection.close()


def test_confidence_projection_resumes_after_a_completed_batch(tmp_path):
    import json
    import sqlite3

    context = multiprocessing.get_context('spawn')
    for interrupt in (True, False):
        root, peer = context.Pipe(duplex=False)
        process = context.Process(target=_partial_race_worker, args=(str(tmp_path), interrupt, peer))
        process.start()
        peer.close()
        try:
            process.join(timeout=180)
            assert process.exitcode == (23 if interrupt else 0)
            if interrupt:
                (path,) = (tmp_path / 'selections').glob('*.progress.sqlite3')
                with sqlite3.connect(path) as connection:
                    ((payload,),) = connection.execute('SELECT payload FROM races_v1').fetchall()
                checkpoint = json.loads(payload)
                assert {index for index, _ in checkpoint['measurements']} == {0, 1}
                assert not list((tmp_path / 'selections').glob('*.json'))
            else:
                result = root.recv()
                assert result['report']['restored_candidates'] == 2
                assert result['report']['benchmarked_candidates'] == 9
                assert result['candidates'][0] == checkpoint['champion']
                assert result['candidates'][1:] == list(range(2, 11))
        finally:
            if process.is_alive():
                process.kill()
                process.join()
            root.close()
