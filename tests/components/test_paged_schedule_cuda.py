"""Shared schedule lifetimes across compiled attention and numerical scratch reuse."""

import copy
import os

import pytest
import torch

from ignition.components.paged_attention import PreparedPagedAttention
from ignition.components.paged_schedule import PreparedPagedSchedules
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU"
)


@torch.no_grad()
def test_compiled_schedule_groups_survive_mutable_inputs_and_scratch_reuse():
    device = torch.device("cuda:0")
    torch.manual_seed(637)
    batch, width, rows, page, context = 3, 6, 18, 128, 4099
    physical = (context + page - 1) // page
    key = (
        torch.randn(1 + batch * physical, page, 1, 128, device=device) * 0.2
    ).bfloat16()
    value = torch.randn_like(key) * 0.2
    table = torch.zeros((batch, 64), dtype=torch.int32, device=device)
    for request in range(batch):
        table[request, :physical] = torch.arange(
            1 + request * physical, 1 + (request + 1) * physical, device=device
        )
    lengths = torch.full((batch,), context, dtype=torch.int32, device=device)
    cu = torch.arange(batch + 1, dtype=torch.int32, device=device) * width
    selected = torch.full((1, rows, 16), -1, dtype=torch.int32, device=device)
    key64, value64 = key.view(-1, 64, 1, 128), value.view(-1, 64, 1, 128)
    table64 = torch.zeros((batch, 128), dtype=torch.int32, device=device)
    for request in range(batch):
        table64[request, : physical * 2] = torch.arange(
            2 + request * physical * 2, 2 + (request + 1) * physical * 2, device=device
        )
    queries = [
        (torch.randn(rows, 16, 128, device=device) * 0.2).bfloat16() for _ in range(3)
    ]
    outputs = [torch.empty_like(q) for q in queries]
    buffers, arena = GraphBufferPool(device), ScratchArena(device, overlap=False)
    with RecordedPreparationSession(
        device=device,
        checkpoint="shared-paged-schedules",
        workload_schema="shared-paged-schedules",
        numerical_policy="default",
        compile_workers=4,
    ) as session:
        session.register_scratch(arena)
        schedules = PreparedPagedSchedules(
            lengths, cu, len(queries), buffers=buffers.cursor()
        )
        operations = [
            PreparedPagedAttention(
                q,
                key if index < 2 else key64,
                value if index < 2 else value64,
                table if index < 2 else table64,
                lengths,
                cu,
                output,
                selected_blocks=selected,
                mode="verify",
                width=width,
                session=session,
                workspace=arena,
                name=f"attention{index}",
                schedules=schedules,
            )
            for index, (q, output) in enumerate(zip(queries, outputs, strict=True))
        ]
        schedules.finalize(session)
        session.tune()
        keys = [schedule.key for schedule in schedules.groups]
        assert keys[0] != keys[2]
        assert [operation.schedule_slot for operation in operations] == [
            keys.index(key) for key in keys
        ]
        assert (
            schedules.storage.untyped_storage().data_ptr()
            != arena.tensor.untyped_storage().data_ptr()
        )

        def forward():
            schedules()
            for operation in operations:
                arena.tensor.fill_(197)
                operation()

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for turn, counts in enumerate(([6, 6, 6], [1, 0, 3], [0, 0, 0], [6, 2, 6])):
            caches = [context - turn * 127 if count else 0 for count in counts]
            lengths.copy_(torch.tensor(caches, dtype=torch.int32, device=device))
            boundaries = [0]
            for count in counts:
                boundaries.append(boundaries[-1] + count)
            cu.copy_(torch.tensor(boundaries, dtype=torch.int32, device=device))
            selected.fill_(-1)
            expected = [torch.empty_like(output) for output in outputs]
            for request, count in enumerate(counts):
                for local in range(count):
                    row = boundaries[request] + local
                    position = caches[request] - count + local
                    blocks = [0, 1, position // page]
                    selected[0, row, :3] = torch.tensor(
                        blocks, dtype=torch.int32, device=device
                    )
                    tokens = (
                        torch.tensor(blocks, device=device)[:, None] * page
                        + torch.arange(page, device=device)
                    ).flatten()
                    tokens = tokens[tokens <= position]
                    pages = table[request, tokens // page].long()
                    k, v = (
                        key[pages, tokens % page, 0].float(),
                        value[pages, tokens % page, 0].float(),
                    )
                    for q, result in zip(queries, expected, strict=True):
                        scores = q[row].float() @ k.T / (128**0.5)
                        result[row] = (scores.softmax(-1) @ v).bfloat16()
            schedules.storage.fill_(197)
            for output in outputs:
                output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert (
                torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
            )
            for output, reference in zip(outputs, expected, strict=True):
                torch.testing.assert_close(
                    output[: boundaries[-1]],
                    reference[: boundaries[-1]],
                    rtol=0.025,
                    atol=0.0003,
                )
                if boundaries[-1] == 0:
                    assert torch.isnan(output).all()
        graph.reset()


@torch.no_grad()
@pytest.mark.parametrize("shared", [False, True], ids=["independent", "shared"])
@pytest.mark.parametrize("empty", [True, False], ids=["empty", "populated"])
def test_ordered_verification_and_prefill_graphs_rebuild_aliased_schedules(
    shared, empty
):
    device = torch.device("cuda:0")
    context = 4099
    torch.manual_seed(639)
    key = (torch.randn(34, 128, 1, 128, device=device) * 0.2).to(torch.float8_e4m3fn)
    value = (torch.randn_like(key.float()) * 0.2).to(key.dtype)
    table = torch.arange(1, 34, dtype=torch.int32, device=device)[None].contiguous()
    buffers, arena = GraphBufferPool(device), ScratchArena(device, overlap=False)
    variants = []
    with RecordedPreparationSession(
        device=device,
        checkpoint="ordered-paged-schedules",
        workload_schema="ordered-paged-schedules",
        numerical_policy="default",
        compile_workers=4,
    ) as session:
        session.register_scratch(arena)
        for mode, rows in (("extend", 4096), ("extend", 33), ("verify", 6)):
            lengths = torch.tensor([context], dtype=torch.int32, device=device)
            cu = torch.tensor([0, rows], dtype=torch.int32, device=device)
            selected = torch.full((1, rows, 16), -1, dtype=torch.int32, device=device)
            if not empty:
                last = (torch.arange(rows, device=device) + context - rows) // 128
                selected[0, :, 0] = 0
                selected[0, :, 1] = torch.where(last >= 1, 1, -1)
                selected[0, :, 2] = torch.where(last >= 2, last, -1)
            query = (torch.randn(rows, 16, 128, device=device) * 0.2).bfloat16()
            output = torch.empty_like(query)
            schedules = PreparedPagedSchedules(lengths, cu, 1, buffers=buffers.cursor())
            operation = PreparedPagedAttention(
                query,
                key,
                value,
                table,
                lengths,
                cu,
                output,
                selected_blocks=selected,
                mode=mode,
                width=rows if mode == "verify" else 1,
                session=session,
                workspace=arena,
                name=f"attention.{mode}{rows}",
                schedules=schedules,
            )
            schedules.finalize(session)
            variants.append((operation, schedules, lengths, cu, selected))
        session.tune()
        assert (
            len({item[1].storage.untyped_storage().data_ptr() for item in variants})
            == 1
        )
        assert variants[-1][0].schedule_slot == 0
        compiled, graphs, controls = [], [], []
        for operation, schedules, *_ in variants:
            if not shared:
                operation.schedule_slot = -1

            def forward(operation=operation, schedules=schedules):
                if shared:
                    schedules()
                operation()

            arena.tensor.fill_(0x55)
            compiled.append(compile_forward(forward))
            try:
                torch.cuda.synchronize()
            except RuntimeError as error:
                error.add_note(
                    f"attention rows={operation.arguments['q'].shape[0]}, shared={shared}, "
                    f"empty={empty}, schedule_slot={operation.schedule_slot}"
                )
                raise
            if empty:
                assert not operation.arguments["output"].count_nonzero()
            control = copy.copy(operation)
            control.schedule_slot = -1
            control.arguments = dict(
                operation.arguments,
                output=torch.empty_like(operation.arguments["output"]),
            )
            controls.append(control)
        for forward in compiled:
            forward()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                forward()
            graphs.append(graph)
        session.freeze()
        for index, count in ((0, 4096), (2, 0), (1, 33), (0, 17), (2, 6), (0, 4096)):
            operation, schedules, lengths, cu, selected = variants[index]
            lengths.fill_(context if count else 0)
            cu.copy_(torch.tensor([0, count], dtype=torch.int32, device=device))
            positions = torch.arange(selected.shape[1], device=device) + context - count
            last = positions // 128
            live = torch.arange(selected.shape[1], device=device) < count
            selected.fill_(-1)
            selected[0, :, 0] = torch.where(live, 0, -1)
            selected[0, :, 1] = torch.where(live & (last >= 1), 1, -1)
            selected[0, :, 2] = torch.where(live & (last >= 2), last, -1)
            schedules.storage.fill_(197)
            operation.arguments["output"].fill_(float("nan"))
            controls[index].arguments["output"].fill_(float("nan"))
            arena.tensor.fill_(0x55)
            allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graphs[index].replay()
            torch.cuda.synchronize()
            assert (
                torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
            )
            arena.tensor.fill_(0x55)
            controls[index]()
            torch.testing.assert_close(
                operation.arguments["output"],
                controls[index].arguments["output"],
                rtol=0,
                atol=0,
                equal_nan=True,
            )
        for graph in graphs:
            graph.reset()
