"""Row-bounded mHC preserves coefficients and outputs through native graph replay."""

import os

import pytest
import torch
from b12x.testing.mhc import make_inputs, post_reference, pre_reference

from ignition.components.mhc import MHCScratch, MHCWeights, PreparedMHC
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")


@pytest.fixture(autouse=True)
def fp32_reference_precision():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("tf32_exact_weights", (False, True))
@pytest.mark.parametrize("norm_dtype", (torch.bfloat16, torch.float32))
@torch.no_grad()
def test_row_bounded_post_pre_alternates_native_capacities(tf32_exact_weights, norm_dtype):
    device = torch.device("cuda", 0)
    residual, branch, fn, scale, bias = make_inputs(
        tokens=8192, hidden_size=4096, seed=6811, device=device)
    if tf32_exact_weights:
        # Exact TF32 operands isolate accumulation from operand conversion while
        # retaining FP32 values that a BF16 weight copy would lose.
        fn.copy_(fn.view(torch.int32).bitwise_and(-8192).view(torch.float32))
        assert not torch.equal(fn, fn.bfloat16().float())
    norm = torch.linspace(.5, 1.5, 4096, device=device).to(norm_dtype)
    post = torch.ones((8192, 4), device=device)
    comb = torch.eye(4, device=device).expand(8192, -1, -1).contiguous()
    inputs = branch, residual, post, comb
    scratch = MHCScratch.allocate(8192, 4096, device)
    control = MHCScratch.allocate(8192, 4096, device)
    arena = ScratchArena(device)
    graphs, owners = {}, []
    options = dict(rms_eps=1e-6, norm_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20)
    with RecordedPreparationSession(device=device, checkpoint="synthetic-mhc",
            workload_schema="mhc.row-capacity.v1", numerical_policy="fp32-weights") as session:
        session.register_scratch(arena)
        variants = {}
        for rows in (8192, 4096, 2051, 2048, 8):
            args = tuple(value[:rows] for value in inputs)
            for name, buffers, limit in ((str(rows), scratch, 2048), (f"full{rows}", control, None)):
                operation = PreparedMHC(args, MHCWeights(fn, scale, bias, norm), buffers.rows(rows),
                    session=session, name=f"mhc{name}", workspace=arena,
                    max_rows_per_call=limit, **options)
                variants[name] = operation, args
        session.tune()
        for rows, (operation, args) in variants.items():
            def forward(operation=operation, args=args):
                operation(*args)
                # Temporary workspace may be reused immediately after all row calls finish.
                arena.tensor.fill_(165)

            compiled = compile_forward(forward)
            compiled()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            graphs[str(rows)] = graph
            owners.append(compiled)
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={"residual": residual},
                                     resources=(variants, inputs, scratch, control, arena, owners))
        worker = bundle.start_worker()
        try:
            for step, rows in enumerate((8192, 8, 4096, 2051, 2048, 8192)):
                residual.normal_(std=.1 * (step + 1))
                branch.normal_(std=.15)
                post.copy_(torch.where(torch.arange(8192, device=device)[:, None] % 2 == 0,
                                       .5, 2.).expand(-1, 4))
                comb.copy_(torch.eye(4, device=device).roll(step % 4, dims=0))
                expected_residual = post_reference(*(value[:rows] for value in inputs))
                collapsed, expected_post, expected_comb = pre_reference(
                    expected_residual, fn, scale, bias, rms_eps=options["rms_eps"],
                    hc_eps=options["hc_eps"], sinkhorn_iters=options["sinkhorn_iters"],
                    y_dtype=torch.float32)
                x = collapsed.float()
                expected_norm = (x.bfloat16().float() * torch.rsqrt(x.square().mean(-1, keepdim=True)
                                 + options["norm_eps"]) * norm.float()).bfloat16()
                torch.cuda.synchronize()
                control_ticket = worker.submit(bundle.graph_indices[f"full{rows}"])
                completion = worker.poll(timeout_ms=120000)
                assert (completion is not None and completion.ticket == control_ticket
                        and completion.error is None)
                for value in scratch.bind_args().values():
                    value.fill_(float("nan"))
                arena.tensor.fill_(90)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                ticket = worker.submit(bundle.graph_indices[str(rows)])
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                torch.testing.assert_close(scratch.residual[:rows], expected_residual, rtol=0, atol=0)
                operation = variants[str(rows)][0]
                baseline_backend = variants[f"full{rows}"][0].plan.selection.config.backend
                calls = operation.row_calls or ((0, rows, operation),)
                for start, end, call in calls:
                    # A short tail can select FP32 arithmetic while the complete
                    # matrix uses TF32. Check that tail against FP32 directly.
                    use_reference = tf32_exact_weights or call.plan.selection.config.backend != baseline_backend
                    post_oracle = expected_post if use_reference else control.post
                    comb_oracle = expected_comb if use_reference else control.comb
                    torch.testing.assert_close(scratch.post[start:end], post_oracle[start:end], rtol=2e-5, atol=4e-5)
                    torch.testing.assert_close(scratch.comb[start:end], comb_oracle[start:end], rtol=2e-5, atol=4e-5)
                    if call.plan.selection.config.backend == "native":
                        torch.testing.assert_close(scratch.normalized[start:end], expected_norm[start:end],
                                                   rtol=2e-5, atol=.008)
                    # Compare each slice separately so a short tail cannot hide
                    # behind the error budget of thousands of other rows.
                    for field, reference in (("post", expected_post), ("comb", expected_comb),
                                             ("normalized", expected_norm)):
                        expected = reference[start:end].float()
                        denominator = expected.norm().clamp_min(1e-20)
                        error = (getattr(scratch, field)[start:end].float() - expected).norm() / denominator
                        baseline = (getattr(control, field)[start:end].float() - expected).norm() / denominator
                        assert error <= baseline + 1e-5, (field, start, end, error.item(), baseline.item())
                assert torch.all(arena.tensor == 165)
                for value in scratch.bind_args().values():
                    assert value[rows:].isnan().all()
        finally:
            worker.close()
            for graph in graphs.values():
                graph.reset()


@pytest.mark.parametrize("operation", ("pre", "post_pre"))
@torch.no_grad()
def test_row_bounded_lagged_coefficients(operation):
    device = torch.device("cuda", 0)
    residual, branch, fn, scale, bias = make_inputs(
        tokens=17, hidden_size=5120, seed=6812, device=device)
    norm = torch.ones(5120, device=device, dtype=torch.bfloat16)
    post = torch.ones((17, 4), device=device)
    comb = torch.eye(4, device=device).expand(17, -1, -1).contiguous()
    inputs = (residual,) if operation == "pre" else (branch, residual, post, comb)
    incoming = torch.zeros((17, 4), device=device)
    predicted = torch.empty_like(incoming)
    scratch = MHCScratch.allocate(17, 5120, device)
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint="synthetic-mhc",
            workload_schema="mhc.lagged-row-capacity.v1", numerical_policy="fp32-weights") as session:
        session.register_scratch(arena)
        prepared = PreparedMHC(inputs, MHCWeights(fn, scale, bias, norm), scratch,
            session=session, name="lagged", workspace=arena, rms_eps=1e-6,
            norm_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20,
            pre_mix=incoming, pre_out=predicted, max_rows_per_call=8)
        session.tune()
        compiled = compile_forward(lambda: prepared(*inputs))
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        try:
            for shift in (0, 3):
                incoming.copy_(torch.eye(4, device=device)[(torch.arange(17, device=device) + shift) % 4])
                predicted.fill_(float("nan"))
                graph.replay()
                current = residual if operation == "pre" else post_reference(*inputs)
                flat = current.flatten(1).float()
                projected = torch.nn.functional.linear(flat, fn) * torch.rsqrt(
                    flat.square().mean(-1, keepdim=True) + 1e-6)
                expected_pre = torch.sigmoid(projected[:, :4] * scale[0] + bias[:4]) + 1e-6
                collapsed = (current.float() * incoming[:, :, None]).sum(1).bfloat16().float()
                expected = (collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True)
                            + 1e-6)).bfloat16()
                torch.testing.assert_close(predicted, expected_pre, rtol=2e-5, atol=4e-5)
                torch.testing.assert_close(scratch.normalized, expected, rtol=2e-5, atol=.008)
        finally:
            graph.reset()
