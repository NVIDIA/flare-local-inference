"""Ordering of writes through typed aliases in captured inference forwards."""

import os
import subprocess
import sys
from collections import namedtuple

import pytest
import torch
import triton
import triton.language as tl

from ignition.preparation.compilation import ForwardTracer, compile_forward

ForwardOutput = namedtuple("ForwardOutput", ("hidden", "auxiliary"))


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@torch.no_grad()
def test_compiled_gather_reads_only_backed_state_rows():
    import ctypes as C
    from b12x.sequence._shared.disk_table import _tensor_from_pointer
    from ignition.components.state import gather_rows
    from ignition.preparation.package_storage import DeviceRegion
    from ignition.preparation.runtime_package import _Cuda

    torch.cuda.init()
    torch.empty(0, device="cuda")
    cuda = _Cuda()
    address = C.c_uint64()
    row_bytes, rows, backed = 2 << 20, 8, 2
    cuda.call("cuMemAddressReserve", C.byref(address), C.c_size_t(rows * row_bytes),
              C.c_size_t(row_bytes), C.c_uint64(0), C.c_uint64(0))
    region = DeviceRegion(cuda, address.value, backed * row_bytes, torch.cuda.current_device())
    try:
        storage = _tensor_from_pointer(address.value, shape=(rows, row_bytes), dtype=torch.uint8,
            device=torch.device("cuda", torch.cuda.current_device()), nbytes=rows * row_bytes)
        selection = torch.tensor([1, 0], dtype=torch.int64, device="cuda")
        result = torch.empty((2, 64), dtype=torch.float32, device="cuda")

        def forward():
            source = storage[:, 256:512].view(torch.float32)
            gather_rows(source, selection, result)

        storage[:backed, 256:512].view(torch.float32).copy_(torch.arange(128, device="cuda").view(2, 64))
        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        for indices in ([1, 0], [0, 1]):
            selection.copy_(torch.tensor(indices, device="cuda"))
            graph.replay()
            torch.testing.assert_close(result, torch.arange(128, device="cuda", dtype=torch.float32).view(2, 64)[indices])
        del graph, compiled, storage
    finally:
        torch.cuda.synchronize()
        cuda.call("cuMemUnmap", address, C.c_size_t(region.nbytes))
        cuda.call("cuMemAddressFree", address, C.c_size_t(rows * row_bytes))


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@torch.no_grad()
def test_shared_tracer_preserves_shape_specific_aliases_and_replay():
    source = torch.randn(8, 64, device="cuda")
    weight = torch.randn(64, device="cuda")
    scratch = torch.empty_like(source)
    output = torch.empty_like(source)
    tracer = ForwardTracer()
    graphs = []
    compiled = []
    for rows in (8, 1, 4):
        def forward(rows=rows):
            scratch[:rows].copy_(source[:rows] * weight)
            output[:rows].copy_(scratch[:rows].sum(dim=1, keepdim=True))
            return output[:rows]

        function = compile_forward(forward, tracer=tracer)
        compiled.append(function)
        function()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            function()
        graphs.append((rows, graph))
    del tracer
    for rows, graph in reversed(graphs):
        source.normal_()
        weight.normal_()
        output.fill_(float('nan'))
        graph.replay()
        expected = (source[:rows] * weight).sum(dim=1, keepdim=True).expand(rows, 64)
        torch.testing.assert_close(output[:rows], expected)
        assert output[rows:].isnan().all()


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@torch.no_grad()
def test_compiled_named_output_preserves_tensor_aliases():
    source = torch.ones(64, device="cuda")
    output = torch.empty_like(source)

    def forward():
        output.copy_(source * 3)
        return ForwardOutput(output, {"input": source, "optional": None})

    compiled = compile_forward(forward)
    result = compiled()
    assert isinstance(result, ForwardOutput)
    assert result.hidden.data_ptr() == output.data_ptr()
    assert result.auxiliary["input"].data_ptr() == source.data_ptr()
    assert result.auxiliary["optional"] is None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    source.fill_(7)
    graph.replay()
    torch.testing.assert_close(result.hidden, torch.full_like(output, 21), rtol=0, atol=0)


@triton.jit
def _cached_fill_triton(out, value: tl.constexpr):
    offsets = tl.arange(0, 64)
    tl.store(out + offsets, value)


@triton.jit
def _cached_negate_triton(out, value: tl.constexpr):
    offsets = tl.arange(0, 64)
    tl.store(out + offsets, -value)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@torch.no_grad()
def test_shared_storage_write_order():
    storage = torch.zeros(4096, dtype=torch.uint8, device="cuda")
    fp32 = storage.view(torch.float32)
    bf16 = storage.view(torch.bfloat16)
    fp32_offset = fp32[32:96]
    before = torch.empty_like(bf16)
    middle = torch.empty_like(bf16)
    after = torch.empty_like(fp32)

    def forward():
        with torch.profiler.record_function("shared_storage_initialization"):
            fp32.fill_(1.25)
            before.copy_(bf16)
        fp32_offset.fill_(2.5)
        middle.copy_(bf16)
        bf16[4:20].fill_(0.5)
        after.copy_(fp32)

    forward()
    expected = [tensor.clone() for tensor in (storage, before, middle, after)]
    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for value in (0, 19, 128):
        storage.fill_(value)
        graph.replay()
        for tensor, reference in zip((storage, before, middle, after), expected, strict=True):
            torch.testing.assert_close(tensor, reference, rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@pytest.mark.parametrize("rows", [1, 512])
@torch.no_grad()
def test_compiled_bf16_gate_rounding(rows):
    logits = torch.randn(rows, 1, dtype=torch.bfloat16, device="cuda")
    projected = torch.randn(rows, 2560, dtype=torch.bfloat16, device="cuda")
    probabilities = torch.empty_like(logits)
    output = torch.empty_like(projected)

    def forward():
        torch.sigmoid(logits, out=probabilities)
        torch.mul(projected, probabilities, out=output)

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for _ in range(3):
        logits.normal_()
        projected.normal_()
        forward()
        expected = output.clone()
        expected_probabilities = probabilities.clone()
        output.fill_(float("nan"))
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(probabilities, expected_probabilities, rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@torch.no_grad()
def test_bf16_add_out_preserves_pooled_aliases_under_native_replay():
    from ignition.preparation.graphs import PreparedGraphBundle

    storage = torch.empty((4, 4096, 5120), dtype=torch.bfloat16, device="cuda")
    routed, shared, partial, consumed = storage.unbind(0)
    graphs, owners = {}, []
    capacities = (1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 128, 4096)
    for rows in capacities:
        def forward(rows=rows):
            torch.add(routed[:rows], shared[:rows], out=partial[:rows])
            consumed[:rows].copy_(partial[:rows])
            routed[:rows].zero_()

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        graphs[str(rows)] = graph
        owners.append(compiled)
    bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={"storage": storage},
                                 resources=(storage, owners))
    worker = bundle.start_worker()
    try:
        for rows in (*capacities, *reversed(capacities)):
            storage.fill_(float("nan"))
            routed[:rows].normal_()
            shared[:rows].normal_()
            shared[:rows, ::2].copy_(-routed[:rows, ::2])
            expected = (routed[:rows].float() + shared[:rows].float()).bfloat16()
            shared_before = shared[:rows].clone()
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            ticket = worker.submit(bundle.graph_indices[str(rows)])
            result = worker.poll(timeout_ms=120000)
            assert result is not None and result.ticket == ticket and result.error is None
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed",
                        "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key]
            assert torch.equal(partial[:rows], expected)
            assert torch.equal(consumed[:rows], expected)
            assert torch.equal(shared[:rows], shared_before)
            assert routed[:rows].count_nonzero().item() == 0
            assert storage[:, rows:].isnan().all()
    finally:
        worker.close()
        for graph in graphs.values():
            graph.reset()


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@torch.no_grad()
def test_fp32_sigmoid_preserves_bf16_gate_rounding():
    gate = torch.tensor([[-0.0296630859375]], dtype=torch.bfloat16, device="cuda")
    normalized = torch.tensor([[-0.23046875]], dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(normalized)

    def forward():
        output.copy_((normalized * torch.sigmoid(gate.float())).bfloat16())

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for value in (-0.0296630859375, 0., 1.):
        gate.fill_(value)
        forward()
        expected = output.clone()
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@pytest.mark.parametrize("approximate", ["none", "tanh"])
@torch.no_grad()
def test_gelu_preserves_aten_rounding(approximate):
    source = torch.linspace(-6, 6, 32768, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(source)

    def forward():
        output.copy_(torch.nn.functional.gelu(source, approximate=approximate))

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for shift in (0., 0.125, -0.25):
        source.add_(shift)
        expected = torch.nn.functional.gelu(source, approximate=approximate)
        output.fill_(float("nan"))
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    graph.reset()


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@pytest.mark.parametrize("rows", [1, 4096])
@torch.no_grad()
def test_layer_norm_preserves_aten_rounding(rows):
    source = torch.zeros((rows, 128), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(128, dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros_like(weight)
    output = torch.empty_like(source)

    def forward():
        output.copy_(torch.nn.functional.layer_norm(source, (128,), weight, bias, 1e-6))

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    generator = torch.Generator(device="cuda").manual_seed(532128)
    for _ in range(3):
        source.normal_(generator=generator)
        weight.normal_(generator=generator)
        bias.normal_(generator=generator)
        forward()
        expected = output.clone()
        output.fill_(float("nan"))
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    graph.reset()


@torch.library.custom_op("test_compilation::fill_output", mutates_args=("out",))
def _fill_output(out: torch.Tensor, value: float) -> None:
    out.fill_(value)


@_fill_output.register_fake
def _fake_fill_output(out, value):
    pass


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@torch.no_grad()
def test_cloned_value_survives_scratch_overwrite():
    storage = torch.empty((4, 64), dtype=torch.bfloat16, device="cuda")
    view = storage.view(-1).view_as(storage)

    def forward():
        _fill_output(storage, 2.)
        saved = view.clone()
        _fill_output(storage, 3.)
        return saved, storage

    compiled = compile_forward(forward)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        saved, final = compiled()
    for initial in (0., -5., 19.):
        storage.fill_(initial)
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        torch.testing.assert_close(saved, torch.full_like(saved, 2.), atol=0, rtol=0)
        torch.testing.assert_close(final, torch.full_like(final, 3.), atol=0, rtol=0)
    graph.reset()


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@pytest.mark.parametrize("eps", [None, 1e-5])
@torch.no_grad()
def test_rms_norm_preserves_aten_rounding(eps):
    generator = torch.Generator(device="cuda").manual_seed(41)
    source = torch.randn((256, 1024), dtype=torch.bfloat16, device="cuda", generator=generator)
    norm = torch.nn.RMSNorm(1024, eps=eps, dtype=torch.bfloat16, device="cuda").requires_grad_(False)
    norm.weight.normal_(generator=generator)
    compiled = compile_forward(lambda: norm(source))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = compiled()
    for scale in (1., .01, 8.):
        source.normal_(std=scale, generator=generator)
        expected = norm(source)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    graph.reset()


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@torch.no_grad()
def test_compilation_cache_preserves_warmup_mutations(tmp_path, monkeypatch):
    monkeypatch.setattr("ignition.preparation.compilation.DEFAULT_CACHE_ROOT", tmp_path)

    for cache_hit in (False, True):
        counter = torch.zeros(64, device="cuda")
        compiled = compile_forward(lambda: _increment_output(counter))
        assert compiled.cache_hit == cache_hit
        torch.testing.assert_close(counter, torch.ones_like(counter), rtol=0, atol=0)
        compiled()
        torch.testing.assert_close(counter, torch.full_like(counter, 2), rtol=0, atol=0)


@torch.library.custom_op("test_compilation::increment_output", mutates_args=("out",))
def _increment_output(out: torch.Tensor) -> None:
    out.add_(1)


@_increment_output.register_fake
def _fake_increment_output(out):
    pass


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@torch.no_grad()
def test_compilation_cache_binds_each_invocation_and_buffer(tmp_path, monkeypatch):
    monkeypatch.setattr("ignition.preparation.compilation.DEFAULT_CACHE_ROOT", tmp_path / "compiled")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))

    def prepare(value):
        source = torch.empty(64, device="cuda")
        output = torch.empty_like(source)

        def forward():
            _fill_output(source, value)
            output.copy_(source + 1)

        return compile_forward(forward), output

    first, first_output = prepare(2.)
    second, second_output = prepare(8.)
    assert not first.cache_hit
    assert second.cache_hit
    for compiled, output, value in ((first, first_output, 3.), (second, second_output, 9.)):
        compiled()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        output.zero_()
        graph.replay()
        torch.testing.assert_close(output, torch.full_like(output, value), rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@pytest.mark.parametrize("native_kind", ["custom", "triton"])
def test_compilation_cache_survives_process_restart(tmp_path, monkeypatch, native_kind):
    program = '''
import sys
from unittest.mock import patch
import torch
from ignition.preparation.compilation import compile_forward
import ignition.preparation.compilation as compilation
from pathlib import Path
compilation.DEFAULT_CACHE_ROOT = Path(sys.argv[4])

@torch.library.custom_op("flare_cache_test::fill", mutates_args=("out",))
def fill(out: torch.Tensor, value: float) -> None:
    out.fill_(value)
@fill.register_fake
def fake_fill(out, value):
    pass

warm = sys.argv[1] == "warm"
source = torch.empty(64, device="cuda")
output = torch.empty_like(source)
if sys.argv[2] == "triton":
    sys.path.insert(0, sys.argv[3])
    from test_compilation import _cached_fill_triton, _cached_negate_triton
    from ignition.preparation.triton import prepare_triton
    kernel = _cached_negate_triton if warm else _cached_fill_triton
    launch = prepare_triton(kernel, (source, 7. if warm else 3.), (1,))
else:
    launch = lambda: fill(source, 7. if warm else 3.)
def forward():
    launch()
    output.copy_(source * 2)
with torch.no_grad():
    if warm:
        with patch("torch._inductor.standalone_compile", side_effect=AssertionError("cache miss")):
            compiled = compile_forward(forward)
    else:
        compiled = compile_forward(forward)
    assert compiled.cache_hit == warm
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    output.zero_()
    graph.replay()
    expected = 14. if warm else 6.
    if warm and sys.argv[2] == "triton":
        expected = -expected
    torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=0)
'''
    for mode in ("cold", "warm"):
        environment = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(tmp_path / (mode + "-inductor")))
        subprocess.run([sys.executable, "-c", program, mode, native_kind, os.path.dirname(__file__), str(tmp_path / "compiled")],
                       env=environment, check=True)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@pytest.mark.parametrize("separate_views", [False, True])
@torch.no_grad()
def test_captured_scratch_aliases_preserve_intermediate_values(separate_views):
    workspace = torch.zeros(5, 336, device="cuda", dtype=torch.bfloat16)
    first_writer = workspace[:] if separate_views else workspace
    projection = workspace[:, :320] if separate_views else workspace
    injection = workspace[:, 320:324]
    snapshot = torch.empty_like(projection)
    injection_snapshot = torch.empty_like(injection)
    final = torch.empty_like(projection)

    def forward(writer=first_writer):
        _fill_output(writer, 1.)
        snapshot.copy_(projection)
        injection_snapshot.copy_(injection)
        _fill_output(workspace, 2.)
        final.copy_(projection)

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for value in (0., 3., -7.):
        workspace.fill_(value)
        graph.replay()
        torch.testing.assert_close(snapshot, torch.ones_like(snapshot), rtol=0, atol=0)
        torch.testing.assert_close(injection_snapshot, torch.ones_like(injection_snapshot), rtol=0, atol=0)
        torch.testing.assert_close(final, torch.full_like(final, 2.), rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@pytest.mark.parametrize("rows", [1, 512])
@torch.no_grad()
def test_compiled_head_norm_preserves_fp32_reduction_order(rows):
    packed = torch.zeros(rows, 13312, device="cuda", dtype=torch.bfloat16)
    source = packed[:, 12288:12800].view(rows, 2, 256)
    weight = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(source)
    variance = torch.empty(rows, 2, 1, device="cuda", dtype=torch.float32)

    def forward():
        value = source.float()
        mean = value.square().mean(-1, keepdim=True)
        variance.copy_(mean)
        normalized = value * torch.rsqrt(mean + 1e-6)
        output.copy_((normalized * (1.0 + weight.float())).bfloat16())

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    generator = torch.Generator(device="cuda").manual_seed(903)
    for _ in range(3):
        source.normal_(generator=generator)
        forward()
        expected, expected_variance = output.clone(), variance.clone()
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(variance, expected_variance, rtol=0, atol=0)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@torch.no_grad()
def test_snapshots_of_reused_copy_destinations():
    source = torch.zeros(1, 6144, device="cuda", dtype=torch.bfloat16)
    staged = torch.empty_like(source)
    snapshots = [torch.empty_like(source), torch.empty_like(source)]

    def forward():
        for value, snapshot in enumerate(snapshots, 1):
            _fill_output(source, float(value))
            staged.copy_(source)
            snapshot.copy_(staged)

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    graph.replay()
    for value, snapshot in enumerate(snapshots, 1):
        torch.testing.assert_close(snapshot, torch.full_like(snapshot, value), rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@pytest.mark.parametrize("operation", ("unbind", "split"))
@torch.no_grad()
def test_list_views_preserve_offsets_and_mutations_under_capture(operation):
    source = torch.arange(192, dtype=torch.float32, device="cuda").reshape(8, 3, 8)
    storage = torch.empty_like(source)
    output = torch.empty((8, 8), device="cuda")

    def forward():
        storage.copy_(source)
        if operation == "unbind":
            a, b, c = storage.unbind(1)
        else:
            a, b, c = (v.squeeze(1) for v in storage.split(1, dim=1))
        b.mul_(3)
        output.copy_(a + b + c)
        a.add_(7)

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for _ in range(3):
        source.add_(1)
        storage.fill_(float("nan"))
        graph.replay()
        expected = source.clone()
        expected[:, 1].mul_(3)
        torch.testing.assert_close(output, expected.sum(1))
        expected[:, 0].add_(7)
        torch.testing.assert_close(storage, expected)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@torch.no_grad()
def test_compilation_cache_relocates_between_cuda_ordinals(tmp_path, monkeypatch):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two explicitly selected CUDA GPUs")
    monkeypatch.setattr("ignition.preparation.compilation.DEFAULT_CACHE_ROOT", tmp_path)
    graphs, outputs, functions = [], [], []
    for ordinal in (0, 1):
        with torch.cuda.device(ordinal):
            source = torch.full((64,), ordinal + 1., device=f"cuda:{ordinal}")
            output = torch.empty_like(source)

            def forward(source=source, output=output):
                output.copy_(source * 3)

            compiled = compile_forward(forward)
            assert compiled.cache_hit == bool(ordinal)
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream(device=ordinal)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(graph, stream=stream):
                compiled()
            torch.cuda.current_stream().wait_stream(stream)
            outputs.append(output)
            graphs.append(graph)
            functions.append((compiled, source))
    assert len(tuple((tmp_path / "compiled").glob("*.bin"))) == 1
    for ordinal, (graph, output, (_, source)) in enumerate(zip(graphs, outputs, functions, strict=True)):
        with torch.cuda.device(ordinal):
            source.fill_(ordinal + 4.)
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.testing.assert_close(output, torch.full_like(output, (ordinal + 4.) * 3), rtol=0, atol=0)
            assert torch.cuda.memory_allocated() == allocated
            graph.reset()
