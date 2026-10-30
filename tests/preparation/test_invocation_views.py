"""Relative tensor views passed to opaque compiled operations."""

import os

import pytest
import torch
from torch.utils._pytree import tree_flatten

from ignition.preparation.compilation import _Invocation, compile_forward


def invocation(target, descriptors):
    leaves, tree = tree_flatten(((None,) * len(descriptors), {}))
    return _Invocation(target, leaves, tree,
        [(position, mutable, index, view)
         for position, (mutable, index, view) in enumerate(descriptors)])


@pytest.mark.parametrize('base_offset', [0, 3])
def test_relative_read_view_preserves_runtime_base_offset(base_offset):
    storage = torch.arange(16, dtype=torch.int32)
    observed = []
    call = invocation(lambda value: observed.append(value.clone()),
        [(False, 0, (torch.int32, (2,), (2,), 2))])
    call.run([storage[base_offset:]], [])
    torch.testing.assert_close(observed[0], storage[base_offset + 2:base_offset + 6:2])


@pytest.mark.parametrize('mutable_source', [False, True])
def test_reinterpreted_read_and_write_views_preserve_aliases(mutable_source):
    storage = torch.arange(12, dtype=torch.int32)
    base = storage.view(torch.uint8)[8:]
    observed = []

    def update(source, destination):
        observed.append(source.clone())
        assert source.untyped_storage()._cdata == destination.untyped_storage()._cdata
        destination.add_(100)
        observed.append(source.clone())

    call = invocation(update, [
        (mutable_source, 0, (torch.int32, (3,), (1,), 1)),
        (True, 0, (torch.int32, (1,), (1,), 2)),
    ])
    call.run([base], [base])
    torch.testing.assert_close(observed[0], torch.tensor([3, 4, 5], dtype=torch.int32))
    torch.testing.assert_close(observed[1], torch.tensor([3, 104, 5], dtype=torch.int32))
    torch.testing.assert_close(storage, torch.tensor([0, 1, 2, 3, 104, 5, 6, 7, 8, 9, 10, 11], dtype=torch.int32))


def test_independent_scalar_view_does_not_reset_runtime_slice():
    counts = torch.tensor([143, 1], dtype=torch.int32)
    observed = []
    call = invocation(lambda value: observed.append(value.item()),
        [(False, 0, (torch.int32, (1,), (1,), 0))])
    call.run([counts[1:]], [])
    assert observed == [1]


class CountedEmbedding:
    def __init__(self, session):
        from ignition.components.embedding import PreparedEmbedding

        self.counts = torch.tensor([143, 1], dtype=torch.int32, device='cuda')
        self.count = torch.zeros(1, dtype=torch.int32, device='cuda')
        self.source_ids = torch.arange(8, dtype=torch.int64, device='cuda')
        self.ids = torch.zeros(4, dtype=torch.int64, device='cuda')
        self.weight = torch.arange(8 * 64, dtype=torch.bfloat16, device='cuda').view(8, 64)
        self.output = torch.zeros((4, 64), dtype=torch.bfloat16, device='cuda')
        self.embedding = PreparedEmbedding(self.weight, self.ids, self.output,
            num_rows=self.count, session=session, name='counted.embedding')

    def __call__(self):
        self.count.copy_(self.counts[1:])
        self.ids.copy_(self.source_ids[::2])
        self.embedding()
        return self.output


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='explicit CUDA assignment required')
@pytest.mark.parametrize('backend', ['fx', 'aot'])
@torch.no_grad()
def test_compiled_b12x_embedding_preserves_copied_scalar_slice(backend, tmp_path, monkeypatch):
    from b12x.preparation import PreparationSession
    from ignition.preparation.aot import aot_forward

    monkeypatch.setattr('ignition.preparation.compilation.DEFAULT_CACHE_ROOT', tmp_path)
    with PreparationSession(device='cuda:0', autotune=False) as session:
        forward = CountedEmbedding(session)
        compiled = compile_forward(forward) if backend == 'fx' else aot_forward(forward)
        if backend == 'aot':
            path = tmp_path / 'counted-embedding.bin'
            compiled.save_compiled_function(str(path))
            with path.open('rb') as file:
                compiled = torch.compiler.load_compiled_function(file)
        run = compiled if backend == 'fx' else lambda: compiled(forward)
        run()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            run()
        for count in (0, 1, 3, 4, 1):
            forward.counts[1] = count
            expected = torch.zeros_like(forward.output)
            expected[:count].copy_(forward.weight[forward.ids[:count]])
            before = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == before
            torch.testing.assert_close(forward.output, expected, rtol=0, atol=0)
        graph.reset()


@pytest.mark.parametrize('offset,stride,opaque', [(0, 1, False), (1, 1, True), (0, 2, True)])
def test_copy_lowering_preserves_destination_for_offset_or_strided_sources(offset, stride, opaque):
    from ignition.preparation.compilation import ForwardTracer, _lower_storage

    source = torch.arange(16, dtype=torch.int32)
    destination = torch.zeros(4, dtype=torch.int32)
    module = ForwardTracer()(lambda: destination.copy_(source[offset:offset + 4 * stride:stride]))
    calls = _lower_storage(module)
    assert any(call.target is torch.ops.aten.copy_.default for call in calls) is opaque


@pytest.mark.parametrize('overlapping', [False, True])
def test_copy_lowering_preserves_destination_for_sources_gathered_from_it(overlapping):
    from ignition.preparation.compilation import ForwardTracer, _lower_storage

    output = torch.zeros((4, 8), dtype=torch.int32)
    physical = output[:2]
    destination = output if overlapping else torch.zeros_like(output)
    indices = torch.zeros(4, dtype=torch.int64)
    module = ForwardTracer()(lambda: destination.copy_(physical[indices]))
    calls = _lower_storage(module)
    assert any(call.target is torch.ops.aten.copy_.default for call in calls) is overlapping


class CountedShortConvolution:
    def __init__(self, session):
        from b12x.sequence import short_conv
        from ignition.components.recurrent import prepare_sequence_binding

        self.counts = torch.tensor([4, 0], dtype=torch.int32, device='cuda')
        self.num_seqs = torch.zeros(1, dtype=torch.int32, device='cuda')
        self.num_tokens = torch.zeros(1, dtype=torch.int32, device='cuda')
        self.source = torch.ones((4, 16), dtype=torch.bfloat16, device='cuda')
        self.history = torch.zeros((2, 16, 3), dtype=torch.bfloat16, device='cuda')
        self.output = torch.zeros_like(self.source)
        self.operation = short_conv.run
        arguments = dict(x=self.source, conv_weight=torch.ones((16, 4), dtype=torch.bfloat16, device='cuda'),
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32, device='cuda'),
            state_slot_ids=torch.tensor([1], dtype=torch.int64, device='cuda'),
            state_is_fresh=torch.ones(1, dtype=torch.bool, device='cuda'),
            num_accepted_tokens=torch.ones(1, dtype=torch.int32, device='cuda'),
            num_seqs=self.num_seqs, num_tokens=self.num_tokens, conv_state=self.history, out=self.output)
        plan = short_conv.plan(short_conv.Caps(device='cuda:0', mode='prefill', channels=16,
            kernel_size=4, max_tokens=4, max_seqs=1, max_state_slots=2),
            invocation=short_conv.invocation_from_tensors(**arguments))
        prepare_sequence_binding(plan, arguments, session=session, name='counted.short_conv',
            public_bind=short_conv.bind, bind_plan=True, mode='prefill',
            set_binding=lambda value: setattr(self, 'binding', value))

    def __call__(self):
        self.num_seqs.copy_(self.counts[1:])
        self.num_tokens.copy_(self.counts[:1])
        self.output.zero_()
        self.operation(self.binding)
        return self.output


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='explicit CUDA assignment required')
@pytest.mark.parametrize('backend', ['fx', 'aot'])
@torch.no_grad()
def test_compiled_b12x_short_convolution_preserves_prepared_count_alignment(backend, tmp_path, monkeypatch):
    from b12x.preparation import PreparationSession
    from ignition.preparation.aot import aot_forward

    monkeypatch.setattr('ignition.preparation.compilation.DEFAULT_CACHE_ROOT', tmp_path)
    with PreparationSession(device='cuda:0', autotune=False) as session:
        forward = CountedShortConvolution(session)
        compiled = compile_forward(forward) if backend == 'fx' else aot_forward(forward)
        if backend == 'aot':
            path = tmp_path / 'counted-short-convolution.bin'
            compiled.save_compiled_function(str(path))
            with path.open('rb') as file:
                compiled = torch.compiler.load_compiled_function(file)
        run = compiled if backend == 'fx' else lambda: compiled(forward)
        run()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            run()
        for count in (1, 0, 1):
            forward.counts[1] = count
            forward()
            expected, history = forward.output.clone(), forward.history.clone()
            reference = torch.nn.functional.silu(torch.arange(1, 5, dtype=torch.bfloat16, device='cuda'))
            reference = reference[:, None].expand_as(expected) if count else torch.zeros_like(expected)
            torch.testing.assert_close(expected, reference, rtol=0, atol=0)
            forward.output.fill_(-1)
            before = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == before
            torch.testing.assert_close(forward.output, expected, rtol=0, atol=0)
            torch.testing.assert_close(forward.history, history, rtol=0, atol=0)
        graph.reset()
