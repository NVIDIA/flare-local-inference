"""Actual Qwen target residual taps and packed DFlash2 state lifetimes."""

import os
from functools import partial
from pathlib import Path

import pytest
import torch

from ignition.recipes import resolve

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA GPU")


@pytest.mark.parametrize("checkpoint,online_mxfp8", [("qwen3.8-27b", False), ("qwen3.8-27b-qad", False), ("qwen3.8-27b-qad", True)])
@torch.no_grad()
def test_checkpoint_compiled_packed_round_preserves_taps_and_inactive_state(checkpoint, online_mxfp8):
    from b12x.preparation import PreparationSession
    from ignition.components.completion import CompletionOutputs
    from ignition.components.paged_batch import BatchInputs
    from ignition.components.parallel_draft import PreparedParallelDraftRound
    from ignition.components.sampling import DeviceRequestState, RequestAdmission
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.recipes.qwen38_dense.decoder import Scratch
    from ignition.recipes.qwen38_dense.dflash import SpeculativePreparation
    from ignition.recipes.qwen38_dense.pool import ContinuationPool
    from ignition.recipes.qwen38_dense.weights import ModelWeights

    hub = Path.home() / ".cache/huggingface/hub"
    target = resolve(checkpoint).snapshot(hub)
    draft = hub / "models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5"
    assert (target / "config.json").is_file() and (draft / "model.safetensors").is_file()
    weights = ModelWeights.load(inspect_indexed_checkpoint(target), torch.device("cuda:0"), mtp=True, draft_snapshot=draft, online_mxfp8=online_mxfp8)
    if checkpoint == "qwen3.8-27b-qad":
        from ignition.recipes.qwen38_dense.config import TEXT_PREFIX
        from ignition.reference.qwen38_quant import CheckpointTensors

        with CheckpointTensors(inspect_indexed_checkpoint(target), byte_budget=2 << 20) as reader:
            for name in ("layers.0.input_layernorm.weight", "layers.0.post_attention_layernorm.weight", "norm.weight"):
                actual = weights.tree.get_parameter(name)
                original = reader.read(TEXT_PREFIX + "." + name)
                assert actual.dtype == original.dtype == torch.float32
                assert torch.equal(actual.cpu(), original)
    pool = ContinuationPool.for_model(weights.config, 3, 2, "cuda:0", mtp=True)
    requests = DeviceRequestState.allocate("cuda:0", vocab_size=weights.config.vocabulary, capacity=2)
    inputs = BatchInputs.allocate(16, 1, "cuda:0", capacity=2)
    inputs.live_rows.fill_(2)
    inputs.row_slot_ids.copy_(torch.tensor([1, 0], device="cuda:0", dtype=torch.int32))
    inputs.row_generations.fill_(1)
    inputs.request_state_rows[0].fill_(1)
    inputs.request_state_rows[1].fill_(2)
    inputs.page_tables.copy_(torch.tensor([[0], [1]], device="cuda:0", dtype=torch.int32))
    inputs.query_lengths.copy_(torch.tensor([8, 1], device="cuda:0", dtype=torch.int32))
    completion = CompletionOutputs.allocate((2, 2, 24), device="cuda:0")
    frame = torch.zeros(1, device="cuda:0", dtype=torch.int32)
    sequence = torch.zeros(1, device="cuda:0", dtype=torch.int64)
    graph_buffers, draft_buffers, private_buffers, target_buffers, context_buffers = (GraphBufferPool("cuda:0", chunk_bytes=2 << 20) for _ in range(5))

    def reset():
        pool.clear_for_preparation()
        for slot in range(2):
            requests.admit(slot, RequestAdmission(generation=1, seed=2718 + slot, max_new_tokens=16,
                top_k=20, top_p=.95), [])
            requests.pending_token[slot] = 11 + slot
        completion.records.zero_()

    reset()
    with PreparationSession(device="cuda:0", autotune=False) as session:
        arena = ScratchArena("cuda:0")
        scratch = Scratch.allocate(9, 16, weights.config, "cuda:0", buffers=graph_buffers.cursor())
        owner = SpeculativePreparation(weights, pool, requests, inputs, scratch, completion=completion,
            frame=frame, sequence=sequence, session=session, workspace=arena, buffers=draft_buffers.cursor(),
            decoder_buffers=target_buffers, context_buffers=context_buffers)
        operation = PreparedParallelDraftRound(owner, 2, 8, "qwen.actual.packed", buffers=private_buffers.cursor(),
            target_token_capacity=9)
        operation.prepare()
        narrow = PreparedParallelDraftRound(owner, 2, 1, "qwen.actual.width1", buffers=private_buffers.cursor())
        narrow.prepare()
        arena.finalize()
        assert operation.decoder.auxiliary.shape == (9, 25600)
        assert operation.decoder.scratch.hidden.shape == (16, 5120)
        assert operation.decoder.scratch.logits.shape == (16, 248320)
        assert owner.draft_scratch.logits.untyped_storage().data_ptr() != operation.probabilities.untyped_storage().data_ptr()
        assert operation.decoder.auxiliary.untyped_storage().data_ptr() != operation.context.local.untyped_storage().data_ptr()
        compiled = compile_forward(operation)
        reset()
        compiled()
        reset()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        reset()
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert not requests.error_code.count_nonzero(), requests.error_code.tolist()
        assert operation.operations[-1].emitted_count.gt(0).all()
        accepted = operation.operations[-1].emitted_count.clone()
        auxiliary = operation.decoder.auxiliary.clone()
        normalized = operation.decoder.scratch.hidden.clone()
        candidates = operation.candidates.clone()
        target_rows = operation.metadata.state_rows.clone()
        target_pages = operation.metadata.page_table.clone()
        pool.clear_for_preparation()
        tapped = {}
        handles = [operation.decoder.layers[index].register_forward_hook(
            lambda module, args, output, index=index: tapped.__setitem__(index, output.clone()))
            for index in weights.config.draft.target_layers]
        operation.decoder()
        torch.cuda.synchronize()
        for handle in handles:
            handle.remove()
        expected = torch.cat([tapped[index] for index in weights.config.draft.target_layers], -1)
        relative = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))
        tap_eager_error = relative(auxiliary, expected)
        hidden_eager_error = relative(normalized, operation.decoder.scratch.hidden)
        assert torch.equal(operation.decoder.auxiliary, expected)
        eager_hidden = operation.decoder.scratch.hidden.clone()
        observed = [torch.empty_like(auxiliary[:, :5120]) for _ in range(5)]
        def observe(destination, module, args, output):
            destination.copy_(output)

        handles = [operation.decoder.layers[index].register_forward_hook(partial(observe, destination))
            for index, destination in zip(weights.config.draft.target_layers, observed, strict=True)]
        standalone = compile_forward(operation.decoder)
        for handle in handles:
            handle.remove()
        pool.clear_for_preparation()
        standalone()
        torch.cuda.synchronize()
        assert torch.equal(operation.decoder.auxiliary, torch.cat(observed, -1))
        standalone_error = relative(operation.decoder.auxiliary, auxiliary)
        feature_hidden = operation.decoder.scratch.hidden.clone()
        saved_auxiliary, operation.decoder.auxiliary = operation.decoder.auxiliary, None
        target_without_features = compile_forward(operation.decoder)
        operation.decoder.auxiliary = saved_auxiliary
        pool.clear_for_preparation()
        target_without_features()
        torch.cuda.synchronize()
        baseline_eager_error = relative(operation.decoder.scratch.hidden, eager_hidden)
        feature_extraction_error = relative(operation.decoder.scratch.hidden, feature_hidden)
        full_lengths = torch.tensor([8, 0], device="cuda:0", dtype=torch.int32)
        for count in range(1, 9):
            counts = torch.tensor([count, 0], device="cuda:0", dtype=torch.int32)
            pool.clear_for_preparation()
            inactive = pool.backing[target_rows[1]].clone()
            operation.metadata.set(torch.zeros_like(target_rows), full_lengths,
                target_rows, target_pages, candidates, inputs.live_rows)
            standalone()
            operation.decoder.commit(counts)
            torch.cuda.synchronize()
            assert operation.decoder.commit_counts.tolist() == [count, 0]
            assert torch.equal(pool.backing[target_rows[1]], inactive)

        inputs.query_lengths.fill_(1)
        compiled_narrow = compile_forward(narrow)
        reset()
        compiled_narrow()
        reset()
        narrow_graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(narrow_graph):
            compiled_narrow()
        reset()
        allocated = torch.cuda.memory_allocated()
        narrow_graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert narrow.operations[-1].emitted_count.tolist() == [1, 1]
        assert requests.consumed_tokens.tolist() == [1, 1]
        for layer in range(5):
            for kind in ("key", "value"):
                assert pool.reserved_region(f"draft{layer}.{kind}")[1:, 0].float().count_nonzero()
        inactive = pool.backing[2].clone()
        inactive_pages = {name: value[1].clone() for name, value in pool.pages.items()}
        inputs.row_slot_ids.copy_(torch.tensor([0, 1], device="cuda:0", dtype=torch.int32))
        inputs.query_lengths.copy_(full_lengths)
        inputs.live_rows.fill_(1)
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert not requests.error_code.count_nonzero(), requests.error_code.tolist()
        assert operation.operations[-1].emitted_count[0] > 0
        assert operation.operations[-1].emitted_count[1] == 0
        assert torch.equal(pool.backing[2], inactive)
        for name, page in inactive_pages.items():
            assert torch.equal(pool.pages[name][1], page), name
        print(dict(physical_tokens=9, logical_samples=16, learned_acceptance=accepted.tolist(),
            tap_eager_relative_l2=tap_eager_error, target_hidden_eager_relative_l2=hidden_eager_error,
            target_without_features_eager_relative_l2=baseline_eager_error,
            round_standalone_relative_l2=standalone_error, feature_extraction_relative_l2=feature_extraction_error,
            accepted_depths=list(range(1, 9)), width_transition=[1, 8], inactive_rows_preserved=True), flush=True)
        narrow_graph.reset()
        graph.reset()
