"""Typed data must not create spurious CUDA resource dependencies."""

import gc
import os
import struct
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.runtime_package import Allocation
from ignition.preparation.tensor_metadata import NumericStorage, tensor_ranges


def test_tensor_views_exclude_holes_and_padding_and_deduplicate_aliases():
    tensor = torch.empty((5, 6), dtype=torch.float32)
    address = tensor.data_ptr()
    assert tensor_ranges(tensor.T) == [(address, address + 120)]
    assert tensor_ranges(tensor[1:4, 2:4]) == [
        (address + 32, address + 40), (address + 56, address + 64), (address + 80, address + 88)]
    assert tensor_ranges(tensor[0, :1].expand(100)) == [(address, address + 4)]
    assert tensor_ranges(tensor[:, :0]) == []


def test_checkpoint_role_tracks_storage_lifetime_without_retaining_weights():
    numeric = NumericStorage()
    tensor = torch.empty(128, dtype=torch.uint8)
    address = tensor.data_ptr()
    view = tensor[17:31]
    numeric.record(view)
    del view
    gc.collect()
    assert numeric.ranges() == [(address + 17, address + 31)]
    del tensor
    gc.collect()
    assert numeric.ranges() == []


def test_sampling_indices_do_not_retain_allocations_from_unaligned_token_ids():
    from ignition.components.speculation import PreparedSpeculativeDistributions

    with NumericStorage() as numeric:
        sampling = PreparedSpeculativeDistributions(
            torch.empty((1, 2, 256)),
            SimpleNamespace(vocab_size=256, generation=torch.zeros(1, dtype=torch.int64)),
            torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int64),
            torch.ones(1, dtype=torch.int32), torch.zeros((1, 1), dtype=torch.int64))
    indices = sampling.vocab_ids
    assert any(start <= indices.data_ptr() and indices.data_ptr() + indices.nbytes <= end
               for start, end in numeric.ranges())


def test_resized_checkpoint_storage_invalidates_numeric_metadata():
    numeric = NumericStorage()
    tensor = torch.empty(128, dtype=torch.uint8)
    numeric.record(tensor)
    tensor.resize_(256)
    with pytest.raises(ValueError, match="storage was resized"):
        numeric.ranges()


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="explicit CUDA assignment required")
@pytest.mark.parametrize("codec", ["iq2_xxs", "q8_0"])
@pytest.mark.parametrize("kind", ["projection", "experts"])
def test_repacked_weight_planes_remain_numeric_leaves(codec, kind):
    from b12x._lib.quant.block_codec import block_codec
    from b12x.gemm.blockscaled import pack_weight
    from b12x.moe import fused_moe
    from ignition.components.linear import QuantizedProjection
    from ignition.components.moe import PreparedRoutedExperts
    from ignition.preparation.tensor_metadata import record_numeric_tensor
    from ignition.preparation.tuning import RecordedPreparationSession

    spec = block_codec(codec)
    source = torch.zeros((1, 256), dtype=torch.bfloat16, device="cuda")
    raw = torch.zeros((2, 256, 256 // spec.block_weights, spec.block_bytes),
                      dtype=torch.uint8, device=source.device)
    raw[..., :2] = torch.full((*raw.shape[:-1], 1), .003, dtype=torch.float16,
                             device=source.device).view(torch.uint8)
    session = RecordedPreparationSession(device=source.device, checkpoint="synthetic",
        workload_schema="numeric-weight-storage", numerical_policy="exact")
    with NumericStorage() as numeric:
        # Direct checkpoint copies classify the source; b12x replaces its storage.
        record_numeric_tensor(raw)
        if kind == "projection":
            weight = pack_weight(raw[0], recipe=codec)
            planes = (weight.values, weight.metadata)
            QuantizedProjection(source, weight, torch.empty_like(source), session=session,
                                name="projection", activation_mode="a16")
        else:
            plan = fused_moe.plan_weights(source=fused_moe.PackedSource(format=codec),
                activation=fused_moe.ActivationSpec(mode="a16", nonlinearity="relu2", io_dtype=torch.bfloat16),
                geometry=fused_moe.MoEGeometry(num_experts=2, hidden_size=256, intermediate_size=256))
            experts = fused_moe.prepare_weights(plan=plan,
                weights=fused_moe.BlockQuantWeights(raw, raw, codec=codec))
            planes = tuple(getattr(experts._impl, name) for name in
                           ("w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale"))
            PreparedRoutedExperts(source, experts,
                torch.zeros((1, 1), dtype=torch.int32, device=source.device),
                torch.ones((1, 1), device=source.device), torch.empty_like(source),
                session=session, name="experts")
    assert all(plane.untyped_storage().data_ptr() != raw.untyped_storage().data_ptr() for plane in planes)
    del raw
    gc.collect()
    unrelated = Allocation("unrelated", 0xee9f4a000000, 10 * 1024 * 1024)
    for plane in planes:
        owner = Allocation("weight", plane.data_ptr(), plane.nbytes)
        # The Windows-failing package had this unaligned address-shaped numeric word.
        data = b"\0" * 6 + struct.pack("<Q", unrelated.address + 0x31451f) + b"\0" * 2
        payload = plane.view(torch.uint8).reshape(-1)[:len(data)]
        payload.copy_(torch.tensor(list(data), dtype=torch.uint8, device=plane.device))
        data = payload.cpu().numpy().tobytes()
        assert any(start <= owner.address and owner.address + len(data) <= end
                   for start, end in numeric.ranges())


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="explicit CUDA assignment required")
def test_triton_launch_metadata_distinguishes_scalar_values_from_pointers():
    import triton
    import triton.language as tl
    from cuda.bindings import driver as cu
    from ignition.preparation.tensor_metadata import _triton_parameters

    @triton.jit
    def scale_values(source, output, count, factor, BLOCK: tl.constexpr):
        indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(source + indices, indices < count, other=0)
        tl.store(output + indices, values * factor, indices < count)

    source = torch.arange(513, dtype=torch.float32, device="cuda")
    output = torch.empty_like(source)
    kernel = scale_values[(triton.cdiv(source.numel(), 128),)](
        source, output, source.numel(), 2.5, 128)
    torch.testing.assert_close(output, source * 2.5, rtol=0, atol=0)
    layout = _triton_parameters(kernel)
    assert layout == ((8, False), (8, False), (4, True), (4, True),
                      (8, False), (8, False))
    for index, (size, _) in enumerate(layout):
        status, _, actual_size = cu.cuFuncGetParamInfo(cu.CUfunction(kernel.function), index)
        assert status == cu.CUresult.CUDA_SUCCESS
        assert actual_size == size


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="explicit CUDA assignment required")
def test_metadata_skips_uninitialized_cache_and_deduplicates_live_views(monkeypatch):
    import ignition.preparation.tensor_metadata as metadata

    cache = torch.empty((32768, 64), dtype=torch.float32, device="cuda")
    weights = torch.empty((4, 4), dtype=torch.float32, device="cuda")
    objects = [cache[:, ::2], weights, weights.view_as(weights), weights.view(torch.uint8)[:2]]
    monkeypatch.setattr(metadata.gc, "get_objects", lambda: objects)
    ranges = metadata.tensor_ranges
    visited = []

    def inspect(tensor):
        assert tensor.data_ptr() != cache.data_ptr()
        visited.append(tensor)
        return ranges(tensor)

    monkeypatch.setattr(metadata, "tensor_ranges", inspect)
    numeric, arguments = metadata.live_metadata([
        Allocation("cache", cache.data_ptr(), cache.nbytes, initialize="zero"),
        Allocation("weights", weights.data_ptr(), weights.nbytes),
    ], NumericStorage(), {})
    assert len(visited) == 2
    assert numeric == [(weights.data_ptr() + 32, weights.data_ptr() + 64)]
    assert arguments == {}


def test_overlapping_recorder_bounds_preserve_integer_alias_exclusion(monkeypatch):
    import ignition.preparation.tensor_metadata as metadata

    storage = torch.zeros(16, dtype=torch.float32)
    floating, pointers = storage[2:4], storage.view(torch.int64)[1:3]
    address = storage.data_ptr()
    # Classification inspects tensor metadata; CPU backing avoids creating a CUDA context.
    monkeypatch.setattr(torch.Tensor, "device", property(lambda _: SimpleNamespace(type="cuda")))
    monkeypatch.setattr(metadata.gc, "get_objects", lambda: [floating, pointers])
    numeric, arguments = metadata.live_metadata([
        Allocation("storage", address, storage.nbytes),
        Allocation("recorded", address + 8, 8),
    ], NumericStorage(), {})
    assert numeric == []
    assert arguments == {}


def test_numeric_views_crossing_zero_coverage_keep_opaque_aliases_and_skip_zero_bytes(monkeypatch):
    import ignition.preparation.tensor_metadata as metadata

    storage = torch.zeros(16, dtype=torch.float32)
    address = storage.data_ptr()
    floating, pointers = storage[:12], storage.view(torch.int64)[1:4]
    monkeypatch.setattr(torch.Tensor, "device", property(lambda _: SimpleNamespace(type="cuda")))
    monkeypatch.setattr(metadata.gc, "get_objects", lambda: [floating, pointers])
    numeric, _ = metadata.live_metadata([
        Allocation("recorded", address, storage.nbytes),
        Allocation("cache", address + 16, 8, initialize="zero"),
    ], NumericStorage(), {})
    assert numeric == [(address, address + 8), (address + 32, address + 48)]
    monkeypatch.setattr(metadata.gc, "get_objects", lambda: [floating])
    numeric, _ = metadata.live_metadata([
        Allocation("recorded", address, storage.nbytes),
        Allocation("cache", address + 16, 8, initialize="zero"),
    ], NumericStorage(), {})
    assert numeric == [(address, address + 16), (address + 24, address + 48)]


def test_declared_zero_ranges_skip_disposable_views_before_span_discovery(monkeypatch):
    import ignition.preparation.tensor_metadata as metadata

    scratch = torch.empty((32, 16), dtype=torch.float32)
    weights = torch.ones(4, dtype=torch.float32)
    objects = [scratch[:, ::2], weights]
    monkeypatch.setattr(torch.Tensor, "device", property(lambda _: SimpleNamespace(type="cuda")))
    monkeypatch.setattr(metadata.gc, "get_objects", lambda: objects)
    ranges = metadata.tensor_ranges
    visited = []

    def classify(tensor):
        assert tensor.data_ptr() != scratch.data_ptr()
        visited.append(tensor)
        return ranges(tensor)

    monkeypatch.setattr(metadata, "tensor_ranges", classify)
    numeric, arguments = metadata.live_metadata([
        Allocation("scratch", scratch.data_ptr(), scratch.nbytes),
        Allocation("recorded-scratch", scratch.data_ptr(), scratch.nbytes),
        Allocation("weights", weights.data_ptr(), weights.nbytes),
    ], NumericStorage(), {}, zero_ranges=[(scratch.data_ptr(), scratch.nbytes)])
    assert len(visited) == 1 and visited[0] is weights
    assert numeric == [(weights.data_ptr(), weights.data_ptr() + weights.nbytes)]
    assert arguments == {}
