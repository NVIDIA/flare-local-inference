"""Block-FP8 preparation, compilation and replay preserve arbitrary FP32 scales."""

import gc
import os
import weakref

import pytest
import torch

from ignition.components.block_fp8 import BlockFP8Projection, BlockFP8Weight
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


@pytest.mark.parametrize("rows", [1, 2, 4, 17, 31, 32, 33, 64, 65, 4096])
def test_block_fp8_compiled_replay(rows, monkeypatch):
    from flashinfer.quantization import per_token_group_quant_8bit

    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(491)
    values = torch.randn((256, 512), device=device, generator=generator).to(torch.float8_e4m3fn)
    scales = torch.tensor([[.1, .2, .7, .3], [.4, .8, .3, .6]], device=device)
    source = torch.randn((rows, 512), device=device, dtype=torch.bfloat16, generator=generator)
    output = torch.empty((rows, 256), device=device, dtype=torch.bfloat16)
    with RecordedPreparationSession(device=device, checkpoint="block-fp8-projection",
            workload_schema="block-fp8-projection", numerical_policy="default") as session:
        projection = BlockFP8Projection(source, BlockFP8Weight(values, scales), session=session, name="projection")
        tuning_inputs = []

        def track_inputs(trial, *args, **kwargs):
            result = per_token_group_quant_8bit(trial, *args, **kwargs)
            tuning_inputs.extend(weakref.ref(value) for value in (trial, *result))
            return result

        with monkeypatch.context() as patch:
            patch.setattr("ignition.components.block_fp8.per_token_group_quant_8bit", track_inputs)
            session.tune()
        gc.collect()
        assert tuning_inputs and all(value() is None for value in tuning_inputs)

        def forward():
            output.copy_(projection(source))

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for multiplier in (1., .2, 2.):
            source.mul_(multiplier)
            quantized, activation_scale = per_token_group_quant_8bit(source, 128)
            expected = ((quantized.float() * activation_scale.repeat_interleave(128, 1))
                        @ (values.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)).T)
            before = torch.cuda.memory_stats(device)
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats(device)
            torch.testing.assert_close(output.float(), expected, rtol=.02, atol=.02)
            assert before["allocation.all.allocated"] == after["allocation.all.allocated"]
        graph.reset()
