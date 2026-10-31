"""Block-FP8 plans declare the activation quantizer's compile job on every architecture."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import block_fp8, block_fp8_sm103
from ignition.components.block_fp8 import BlockFP8Projection, BlockFP8Weight
from ignition.components.native_operator import native_contract, native_plan

QUANTIZER = "ignition.components.block_fp8:compile_token_group_quantization"


def quantizer_jobs(plan, ordinal):
    config = SimpleNamespace(tactic=1)
    return plan._compile_jobs(config, SimpleNamespace(ordinal=ordinal))


@pytest.mark.parametrize("sm103", [True, False])
def test_projection_plan_compiles_quantizer_for_its_input_width(sm103, monkeypatch):
    monkeypatch.setattr("ignition.components.sm103.is_sm103", lambda device: sm103)
    monkeypatch.setattr(block_fp8, "declare", lambda session, requests: None)

    def gemm_plan(query):
        # Stands in for b12x's block-FP8 GEMM plan, whose own compile jobs must be preserved.
        return native_plan(contract=block_fp8_sm103.CONTRACT, query=block_fp8_sm103.BlockFP8Query(64, 512, 256),
                           device=torch.device("cuda", 0), scratch_bytes=lambda tactic: 0,
                           materialize=lambda tactic: None, compile_jobs=lambda tactic, detected: ("gemm",))

    monkeypatch.setattr(block_fp8.blockscaled, "plan", gemm_plan)
    weight = BlockFP8Weight(torch.zeros((256, 512), dtype=torch.float8_e4m3fn),
                            torch.ones((2, 4), dtype=torch.float32))
    projection = BlockFP8Projection(torch.zeros((64, 512), dtype=torch.bfloat16), weight,
                                    session=None, name="projection")
    jobs = quantizer_jobs(projection.plan, 3)
    quantizers = [job for job in jobs if job != "gemm"]
    assert [(job.factory, job.args) for job in quantizers] == [(QUANTIZER, (512, 3))]
    assert ("gemm" in jobs) is (not sm103)


@pytest.mark.parametrize("output,in_place", [
    (lambda: torch.zeros((64, 256), dtype=torch.bfloat16), True),
    (lambda: torch.zeros((64, 512), dtype=torch.bfloat16)[:, 128:384], False),
    (lambda: torch.zeros(64 * 256 + 1, dtype=torch.bfloat16)[1:].view(64, 256), False),
    (None, False)])
def test_only_contiguous_aligned_outputs_are_written_in_place(output, in_place, monkeypatch):
    monkeypatch.setattr("ignition.components.sm103.is_sm103", lambda device: True)
    monkeypatch.setattr(block_fp8, "declare", lambda session, requests: None)
    weight = BlockFP8Weight(torch.zeros((256, 512), dtype=torch.float8_e4m3fn),
                            torch.ones((2, 4), dtype=torch.float32))
    projection = BlockFP8Projection(torch.zeros((64, 512), dtype=torch.bfloat16), weight,
                                    None if output is None else output(), session=None, name="projection")
    assert projection.in_place is in_place


def test_other_architectures_copy_the_result_instead_of_writing_in_place(monkeypatch):
    other = native_contract(component_id="test.other_block_fp8", query_type=block_fp8_sm103.BlockFP8Query,
                            tactics=lambda query: (1,))
    monkeypatch.setattr("ignition.components.sm103.is_sm103", lambda device: False)
    monkeypatch.setattr(block_fp8, "declare", lambda session, requests: None)
    monkeypatch.setattr(block_fp8.blockscaled, "plan", lambda query: native_plan(
        contract=other, query=block_fp8_sm103.BlockFP8Query(64, 512, 256),
        device=torch.device("cuda", 0), scratch_bytes=lambda tactic: 0,
        materialize=lambda tactic: None, compile_jobs=lambda tactic, detected: ()))
    weight = BlockFP8Weight(torch.zeros((256, 512), dtype=torch.float8_e4m3fn),
                            torch.ones((2, 4), dtype=torch.float32))
    projection = BlockFP8Projection(torch.zeros((64, 512), dtype=torch.bfloat16), weight,
                                    torch.zeros((64, 256), dtype=torch.bfloat16), session=None, name="projection")
    assert projection.in_place is False
