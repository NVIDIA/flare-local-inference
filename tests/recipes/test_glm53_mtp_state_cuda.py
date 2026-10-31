# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Accepted KDA prefixes, replayed from verification records, match plain decode across widths."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.components.recurrent import DeltaMetadata
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.glm53.kda import KDARecords, KDAScratch, PreparedGlmKDA
from ignition.recipes.glm53.model import create_model_tree
from ignition.recipes.glm53.pool import ContinuationPool
from ignition.recipes.glm53.speculative import PreparedKDACommit, PreparedRound, convolution_regions
from ignition.recipes.glm53.weights import GlmWeightLoader, TEXT_PREFIX

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


@torch.no_grad()
def test_learned_kda_verification_commits_each_prefix_before_plain_decode():
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    manifest = inspect_indexed_checkpoint(snapshot)
    module = create_model_tree(snapshot).model.language_model.layers[0].self_attn
    with GlmWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 4),
                         prefixes=(f"{TEXT_PREFIX}.layers.0.self_attn.",)) as loader:
        weights = loader.load_attention(0)
    pool = ContinuationPool.allocate(("linear_attention",), 4, 10, 1, "cuda:0", mtp=True)
    state = pool.decoder.layers[0]
    arena = ScratchArena("cuda:0")
    metas = {width: DeltaMetadata.allocate(1, "cuda:0", verification_columns=8) for width in (1, 6)}
    sources = {width: torch.zeros((width, 4096), dtype=torch.bfloat16, device="cuda:0") for width in metas}
    slots = torch.tensor([0], dtype=torch.int32, device="cuda:0")
    counts = torch.ones(1, dtype=torch.int32, device="cuda:0")
    op = PreparedRound.__new__(PreparedRound)
    op.owner = SimpleNamespace(pool=pool, requests=SimpleNamespace(capacity=1),
                              inputs=SimpleNamespace(row_slot_ids=slots,
                                  request_state_rows=torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]], device="cuda:0")))
    op.width, op.batch, op.rows = 6, 1, torch.arange(1, device="cuda:0")
    op.convolution_regions = convolution_regions(pool)
    records = KDARecords.allocate({0: weights}, state.recurrent.shape[0], state.recurrent.shape[1], "cuda:0")
    op.decoder = SimpleNamespace(scratch=SimpleNamespace(hidden=torch.zeros((6, 4096), dtype=torch.bfloat16, device="cuda:0")))
    operators = {}
    with RecordedPreparationSession(device=torch.device("cuda:0"), checkpoint="glm53-mtp-state",
            workload_schema="glm-kda-prefix-v1", numerical_policy="default") as session:
        session.register_scratch(arena)
        for width in metas:
            operators[width] = PreparedGlmKDA(module, weights, sources[width],
                KDAScratch.allocate(width, 16, "cuda:0"), state, metas[width],
                mode="decode", session=session, name=f"decode{width}", workspace=arena,
                records=records.layer(0) if width > 1 else None)
        op.kda_commit = PreparedKDACommit(pool, records, operators[6].recurrent_plan,
            metas[6].state_indices[:, 0], metas[6].num_seqs, session=session, workspace=arena)
        session.tune()
        for width, meta in metas.items():
            meta.query_start_loc.copy_(torch.tensor([0, width], dtype=torch.int32))
            meta.convolution_slots.fill_(1)
            # Verification reads the live row in column zero; plain decode writes it back there.
            meta.state_indices.copy_(torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]]))
            meta.num_seqs.fill_(1)
            meta.num_tokens.fill_(width)
        generator = torch.Generator(device="cuda:0").manual_seed(531)
        chain = torch.randn((7, 4096), generator=generator, device="cuda:0", dtype=torch.bfloat16) * .1
        state.recurrent.normal_(0, .01, generator=generator)
        state.convolution.normal_(0, .1, generator=generator)
        initial_r, initial_c = state.recurrent.clone(), state.convolution.clone()
        sources[6].copy_(chain[:6])
        operators[6](sources[6])
        op.commit_state(slots.long(), counts)
        # Plain decode runs eagerly after the freeze, so its programs load first.
        sources[1].copy_(chain[:1])
        operators[1](sources[1])
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            operators[6](sources[6])
            op.commit_state(slots.long(), counts)
        session.freeze()
        # Each verified token's output matches decoding the tokens one at a time. Plain decode and verification
        # read different columns of an arbitrary convolution history, so this check starts from an empty one.
        plain = []
        state.recurrent.copy_(initial_r)
        state.convolution.zero_()
        for index in range(6):
            sources[1].copy_(chain[index:index + 1])
            plain.append(operators[1](sources[1]).clone())
        state.recurrent.copy_(initial_r)
        state.convolution.zero_()
        counts.fill_(1)
        graph.replay()
        torch.cuda.synchronize()
        verified = operators[6].projections["o_proj"].output.float()
        expected = torch.cat(plain).float()
        relative = ((verified - expected).norm() / expected.norm()).item()
        # Recording rounds the recurrence to BF16 before the gated norm, as b12x's recording does.
        assert relative < 1e-2, relative
        for accepted in range(1, 7):
            state.recurrent.copy_(initial_r)
            state.convolution.copy_(initial_c)
            for index in range(accepted):
                sources[1].copy_(chain[index:index + 1])
                operators[1](sources[1])
            expected_r, expected_c = state.recurrent[1].clone(), state.convolution[1, :, :3].clone()
            sources[1].copy_(chain[6:])
            expected_output = operators[1](sources[1]).clone()
            state.recurrent.copy_(initial_r)
            state.convolution.copy_(initial_c)
            counts.fill_(accepted)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(state.recurrent[1], expected_r, rtol=.025, atol=.002)
            torch.testing.assert_close(state.convolution[1, :, :3], expected_c, rtol=.025, atol=.002)
            actual_output = operators[1](sources[1])
            torch.testing.assert_close(actual_output, expected_output, rtol=.025, atol=.005)
