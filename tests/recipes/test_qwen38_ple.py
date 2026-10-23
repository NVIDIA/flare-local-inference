"""Exact-checkpoint PLE compact rows and native CUDA graph continuity."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.qwen38 import PLE_PREFIX
from ignition.recipes.qwen38.ple import (
    PLE_LAYER_PREFIX,
    TABLE_BYTES,
    PLETable,
    PLEMetadata,
    PLEScratch,
    PreparedPLE,
    _validate_geometry,
    allocate_convolution_state,
    table_layout,
)
from ignition.recipes.qwen38.weights import QwenWeightLoader
from ignition.reference.qwen38_quant import (
    CheckpointTensors,
    read_ple_rows,
)

CUDA = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires explicit CUDA_VISIBLE_DEVICES and FLARE_RUN_CUDA_TESTS=1",
)


@pytest.fixture(scope="module")
def manifest():
    default = Path("/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD")
    path = Path(os.environ.get("FLARE_QWEN38_SNAPSHOT", default))
    if not path.exists():
        pytest.skip("exact Qwen checkpoint snapshot is unavailable")
    return inspect_qwen38_checkpoint(path)


def _module(manifest):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer

    config = Qwen4ExpTextConfig.from_dict(
        json.loads((manifest.snapshot / "config.json").read_text())["text_config"]
    )
    with torch.device("meta"):
        module = Qwen4ExpTextPLELayer(config, layer_idx=1, ple_layer_index=0)
    return module.eval().requires_grad_(False)


def _parameters(module, load):
    for name in (
        "conv1d.weight",
        "norm_key.weight",
        "norm_query.weight",
        "norm_conv.weight",
    ):
        parent, _, attribute = name.rpartition(".")
        setattr(
            module.get_submodule(parent),
            attribute,
            torch.nn.Parameter(load(f"{PLE_LAYER_PREFIX}.{name}"), requires_grad=False),
        )


def test_exact_ple_geometry_and_table_extent(manifest):
    module = _module(manifest)
    _validate_geometry(module)
    _, layout = table_layout(manifest, "cuda:0")
    assert layout.weight_shape == (128 * 2500012, 80)
    assert layout.weight_scale_shape == (128 * 2500012, 10)
    assert layout.caps.table_memory == "io_uring"
    assert layout.caps.eos_token_id == 248044
    assert TABLE_BYTES == 28800138240
    assert layout.geometry.prime_sizes == tuple(module.ple_embedding.head_vocab_sizes)
    assert layout.geometry.table_offsets == tuple(module.ple_embedding.head_offsets)
    for projection in ("key_proj", "value_proj"):
        assert (
            manifest.tensors[f"{PLE_LAYER_PREFIX}.{projection}.weight"].dtype == "BF16"
        )
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        _build_layer_multipliers,
    )

    assert layout.geometry.multipliers == tuple(
        _build_layer_multipliers(248320, 3, 0, module.ple_embedding.seed).tolist()
    )
    module.norm_key.weight = torch.nn.Parameter(
        torch.empty(10240, dtype=torch.bfloat16)
    )
    with pytest.raises(ValueError, match="FP32"):
        _validate_geometry(module)


def test_ple_fixed_buffer_contracts():
    metadata = PLEMetadata.allocate(17, 3, "cpu")
    scratch = PLEScratch.allocate(17, "cpu")
    state = allocate_convolution_state(4, "cpu", speculative_tokens=2)
    assert metadata.committed_history.shape == (3, 2)
    assert torch.all(metadata.committed_history == 248044)
    assert metadata.num_seqs.item() == metadata.num_tokens.item() == 0
    assert scratch.output.shape == (17, 10240)
    assert state.shape == (4, 10240, 11)
    for function, arguments in (
        (PLEMetadata.allocate, (0, 3, "cpu")),
        (PLEScratch.allocate, (0, "cpu")),
        (allocate_convolution_state, (0, "cpu")),
    ):
        with pytest.raises(ValueError):
            function(*arguments)


def test_export_metadata_preserves_hashes_encoding_scale_and_source_spans(tmp_path):
    from types import SimpleNamespace
    from ignition.preparation.runtime_package import FileSource

    geometry = SimpleNamespace(multipliers=(13, 17, 19, 23), prime_sizes=(29, 31, 37, 41),
                               table_offsets=(0, 29, 60, 97))
    layout = SimpleNamespace(caps=SimpleNamespace(eos_token_id=248044), geometry=geometry)
    for plane in ("weights", "scales"):
        for part in range(2):
            (tmp_path / f"{plane}-{part}").write_bytes(bytes(256))
    sources = tuple(tuple(FileSource(tmp_path / f"{plane}-{part}", 37 + part, 90 + part)
                          for part in range(2)) for plane in ("weights", "scales"))
    table = PLETable(dict(vocab_size=248320, ngram_size=3, heads_per_ngram=2), layout,
                     None, torch.tensor([0.125], dtype=torch.float32), None, None, None,
                     None, None, sources, 0.125)
    metadata = table.export_metadata()
    assert metadata["pad_token_id"] == 248044
    assert metadata["compressed_vocabulary"] == 248320
    stored = metadata["tables"][0]
    assert (stored["rows"], stored["embedding_width"], stored["scale_block"]) == (320001536, 160, 16)
    assert stored["encoding"] == "NVFP4_E4M3_GLOBAL"
    assert stored["global_scale"] == 0.125
    assert stored["hash"] == dict(method="EOS_XOR_MODULO", orders=[2, 2, 3, 3],
                                  multipliers=[13, 17, 19, 23], primes=[29, 31, 37, 41],
                                  offsets=[0, 29, 60, 97])
    for name, spans in zip(("weights", "scales"), sources, strict=True):
        assert stored[name] == [dict(path=str(span.path), offset=span.offset, nbytes=span.nbytes) for span in spans]
    assert len(list(tmp_path.iterdir())) == 4
    assert all(path.read_bytes() == bytes(256) for path in tmp_path.iterdir())


class _ReferenceTable(torch.nn.Module):
    def __init__(self, manifest):
        super().__init__()
        self.manifest = manifest
        self.register_buffer("weight", torch.empty(0))

    def forward(self, ids):
        with CheckpointTensors(self.manifest, byte_budget=4 << 20) as reader:
            return read_ple_rows(reader, ids)


@pytest.fixture(scope="module")
def reference_module(manifest):
    module = _module(manifest)
    with CheckpointTensors(manifest, byte_budget=256 << 20) as reader:
        _parameters(module, reader.read)
        for projection in ("key_proj", "value_proj"):
            prefix = f"{PLE_LAYER_PREFIX}.{projection}"
            module.get_submodule(projection).weight = torch.nn.Parameter(
                reader.read(prefix + ".weight"), requires_grad=False
            )
        for name in (
            "layer_multipliers",
            "ngram_heads_vocab_sizes",
            "ngram_heads_offsets",
        ):
            setattr(module.ple_embedding, name, reader.read(f"{PLE_PREFIX}.{name}"))
    module.ple_embedding.ngram_embedding = _ReferenceTable(manifest)
    return module


@pytest.fixture(scope="module")
def cuda_components(manifest):
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    assert torch.cuda.get_device_capability(0) in ((12, 0), (12, 1))
    module = _module(manifest)
    with QwenWeightLoader(
        manifest, "cuda:0", prefixes=(PLE_LAYER_PREFIX + ".",)
    ) as loader:
        _parameters(module, loader.load_tensor)
        projections = {
            name: loader.load_tensor(f"{PLE_LAYER_PREFIX}.{name}.weight")
            for name in ("key_proj", "value_proj")
        }
    table = PLETable.load(manifest, "cuda:0", tokens=32)
    table.sequence = torch.zeros(1, dtype=torch.int64, device="cuda:0")
    from ignition.components.readiness import ReadinessTimings
    table.timings = ReadinessTimings.allocate(torch.zeros(1, dtype=torch.int32, device="cuda:0"), 1)
    # Native owners retain mappings if stream drain cannot be confirmed.
    yield module, projections, table


def _cache():
    from transformers.cache_utils import Cache, LinearAttentionLayer

    return Cache(
        layers=[LinearAttentionLayer(), LinearAttentionLayer(number_of_states=3)]
    )


def _numerical(actual, expected):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    relative = float((actual - expected).norm() / expected.norm().clamp_min(1e-12))
    assert relative < 0.01, f"relative L2 {relative} >= .01"
    torch.testing.assert_close(
        actual, expected, rtol=0.03, atol=float(expected.abs().max()) * 0.01 + 1e-6
    )
    return relative


def _metadata(metadata, tokens, caches, *, fresh):
    lengths = [item.numel() for item in tokens]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    metadata.token_ids.fill_(248044)
    if starts[-1]:
        metadata.token_ids[: starts[-1]].copy_(torch.cat(tokens))
    metadata.query_start_loc.fill_(-999)
    metadata.query_start_loc[: len(starts)].copy_(
        torch.tensor(starts, dtype=torch.int32)
    )
    metadata.num_tokens.fill_(starts[-1])
    metadata.num_seqs.fill_(len(tokens))
    metadata.convolution_slots.fill_(-1)
    metadata.fresh.fill_(fresh)
    metadata.accepted_tokens.fill_(1)
    metadata.committed_history.fill_(248044)
    for index in range(len(tokens)):
        metadata.convolution_slots[index] = index + 1
        if not fresh:
            metadata.committed_history[index].copy_(
                caches[index].layers[1].conv_states[2][0]
            )


def _stage_checkpoint_rows(table, lookup):
    lookup.publish()
    ids = lookup.ids.cpu().flatten().tolist()
    for destination, sources, row_bytes in ((table.weights, table.sources[0], 80),
                                            (table.scales, table.sources[1], 10)):
        packed = bytearray(len(ids) * row_bytes)
        for output_row, row in enumerate(ids):
            if row < 0:
                continue
            plane, local = divmod(row, 2500012)
            source = sources[plane]
            with source.path.open("rb", buffering=0) as file:
                file.seek(source.offset + local * row_bytes)
                data = file.read(row_bytes)
            assert len(data) == row_bytes
            packed[output_row * row_bytes:(output_row + 1) * row_bytes] = data
        destination[:len(ids)].copy_(torch.frombuffer(packed, dtype=torch.uint8).view(len(ids), row_bytes))
    table.ready.record()


@CUDA
@torch.no_grad()
@pytest.mark.parametrize("initial_tokens", (1, 17))
def test_full_checkpoint_ple_native_graph_continuity(
    cuda_components, reference_module, initial_tokens, record_property
):
    from b12x.preparation import PreparationSession

    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.scratch import GraphBufferPool

    module, projections, table = cuda_components
    device = torch.device("cuda", 0)
    convolution = allocate_convolution_state(4, device)
    sources = {
        "prefill": torch.zeros((32, 10240), dtype=torch.bfloat16, device=device),
        "decode": torch.zeros((3, 10240), dtype=torch.bfloat16, device=device),
    }
    buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
    scratch = {
        name: PLEScratch.allocate(source.shape[0], device, buffers=buffers.cursor())
        for name, source in sources.items()
    }
    metadata = {
        name: PLEMetadata.allocate(source.shape[0], 3, device)
        for name, source in sources.items()
    }
    with PreparationSession(
        device=device, autotune=False, compile_workers=1
    ) as session:
        adapters = {
            name: PreparedPLE(
                module,
                projections,
                source,
                scratch[name],
                convolution,
                metadata[name],
                table,
                mode=name,
                session=session,
                name=f"ple.{name}",
            )
            for name, source in sources.items()
        }
        for name, adapter in adapters.items():
            adapter.embedding_lookup.publish()
            adapter(sources[name])
        torch.cuda.synchronize()
        session.freeze()
        pool = torch.cuda.graph_pool_handle()
        graphs = {name: torch.cuda.CUDAGraph() for name in sources}
        for name, graph in graphs.items():
            with torch.cuda.graph(graph, pool=pool):
                adapters[name].embedding_lookup.publish()
                adapters[name](sources[name])
        bundle = PreparedGraphBundle(
            device=0,
            graphs=graphs,
            buffers={
                "convolution": convolution,
                **sources,
                **{f"{name}.output": value.output for name, value in scratch.items()},
            },
            resources=(table, session, adapters, metadata, scratch),
        )
        worker = bundle.start_worker()
        try:
            caches = [_cache(), _cache()]
            generator = torch.Generator().manual_seed(1647)
            errors = {}
            convolution.fill_(9)
            for step, (mode, lengths) in enumerate(
                (
                    ("prefill", (initial_tokens, 1)),
                    ("decode", (1, 1)),
                    ("prefill", (3, 2)),
                )
            ):
                input_tokens = [
                    torch.arange(length, dtype=torch.int64) + 11 + 30 * index
                    for index, length in enumerate(lengths)
                ]
                input_tokens[1][-1] = 248044
                if initial_tokens > 3 and step == 0:
                    input_tokens[0][2] = 248044
                _metadata(metadata[mode], input_tokens, caches, fresh=step == 0)
                source = (
                    torch.randn((sum(lengths), 10240), generator=generator) * 0.1
                ).to(torch.bfloat16)
                sources[mode].zero_()
                sources[mode][: source.shape[0]].copy_(source)
                expected, expected_embedding = [], []
                start = 0
                for request, length in enumerate(lengths):
                    history_cache = _cache()
                    history_cache.layers[1].update_conv_state(
                        metadata[mode].committed_history[request : request + 1].cpu(),
                        state_idx=2,
                        conv_kernel_size=2,
                    )
                    expected_embedding.append(
                        reference_module.ple_embedding(
                            input_tokens[request][None], history_cache
                        )[0]
                    )
                    expected.append(
                        reference_module(
                            source[start : start + length][None],
                            input_tokens[request][None],
                            caches[request],
                        )[0]
                    )
                    start += length
                _stage_checkpoint_rows(table, adapters[mode].embedding_lookup)
                for chunks in buffers.chunks.values():
                    for tensor in chunks:
                        tensor.fill_(float("nan") if tensor.is_floating_point() else -37)
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                ticket = worker.submit(bundle.graph_indices[mode])
                completion = worker.poll(timeout_ms=30000)
                assert (
                    completion is not None
                    and completion.ticket == ticket
                    and completion.error is None
                )
                assert (
                    torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                )
                torch.testing.assert_close(
                    scratch[mode].embedding[: sum(lengths)].cpu(),
                    torch.cat(expected_embedding),
                    rtol=0,
                    atol=0,
                )
                errors[f"{step}.{mode}.output"] = _numerical(
                    scratch[mode].output[: sum(lengths)], torch.cat(expected)
                )
                assert torch.count_nonzero(scratch[mode].output[sum(lengths) :]) == 0
                for request in range(2):
                    errors[f"{step}.{mode}.history{request}"] = _numerical(
                        convolution[request + 1],
                        caches[request].layers[1].conv_states[1][0],
                    )
            previous = convolution.clone()
            _metadata(metadata["prefill"], [], caches, fresh=False)
            torch.cuda.synchronize()
            ticket = worker.submit(bundle.graph_indices["prefill"])
            completion = worker.poll(timeout_ms=30000)
            assert (
                completion is not None
                and completion.ticket == ticket
                and completion.error is None
            )
            assert torch.count_nonzero(scratch["prefill"].output) == 0
            torch.testing.assert_close(convolution, previous, rtol=0, atol=0)
            record_property("relative_l2", json.dumps(errors, sort_keys=True))
            record_property("compact_bytes", table.weights.nbytes + table.scales.nbytes)
        finally:
            worker.close()
            for graph in graphs.values():
                graph.reset()


@pytest.mark.parametrize("world", (2, 4))
def test_ple_rank_rows_cover_the_checkpoint_once(manifest, world):
    layouts = [table_layout(manifest, "cuda:0", rank=rank, world_size=world)[1] for rank in range(world)]
    assert layouts[0].shard_start == 0
    assert layouts[-1].shard_end == 320001536
    for rank, layout in enumerate(layouts):
        assert layout.caps.tp_rank == rank and layout.caps.tp_size == world
        assert layout.weight_shape == (320001536 // world, 80)
        assert layout.weight_scale_shape == (320001536 // world, 10)
        if rank:
            assert layouts[rank - 1].shard_end == layout.shard_start
    assert sum(layout.weight_shape[0] * 90 for layout in layouts) == TABLE_BYTES
