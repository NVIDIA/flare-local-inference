"""Engram export and resident delivery use identical interleaved checkpoint rows."""

from pathlib import Path

import pytest
from b12x.sequence import engram

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.runtime_package import FileSource
from ignition.recipes.deepseek41.engram import Tables


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_package_planes_and_resident_segments_preserve_head_bytes(tmp_path, world):
    geometry = engram.build_geometry(layer_ids=(1,), base_table_size=101, compressed_vocab_size=32)
    rows = geometry.num_embeddings[0]
    sources, contents = [], []
    for width in (256, 8):
        data = bytes((row * 13 + column) % 251 for row in range(rows) for column in range(width))
        path = tmp_path / str(width)
        path.write_bytes(bytes(37) + data)
        sources.append(FileSource(path, 37, len(data)))
        contents.append(data)
    tables = Tables(geometry, tuple(range(32)), {1: tuple(sources)}, {}, {}, None,
                    TensorParallelShard(0, world), {}, None)
    metadata = tables.export_metadata()
    table = metadata["tables"][0]
    assert table["hash"]["primes"] == list(geometry.primes[0])
    assert table["hash"]["multipliers"] == list(geometry.multipliers[0])
    global_row = 0
    for rank in range(world):
        partition = tables.partition(0, rank)
        stored = table if world == 1 else table["shards"][rank]
        if world > 1:
            assert (stored["first_row"], stored["rows"]) == (global_row, partition.rows)
        for head in partition.heads:
            assert table["hash"]["offsets"][head] == global_row + partition.row_mapping[head][2]
        rank_tables = Tables(geometry, tables.token_map, tables.sources, {}, {}, None,
                             TensorParallelShard(rank, world), {}, None)
        resident = rank_tables.resident_catalog()["tables"][0]["planes"]
        for ordinal, (name, width, source) in enumerate(zip(("weights", "scales"), (256, 8), contents)):
            chunks = stored[name]
            actual = b"".join(Path(chunk["path"]).read_bytes()[chunk["offset"]:chunk["offset"] + chunk["nbytes"]]
                              for chunk in chunks)
            expected = b"".join(source[first * width:last * width] for first, last in partition.source_rows)
            assert actual == expected and len(actual) == partition.rows * width
            assert resident[ordinal]["segments"] == chunks
        global_row += partition.rows
    assert global_row == rows
    planes = [table] if world == 1 else table["shards"]
    assert sum(chunk["nbytes"] for plane in planes for name in ("weights", "scales")
               for chunk in plane[name]) == rows * 264
