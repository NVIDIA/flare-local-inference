"""Safetensors interoperability, bounded metadata reads, and stable storage placement."""

import json
from pathlib import Path
import struct

import pytest
from safetensors import safe_open

from ignition.preparation import runtime_package_pb2 as proto
from ignition.preparation.package_file import (
    ALIGNMENT, MAX_HEADER_BYTES, MAX_INDEX_BYTES, align, combine, publish, read_index, write_sources,
)


def index():
    return proto.ModelIndex(version=4, ranks=[proto.BundleIndex()])


def header(payload_size, index_size):
    return {
        "__metadata__": {"format": "flare-staging"},
        "flare.payload": {"dtype": "U8", "shape": [payload_size], "data_offsets": [0, payload_size]},
        "flare.metadata": {"dtype": "U8", "shape": [index_size],
                           "data_offsets": [payload_size, payload_size + index_size]},
    }


def write_container(path, description, payload, metadata):
    encoded = json.dumps(description, separators=(",", ":")).encode()
    encoded = encoded.ljust(align(8 + len(encoded)) - 8, b" ")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload + metadata)


@pytest.mark.parametrize("data", [b"", b"weights" + bytes(17)])
def test_official_safetensors_reader_sees_both_byte_tensors(tmp_path, data):
    source = tmp_path / "storage"
    source.write_bytes(data)
    description = index()
    description.ranks[0].resources["storage"].CopyFrom(proto.Span(offset=0, nbytes=len(data)))
    path = tmp_path / "model.safetensors"
    publish(path, description, [(0, source, 0, len(data))])
    with safe_open(path, framework="numpy") as tensors:
        assert tensors.keys() == ["flare.metadata", "flare.payload"]
        assert tensors.metadata() == {"format": "flare-staging"}
        assert tensors.get_tensor("flare.payload").tobytes() == data
        assert proto.ModelIndex.FromString(tensors.get_tensor("flare.metadata").tobytes()) == description
    assert read_index(path) == (description, ALIGNMENT)


def test_metadata_growth_preserves_payload_bytes_and_offsets(tmp_path):
    source = tmp_path / "storage"
    source.write_bytes(b"unchanged weights")
    description = index()
    description.ranks[0].resources["storage"].CopyFrom(proto.Span(offset=4096, nbytes=source.stat().st_size))
    packages = []
    for generation in range(2):
        path = tmp_path / f"model-{generation}.safetensors"
        description.contract.model_id = "x" * (20000 * generation)
        publish(path, description, [(4096, source, 0, source.stat().st_size)])
        packages.append(path)
    assert [read_index(path)[1] for path in packages] == [ALIGNMENT, ALIGNMENT]
    with safe_open(packages[0], framework="numpy") as first, safe_open(packages[1], framework="numpy") as second:
        assert first.get_tensor("flare.payload").tobytes() == second.get_tensor("flare.payload").tobytes()
        assert first.get_slice("flare.metadata").get_shape() != second.get_slice("flare.metadata").get_shape()


def test_tp8_kernel_catalogs_fit_the_combined_index(tmp_path):
    ranks = []
    for rank in range(8):
        description = index()
        for function in range(8192):
            description.ranks[0].functions.add(module="kernels", symbol=f"kernel_{function}_" + "x" * 384)
        path = tmp_path / f"rank-{rank}.json"
        write_sources(path, description, [])
        ranks.append(path)
    combined = tmp_path / "model.safetensors"
    combine(combined, ranks, [])
    description, _ = read_index(combined)
    assert [len(rank.functions) for rank in description.ranks] == [8192] * 8
    assert description.ByteSize() > 16 << 20


def test_index_read_does_not_touch_the_payload(tmp_path):
    import re
    import shutil
    import subprocess
    import sys

    trace_program = shutil.which("strace")
    if trace_program is None:
        pytest.skip("requires strace to observe native file reads")
    description = index().SerializeToString()
    path = tmp_path / "model.safetensors"
    write_container(path, header(2 << 20, len(description)), bytes(2 << 20), description)
    trace = tmp_path / "reads.txt"
    result = subprocess.run([trace_program, "-yy", "-e", "trace=read,pread64", "-o", str(trace),
        sys.executable, "-c", "from ignition.preparation.package_file import read_index; "
        "import sys; index,base=read_index(sys.argv[1]); assert index.version==4 and base==4096",
        str(path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    reads = []
    for line in trace.read_text().splitlines():
        if f"<{path}>" not in line:
            continue
        assert line.startswith("pread64("), line
        match = re.search(r", (\d+), (\d+)\) = (\d+)$", line)
        assert match, line
        size, offset, returned = map(int, match.groups())
        assert returned == size
        reads.append((offset, size))
    assert reads == [(0, 8), (8, ALIGNMENT - 8), (ALIGNMENT + (2 << 20), len(description))]


@pytest.mark.parametrize("world", [1, 2, 3, 4, 8, 16])
def test_tp_weights_and_tables_precede_all_execution_data(tmp_path, world):
    outputs = []
    for generation in range(2):
        ranks = []
        for rank in range(world):
            contents = [b"graph" * (1 + generation * 2000), f"weights-{rank}".encode(), b"embedding-table"]
            source = tmp_path / f"source-{generation}-{rank}"
            source.write_bytes(b"".join(contents))
            description = index()
            bundle = description.ranks[0]
            bundle.allocations.add(name="weights", file="storage.bin")
            bundle.graphs.add(name="decode", resource="graph.bin")
            cursor = 0
            for key, data in zip(("graph.bin", "storage.bin"), contents):
                bundle.resources[key].CopyFrom(proto.Span(offset=cursor, nbytes=len(data)))
                cursor += len(data)
            if rank == 0:
                table = description.engram.tables.add(layer=1)
                table.weights.CopyFrom(proto.Span(offset=cursor, nbytes=len(contents[2])))
            path = tmp_path / f"rank-{generation}-{rank}.json"
            write_sources(path, description, [(0, source, 0, source.stat().st_size)])
            ranks.append(path)
        output = tmp_path / f"model-{generation}.safetensors"
        combine(output, ranks, [])
        description, start = read_index(output)
        storage = [bundle.resources["storage.bin"] for bundle in description.ranks]
        storage.append(description.engram.tables[0].weights)
        graphs = [bundle.resources["graph.bin"] for bundle in description.ranks]
        assert max(span.offset + span.nbytes for span in storage) <= min(span.offset for span in graphs)
        with output.open("rb") as source:
            for rank, bundle in enumerate(description.ranks):
                for name, expected in (("storage.bin", f"weights-{rank}".encode()),
                                       ("graph.bin", b"graph" * (1 + generation * 2000))):
                    span = bundle.resources[name]
                    source.seek(start + span.offset)
                    assert source.read(span.nbytes) == expected
            source.seek(start)
            outputs.append(source.read(min(span.offset for span in graphs)))
        with safe_open(output, framework="numpy") as tensors:
            assert tensors.get_slice("flare.payload").get_shape()[0] >= graphs[-1].offset + graphs[-1].nbytes
    assert outputs[0] == outputs[1]


def test_combined_tables_copy_every_recorded_shard_of_one_span(tmp_path):
    shards = [tmp_path / f"shard-{part}" for part in range(3)]
    for part, shard in enumerate(shards):
        shard.write_bytes(b"x" * part + f"table-part-{part}".encode())
    weights = tmp_path / "weights"
    weights.write_bytes(b"rank-weights")
    description = index()
    bundle = description.ranks[0]
    bundle.allocations.add(name="weights", file="storage.bin")
    bundle.resources["storage.bin"].CopyFrom(proto.Span(offset=0, nbytes=12))
    parts = [f"table-part-{part}".encode() for part in range(3)]
    table = description.engram.tables.add(layer=1)
    table.weights.CopyFrom(proto.Span(offset=12, nbytes=sum(map(len, parts))))
    payload, cursor = [(0, weights, 0, 12)], 12
    for number, part in enumerate(parts):
        payload.append((cursor, shards[number], number, len(part)))
        cursor += len(part)
    recorded = tmp_path / "rank.json"
    write_sources(recorded, description, payload)
    output = tmp_path / "model.safetensors"
    combine(output, [recorded], [])
    combined, start = read_index(output)
    span = combined.engram.tables[0].weights
    with output.open("rb") as source:
        source.seek(start + span.offset)
        assert source.read(span.nbytes) == b"".join(parts)
    # A span over a gap between its recorded ranges is rejected.
    gapped = index()
    gapped.ranks[0].CopyFrom(description.ranks[0])
    gapped.engram.CopyFrom(description.engram)
    gapped.engram.tables[0].weights.CopyFrom(proto.Span(offset=12, nbytes=len(parts[0]) + 2))
    path = tmp_path / "gapped.json"
    write_sources(path, gapped, [payload[0], payload[1], (13 + len(parts[0]), *payload[2][1:])])
    with pytest.raises(ValueError, match="outside its recorded sources"):
        combine(tmp_path / "gapped.safetensors", [path], [])


@pytest.mark.parametrize("field,value", [
    (("__metadata__", "format"), "pt"),
    (("flare.payload", "dtype"), "F32"),
    (("flare.payload", "shape"), [2, 2]),
    (("flare.payload", "shape"), [True]),
    (("flare.payload", "data_offsets"), [1, 5]),
    (("flare.payload", "data_offsets"), [-1, 3]),
    (("flare.metadata", "data_offsets"), [3, 7]),
    (("flare.metadata", "data_offsets"), [4, 1 << 64]),
    (("flare.metadata", "shape"), [MAX_INDEX_BYTES + 1]),
])
def test_invalid_safetensors_header_is_rejected(tmp_path, field, value):
    description = index().SerializeToString()
    container = header(4, len(description))
    container[field[0]][field[1]] = value
    path = tmp_path / "invalid.safetensors"
    write_container(path, container, b"data", description)
    with pytest.raises(ValueError):
        read_index(path)


def test_resources_cannot_address_protobuf_bytes(tmp_path):
    description = index()
    description.ranks[0].resources["outside"].CopyFrom(proto.Span(offset=4, nbytes=1))
    encoded = description.SerializeToString()
    path = tmp_path / "invalid.safetensors"
    write_container(path, header(4, len(encoded)), b"data", encoded)
    with pytest.raises(ValueError, match="resource exceeds"):
        read_index(path)


def test_index_size_is_bounded_before_reading(tmp_path):
    path = tmp_path / "invalid.safetensors"
    write_container(path, header(0, MAX_INDEX_BYTES + 1), b"", b"")
    with pytest.raises(ValueError, match="invalid model index size"):
        read_index(path)


@pytest.mark.parametrize("prefix", [b"", struct.pack("<Q", 0), struct.pack("<Q", MAX_HEADER_BYTES + 1)])
def test_header_size_is_bounded_before_reading(tmp_path, prefix):
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(prefix)
    with pytest.raises(ValueError):
        read_index(path)


def test_publication_failure_and_overwrite_leave_files_untouched(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"data")
    output = tmp_path / "model.safetensors"
    with pytest.raises(ValueError, match="source range exceeds file"):
        publish(output, index(), [(0, source, 0, 5)])
    assert list(tmp_path.iterdir()) == [source]
    publish(output, index(), [(0, source, 0, 4)])
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        publish(output, index(), [])
    assert output.read_bytes() == saved


def test_multi_host_package_combines_every_rank_and_preserves_transport_geometry(tmp_path):
    files, contents = [], [b"weights-zero" * 1000, b"weights-one" + bytes(5000)]
    for rank, data in enumerate(contents):
        source = tmp_path / f"source-{rank}"
        source.write_bytes(data)
        description = index()
        description.ranks[0].allocations.add(name="weights", file="storage.bin")
        description.ranks[0].resources["storage.bin"].CopyFrom(proto.Span(offset=0, nbytes=len(data)))
        if rank == 0:
            description.contract.model_id = "glm"
        description.ranks[0].roce.CopyFrom(proto.RoceTransport(
            proxy_abi=7, world=2, rank=rank, region_address=0x100000000000,
            region_bytes=1 << 20, counters_address=0x200000001000, counters_bytes=1024,
            slot_bytes=1 << 20, pipe_bytes=64 << 20, ctrl_offset=4 << 20,
            **({"varlen_bytes": 8 << 20} if rank == 1 else {})))
        files.append(tmp_path / f"rank-{rank}.json")
        write_sources(files[-1], description, [(0, source, 0, len(data))])
    output = tmp_path / "model.safetensors"
    combine(output, files, [], cluster=(2, "export-a"))
    result, start = read_index(output)
    assert (result.cluster.world, result.cluster.export_id) == (2, "export-a")
    assert result.contract.model_id == "glm"
    assert [bundle.roce.rank for bundle in result.ranks] == [0, 1]
    # A description from before the variable-length all-gather records none.
    assert [bundle.roce.varlen_bytes for bundle in result.ranks] == [0, 8 << 20]
    assert all(field not in proto.RoceTransport.DESCRIPTOR.fields_by_name
               for field in ("hcas", "gid_index", "traffic_class", "rails", "routes"))
    with output.open("rb") as package:
        for bundle, expected in zip(result.ranks, contents, strict=True):
            span = bundle.resources["storage.bin"]
            assert span.offset % ALIGNMENT == 0
            package.seek(start + span.offset)
            assert package.read(span.nbytes) == expected
    with pytest.raises(ValueError):
        combine(tmp_path / "incomplete.safetensors", files[:1], [], cluster=(2, "export-a"))
