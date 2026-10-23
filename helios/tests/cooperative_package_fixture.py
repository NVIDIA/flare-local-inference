"""Export cooperative worker grids that exceed the assigned GPU's residency limit."""

import ctypes as C
import json
from pathlib import Path
import subprocess
import struct
import sys
from types import SimpleNamespace

from ignition.preparation.runtime_package import (
    Allocation, Argument, Kernel, RuntimePackage, _Cuda,
)
from cuda_graph_fixture import KernelParams
from signing import publisher


def prepare(root):
    cuda = _Cuda()
    cuda.call("cuInit", C.c_uint(0))
    device, context = C.c_int(), C.c_void_p()
    cuda.call("cuDeviceGet", C.byref(device), C.c_int(0))
    cuda.call("cuCtxCreate_v2", C.byref(context), C.c_uint(0), device)
    attributes = []
    for attribute in (75, 76, 16):
        value = C.c_int()
        cuda.call("cuDeviceGetAttribute", C.byref(value), C.c_int(attribute), device)
        attributes.append(value.value)
    major, minor, sms = attributes
    source = root / "workers.cu"
    source.write_text('''
#include <cooperative_groups.h>
template<bool Cooperative> __device__ void work(int *out, int n) {
    int block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    int blocks = gridDim.x * gridDim.y * gridDim.z;
    int thread = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    int threads = blockDim.x * blockDim.y * blockDim.z;
    int start = block * threads + thread;
    for (int i = start; i < n; i += blocks * threads) out[i] = i * 3 + 7;
    if constexpr (Cooperative) {
        cooperative_groups::this_grid().sync();
        for (int i = start; i < n; i += blocks * threads) out[n + i] = out[n - 1 - i] + 11;
    }
    if (start == 0) {
        out[2 * n] = gridDim.x;
        out[2 * n + 1] = gridDim.y;
        out[2 * n + 2] = gridDim.z;
    }
}
extern "C" __global__ void cooperative(int *out, int n) { work<true>(out, n); }
extern "C" __global__ void ordinary(int *out, int n) { work<false>(out, n); }
''')
    cubin = root / "workers.cubin"
    subprocess.run(["nvcc", "--cubin", "-std=c++17", f"-arch=sm_{major}{minor}",
                    str(source), "-o", str(cubin)], check=True)
    module = C.c_void_p()
    cuda.call("cuModuleLoad", C.byref(module), C.c_char_p(str(cubin).encode()))
    functions = {}
    for name in ("cooperative", "ordinary"):
        function = C.c_void_p()
        cuda.call("cuModuleGetFunction", C.byref(function), module, C.c_char_p(name.encode()))
        cuda.call("cuFuncSetAttribute", function, C.c_int(8), C.c_int(65536))
        functions[name] = function
    elements = 500003
    allocations, graphs, expected, exported_grids = [], {}, {}, {}
    for name, block, shared, other, size, axis in (
        ("smaller", (32, 4, 1), 0, (1, 1), "small", 0),
        ("exact", (32, 4, 1), 0, (1, 1), "exact", 0),
        ("oversized", (32, 4, 1), 0, (1, 1), "large", 0),
        ("oversized_y", (32, 4, 1), 0, (1, 1), "large", 1),
        ("oversized_z", (32, 4, 1), 65536, (1, 1), "large", 2),
        ("thread_limited", (32, 8, 4), 0, (1, 1), "large", 0),
        ("shared_limited", (32, 4, 1), 65536, (1, 1), "large", 0),
        ("partitioned", (32, 4, 1), 0, (2, 3), "large", 0),
        ("partitioned_y", (32, 4, 1), 0, (2, 3), "large", 1),
        ("partitioned_z", (32, 4, 1), 0, (2, 3), "large", 2),
        ("ordinary", (32, 4, 1), 0, (1, 1), "large", 0),
    ):
        function = functions["ordinary" if name == "ordinary" else "cooperative"]
        occupancy = C.c_int()
        cuda.call("cuOccupancyMaxActiveBlocksPerMultiprocessor", C.byref(occupancy), function,
                  C.c_int(block[0] * block[1] * block[2]), C.c_size_t(shared))
        capacity = occupancy.value * sms
        max_workers = capacity // (other[0] * other[1])
        workers = 3 if size == "small" else max_workers if size == "exact" else max_workers + 7
        captured_grid, expected_grid = list(other), list(other)
        captured_grid.insert(axis, workers)
        expected_grid.insert(axis, workers if name == "ordinary" else min(workers, max_workers))
        output = C.c_uint64()
        nbytes = (elements * 2 + 3) * 4
        cuda.call("cuMemAlloc_v2", C.byref(output), C.c_size_t(nbytes))
        allocations.append(Allocation(name, output.value, nbytes, initialize="zero"))
        count = C.c_int(elements)
        arguments = (C.c_void_p * 2)(C.addressof(output), C.addressof(count))
        graph, node, executable = C.c_void_p(), C.c_void_p(), C.c_void_p()
        cuda.call("cuGraphCreate", C.byref(graph), C.c_uint(0))
        params = KernelParams(func=function.value, grid=(C.c_uint * 3)(*captured_grid),
                               block=(C.c_uint * 3)(*block), shared=shared, params=arguments)
        cuda.call("cuGraphAddKernelNode_v2", C.byref(node), graph, None, C.c_size_t(0), C.byref(params))
        if name != "ordinary":
            cooperative = (C.c_uint64 * 8)(1)
            status = cuda.lib.cuGraphKernelNodeSetAttribute(node, C.c_int(2), C.byref(cooperative))
            if size == "large":
                assert status == 720, f"oversized {name} grid returned CUDA status {status}"
                params.grid[axis] = expected_grid[axis]
                cuda.call("cuGraphKernelNodeSetParams_v2", node, C.byref(params))
                cuda.call("cuGraphKernelNodeSetAttribute", node, C.c_int(2), C.byref(cooperative))
            else:
                cuda.check("cuGraphKernelNodeSetAttribute", status)
        cuda.call("cuGraphInstantiateWithFlags", C.byref(executable), graph, C.c_uint64(0))
        graphs[name] = SimpleNamespace(raw_cuda_graph=lambda g=graph: g.value,
                                      raw_cuda_graph_exec=lambda e=executable: e.value)
        exported_grids[name] = captured_grid
        expected[name] = expected_grid

    package = RuntimePackage(allocations=allocations, modules={"workers": cubin},
        kernels=[Kernel(f.value, "workers", (Argument(8), Argument(4))) for f in functions.values()])
    stage = root / "rank.json"
    package.write(stage, graphs, sources=True)
    from ignition.preparation.package_file import combine, read_sources, write_sources
    from ignition.preparation import runtime_package_pb2 as proto

    index, locate = read_sources(stage)
    bundle = index.ranks[0]
    replacements = {}
    for graph in bundle.graphs:
        span = bundle.resources[graph.resource]
        pieces = locate(span)
        assert len(pieces) == 1
        source, offset, length = pieces[0]
        with source.open("rb") as stream:
            stream.seek(offset)
            data = stream.read(length)
        size, = struct.unpack_from("<I", data)
        node = proto.Node.FromString(data[4:4 + size])
        assert graph.node_count == 1 and graph.edge_count == 0
        assert node.WhichOneof("operation") == "kernel"
        address = int.from_bytes(node.kernel.arguments[0].data, "little")
        allocation = next(a for a in allocations if a.address == address)
        node.kernel.arguments[0].relocations.add(target=proto.Reference(allocation=allocation.name))
        # Encode the larger preparation grid without requiring a larger physical GPU.
        node.kernel.grid[:] = exported_grids[graph.name]
        encoded = node.SerializeToString()
        target = root / f"{graph.name}.graph"
        target.write_bytes(struct.pack("<I", len(encoded)) + encoded)
        replacements[graph.resource] = (target, target.stat().st_size)
    payload = []
    for name, span in bundle.resources.items():
        if name in replacements:
            source, length = replacements[name]
            span.nbytes = length
            payload.append((span.offset, source, 0, length))
        else:
            cursor = span.offset
            for source, offset, length in locate(span):
                payload.append((cursor, source, offset, length))
                cursor += length
    rewritten = root / "workers.json"
    write_sources(rewritten, index, sorted(payload))
    combine(root / "package", [rewritten], [], publisher=publisher())
    (root / "expected.json").write_text(json.dumps({"length": elements, "grids": expected}))
    cuda.call("cuCtxDestroy_v2", context)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    prepare(root)
