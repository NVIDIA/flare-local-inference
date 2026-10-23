"""Export graphs whose shared-memory requirements exceed the function's final allowance."""

import ctypes as C
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from ignition.preparation.runtime_package import (
    Allocation, Argument, Kernel, Reference, Relocation, RuntimePackage, _Cuda,
)
from cuda_graph_fixture import KernelParams
from signing import publisher


def prepare(root):
    cuda = _Cuda()
    cuda.call("cuInit", C.c_uint(0))
    device, context = C.c_int(), C.c_void_p()
    cuda.call("cuDeviceGet", C.byref(device), C.c_int(0))
    cuda.call("cuCtxCreate_v2", C.byref(context), C.c_uint(0), device)
    architecture = []
    for attribute in (75, 76):
        value = C.c_int()
        cuda.call("cuDeviceGetAttribute", C.byref(value), C.c_int(attribute), device)
        architecture.append(value.value)
    source = root / "shared.cu"
    source.write_text('''
extern "C" __global__ void shared_requirement(int *out, int bytes) {
    extern __shared__ unsigned char values[];
    for (int i = threadIdx.x; i < bytes; i += blockDim.x) values[i] = i % 251;
    __syncthreads();
    if (threadIdx.x == 0) out[0] = bytes + values[0] + values[bytes - 1];
}
''')
    cubin = root / "shared.cubin"
    subprocess.run(["nvcc", "--cubin", f"-arch=sm_{architecture[0]}{architecture[1]}",
                    str(source), "-o", str(cubin)], check=True)
    module, function = C.c_void_p(), C.c_void_p()
    image = C.create_string_buffer(cubin.read_bytes())
    cuda.call("cuModuleLoadData", C.byref(module), image)
    cuda.call("cuModuleGetFunction", C.byref(function), module, b"shared_requirement")
    cuda.call("cuFuncSetAttribute", function, C.c_int(8), C.c_int(96 << 10))
    output = C.c_uint64()
    cuda.call("cuMemAlloc_v2", C.byref(output), C.c_size_t(4))
    graphs = {}
    for shared in (64 << 10, 96 << 10, 60 << 10):
        count = C.c_int(shared)
        arguments = (C.c_void_p * 2)(C.addressof(output), C.addressof(count))
        graph, node, executable = C.c_void_p(), C.c_void_p(), C.c_void_p()
        cuda.call("cuGraphCreate", C.byref(graph), C.c_uint(0))
        params = KernelParams(func=function.value, grid=(C.c_uint * 3)(1, 1, 1),
            block=(C.c_uint * 3)(128, 1, 1), shared=shared, params=arguments)
        cuda.call("cuGraphAddKernelNode_v2", C.byref(node), graph, None, C.c_size_t(0), C.byref(params))
        cuda.call("cuGraphInstantiateWithFlags", C.byref(executable), graph, C.c_uint64(0))
        graphs[str(shared)] = SimpleNamespace(raw_cuda_graph=lambda g=graph: g.value,
            raw_cuda_graph_exec=lambda e=executable: e.value)
    # A function attribute can change after capture while the graph nodes keep their launch sizes.
    cuda.call("cuFuncSetAttribute", function, C.c_int(8), C.c_int(48 << 10))
    package = RuntimePackage(allocations=[Allocation("output", output.value, 4, initialize="zero")],
        modules={"shared": cubin}, kernels=[Kernel(function.value, "shared", (
            Argument(8, (Relocation(0, Reference("output")),)), Argument(4)))])
    package.write(root / "package", graphs, publisher=publisher())
    cuda.call("cuCtxDestroy_v2", context)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    prepare(root)
