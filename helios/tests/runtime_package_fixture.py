"""Build the b12x graph workload consumed by the standalone Rust replay test."""

import argparse
import ctypes as C
import json
from pathlib import Path
from signing import publisher


def prepare(root, shard_size=64 << 30):
    import torch
    from ignition.preparation.module_images import ModuleImages

    torch.cuda.set_device(0)
    with ModuleImages() as images:
        _prepare(root, images, shard_size)


def _prepare(root, images, shard_size):
    import torch
    import cutlass
    from cuda.bindings import driver as cuda
    try:
        from b12x._lib.dense_gemm import _DenseGemmPolicy, _get_compiled_dense_gemm
    except ImportError:
        from b12x._lib.dense_gemm.config import _DenseGemmPolicy
        from b12x._lib.dense_gemm.compile import _get_compiled_dense_gemm
    from ignition.preparation.runtime_package import Allocation, Argument, Kernel, Reference, Relocation, RuntimePackage

    torch.cuda.set_device(0)
    torch.manual_seed(21)
    settings = dict(
        n=128, k=128, l=1, c_l=1, a_major="k", b_major="k", c_major="n",
        ab_dtype=cutlass.Float8E4M3FN, sf_dtype=cutlass.Float8E8M0FNU,
        c_dtype=cutlass.BFloat16, alpha_dtype=cutlass.Float32, sf_vec_size=32,
        mma_k=32, tile_k=128, mma_tiler_mn=(64, 128), cluster_shape_mn=(1, 1),
        policy=_DenseGemmPolicy(False, False, False, 1, False, False),
        sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
        sm_version="sm_120", load_path="tma", swap_ab=False, sfb_k_reuse=False,
        b_tile_major=False, plain_fp8=True, alpha_is_one=True,
    )
    gemm = _get_compiled_dense_gemm(**settings)
    other_gemm = _get_compiled_dense_gemm(**{**settings, "n": 256})
    tensors = {
        "input": torch.randn(65, 128, 1, device="cuda").to(torch.float8_e4m3fn),
        "weight": torch.randn(128, 128, 1, device="cuda").to(torch.float8_e4m3fn),
        "scale": torch.zeros(1024, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu),
        "alpha": torch.ones(1, dtype=torch.float32, device="cuda"),
        "scratch": torch.zeros(65, 128, 1, dtype=torch.bfloat16, device="cuda"),
    }
    other_weight = torch.zeros((256, 128, 1), dtype=torch.float8_e4m3fn, device="cuda")
    other_output = torch.zeros((64, 256, 1), dtype=torch.bfloat16, device="cuda")
    other_gemm(tensors["input"][:64], other_weight, tensors["scale"], tensors["scale"],
               other_output, tensors["alpha"])
    graphs = {}
    for rows in (32, 64):
        source, scratch = tensors["input"][1:rows + 1], tensors["scratch"][1:rows + 1]
        output = torch.zeros(rows, 128, 1, dtype=torch.bfloat16, device="cuda")
        tensors[f"rows_{rows}"] = output
        def run():
            scratch.zero_()
            gemm(source, tensors["weight"], tensors["scale"], tensors["scale"],
                 scratch, tensors["alpha"])
            output.copy_(scratch)
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            run()
        graph.instantiate()
        graphs[f"rows_{rows}"] = graph
        graph.replay()
        torch.cuda.synchronize()
        (root / f"expected-{rows}.bin").write_bytes(output.view(torch.uint8).cpu().numpy().tobytes())
    status, nodes, count = cuda.cuGraphGetNodes(cuda.CUgraph(graph.raw_cuda_graph()), 64)
    assert not status and count == 3
    kernel_nodes = [node for node in nodes[:count] if cuda.cuGraphNodeGetType(node)[1] == 0]
    assert len(kernel_nodes) == 2
    parameters = [cuda.cuGraphKernelNodeGetParams(node)[1] for node in kernel_nodes]
    params, = [p for p in parameters if cuda.cuFuncGetName(p.func)[1].startswith(b"kernel_cutlass_")]
    fill, = [p for p in parameters if b"FillFunctor" in cuda.cuFuncGetName(p.func)[1]]
    assert not status
    function = int(params.func)
    status, symbol = cuda.cuFuncGetName(params.func)
    assert not status
    pointers = C.cast(params.kernelParams, C.POINTER(C.c_void_p))
    sizes = []
    for index in range(64):
        status, offset, size = cuda.cuFuncGetParamInfo(params.func, index)
        if status:
            assert status == cuda.CUresult.CUDA_ERROR_INVALID_VALUE
            break
        sizes.append(size)
    assert sizes == [128, 4, 16, 16, 8, 8, 128, 8, 128, 4, 16, 128, 8, 128,
                     4, 16, 16, 16, 8, 16, 16, 4, 12, 12, 4, 12, 12, 8, 8, 8]
    tensor_maps = {0: "input", 6: "weight", 8: "scale", 11: "scale", 13: "scratch"}
    pointer_fields = {2: "input", 3: "input", 4: "alpha", 5: "alpha", 7: "weight",
                      10: "scale", 12: "scale", 15: "scratch", 27: "alpha", 28: "alpha", 29: "alpha"}
    arguments = []
    def reference(name):
        return Reference(name, {"input": 128, "scratch": 256}.get(name, 0))
    for index, size in enumerate(sizes):
        relocations = ()
        if index in tensor_maps:
            relocations = (Relocation(0, reference(tensor_maps[index]), "tensor_map"),)
        elif index in pointer_fields:
            relocations = (Relocation(0, reference(pointer_fields[index])),)
        arguments.append(Argument(size, relocations))
    descriptor = C.string_at(pointers[0], 128)
    tensors["descriptor"] = torch.tensor(list(descriptor), dtype=torch.uint8, device="cuda")
    table_targets = (reference("input"), reference("weight"), reference("scratch"), reference("descriptor"))
    tensors["pointers"] = torch.tensor([tensors[r.allocation].data_ptr() + r.offset for r in table_targets],
                                      dtype=torch.int64, device="cuda")
    storage_relocations = {
        "descriptor": (Relocation(0, reference("input"), "tensor_map"),),
        "pointers": tuple(Relocation(index * 8, target) for index, target in enumerate(table_targets)),
    }
    tensors["payload"] = (torch.arange((20 << 20) + 17, device="cuda", dtype=torch.int32) % 251).to(torch.uint8)
    allocations = [Allocation(name, tensor.data_ptr(), tensor.numel() * tensor.element_size(),
                             initialize="zero" if name == "scratch" or name.startswith("rows_") else "snapshot",
                             relocations=storage_relocations.get(name, ()))
                   for name, tensor in tensors.items()]
    fill_arguments = (Argument(4), Argument(2), Argument(8, (Relocation(0, reference("scratch")),)))
    package = RuntimePackage(allocations=allocations, modules={}, recorder=images.native,
                             kernels=[Kernel(function, "dense", tuple(arguments)),
                                      Kernel(int(fill.func), "fill", fill_arguments)],
                             owners=(gemm, tensors))
    package.write(root / "package", graphs, publisher=publisher(), shard_size=shard_size,
                  rebuild_weights={"weight": tensors["weight"]})
    tensors["input"].copy_((tensors["input"].float() * 0.25 + 0.5).to(torch.float8_e4m3fn))
    (root / "modified-input.bin").write_bytes(tensors["input"].view(torch.uint8).cpu().numpy().tobytes())
    for rows in (32, 64):
        graphs[f"rows_{rows}"].replay()
        torch.cuda.synchronize()
        (root / f"modified-{rows}.bin").write_bytes(tensors[f"rows_{rows}"].view(torch.uint8).cpu().numpy().tobytes())
    (root / "expected.json").write_text(json.dumps({
        "addresses": {name: tensor.data_ptr() for name, tensor in tensors.items()},
        "outputs": {f"rows_{rows}": f"expected-{rows}.bin" for rows in (32, 64)},
        "modified_outputs": {f"rows_{rows}": f"modified-{rows}.bin" for rows in (32, 64)},
        "pointers": [{"allocation": ref.allocation, "offset": ref.offset} for ref in table_targets],
    }))
    print(f"exported {len(graphs)} b12x graphs and {len(allocations)} allocations", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--shard-size", type=int, default=64 << 30)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    prepare(args.directory, args.shard_size)
