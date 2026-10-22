"""Direct checkpoint transfers and compressed-scale preparation."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from functools import cached_property
from math import prod
import json
import os
import sys
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

from ignition.preparation.checkpoint import IndexedCheckpoint
from ignition.progress import phase
from ignition.preparation.quantization import projection_group_format, quantized_layers

if TYPE_CHECKING:
    import torch
    from b12x.gemm.blockscaled import Weight


@dataclass(frozen=True)
class MLPWeights:
    gate_up: Weight | torch.Tensor
    down: Weight | torch.Tensor


class SharedColumnReads:
    """Ranks on one host read each column-sharded checkpoint tensor once and copy their columns.

    A rank's column slice of a row-major tensor spans every row, so a direct read of the slice reads
    the whole tensor, and every rank of the host reads it again. Instead each tensor has one reader
    rank, which stages it whole in symmetric peer storage; after a barrier every rank copies its own
    columns from the readers' GPUs. Every rank must stage and copy the same tensors in the same order.
    """

    def __init__(self, group, peer_storage, device, capacity):
        import torch
        import torch.distributed as dist

        self.group, self.peer_storage, self.device = group, peer_storage, torch.device(device)
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.capacity = capacity
        self.pointer = peer_storage.cudaMalloc(capacity)
        handles = [None] * self.world
        dist.all_gather_object(handles, peer_storage.cudaIpcGetMemHandleBytes(self.pointer), group=group)
        self.bases = [self.pointer if rank == self.rank else peer_storage.cudaIpcOpenMemHandleBytes(handle)
                      for rank, handle in enumerate(handles)]

    def reader(self, index):
        return index % self.world

    def view(self, rank, offset, shape, dtype):
        """``rank``'s staged bytes at ``offset``, addressed from this rank's GPU."""
        import torch

        nbytes = prod(shape) * dtype.itemsize
        if offset < 0 or offset % 256 or offset + nbytes > self.capacity:
            raise ValueError("shared column staging lies outside its aligned capacity")
        storage = torch._C._construct_storage_from_data_pointer(self.bases[rank] + offset, self.device, nbytes)
        strides = tuple(prod(shape[index + 1:]) for index in range(len(shape)))
        return torch.empty(0, dtype=dtype, device=self.device).set_(storage, 0, shape, strides)

    def read_columns(self, loader, source, columns, out):
        """Fill ``out`` with ``source[:, columns]``: ranks read disjoint blocks of whole rows and copy their columns.

        Each round every rank stages one row block in its staging slot zero, so the reads of one
        tensor proceed on all ranks at once and each checkpoint byte is read once.
        """
        from ignition.preparation.tensor_metadata import record_numeric_tensor

        rows, width = source.shape
        first, last = columns
        if out.shape != (rows, last - first) or out.dtype != source.dtype:
            raise ValueError("shared column output must hold the selected columns of every row")
        block = max(1, min(-(-rows // self.world), self.capacity // (width * source.dtype.itemsize)))
        for start in range(0, rows, self.world * block):
            mine = start + self.rank * block
            if mine < rows:
                count = min(block, rows - mine)
                loader._copy(self.view(self.rank, 0, (count, width), source.dtype), source[mine:mine + count],
                             numeric=False)
            loader._session.flush()
            self.exchange()
            for reader in range(self.world):
                begin = start + reader * block
                if begin >= rows:
                    break
                count = min(block, rows - begin)
                out[begin:begin + count].copy_(self.view(reader, 0, (count, width), source.dtype)[:, first:last])
            self.exchange()
        record_numeric_tensor(out)
        return out

    def exchange(self):
        """Complete this rank's staging or copies on its GPU, then wait for every rank."""
        import torch
        import torch.distributed as dist

        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)

    def close(self):
        self.exchange()
        for rank, base in enumerate(self.bases):
            if rank != self.rank:
                self.peer_storage.cudaIpcCloseMemHandle(base)
        self.exchange()
        self.peer_storage.cudaFree(self.pointer)


def fp4_down_staging_bytes(hidden_size, intermediate_size, group):
    """Bytes one whole packed FP4 down projection and its block scales occupy in shared staging."""
    return -(-(hidden_size * (intermediate_size // 2 + intermediate_size // group)) // 4096) * 4096


def _checkpoint_read_mode(paths):
    """Use b12x's pinned-host reader for RAM files that GDS cannot register."""
    if sys.platform != "linux":
        return "auto"
    memory_devices = set()
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        mount, filesystem = line.split(' - ', 1)
        if filesystem.split()[0] in ('tmpfs', 'ramfs'):
            memory_devices.add(tuple(map(int, mount.split()[2].split(':'))))
    for path in paths:
        device = path.stat().st_dev
        if (os.major(device), os.minor(device)) in memory_devices:
            return "bounce"
    return _read_mode()


def quantize_bf16_linear_as_mxfp8(weight):
    """Convert a BF16 projection to b12x MXFP8 storage during loading."""
    import torch
    from b12x.gemm._shared.wo_mxfp8 import quantize_mxfp8_rows_torch
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight

    if (weight.ndim != 2 or weight.dtype != torch.bfloat16 or not weight.is_cuda
            or min(weight.shape) <= 0 or weight.shape[1] % 32):
        raise ValueError("MXFP8 conversion requires a BF16 CUDA matrix with 32-aligned input width")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("MXFP8 conversion requires finite BF16 weights")
    rows, columns = weight.shape
    padded = (columns + 127) // 128 * 128
    source = weight if columns == padded else torch.nn.functional.pad(weight, (0, padded - columns))
    return MXFP8LinearWeight(quantize_mxfp8_rows_torch(source), columns, padded, rows)


def _read_mode():
    """Checkpoint transport for a rank process that allocates only on its own GPU.

    Without the nvidia-fs driver, cuFile only emulates direct reads through host
    memory, yet still retains a context and bounce buffers on every visible GPU,
    about 0.5 GiB each. With many ranks per host those claims can exhaust a GPU
    before its weights fit. The bounce reader performs the same reads on its own
    device only.
    """
    import os

    return "auto" if os.path.exists("/proc/driver/nvidia-fs") else "bounce"


def _release_unused_contexts(device):
    """Destroy primary contexts that checkpoint reads opened on other visible GPUs.

    GPUDirect Storage reads retain a CUDA context, about 0.5 GiB, on every visible
    device. A rank process allocates only on its own device, so those contexts only
    take memory from its peers' KV pages.
    """
    import ctypes

    import torch

    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return
    for ordinal in range(torch.cuda.device_count()):
        if ordinal == device or torch.cuda.memory_reserved(ordinal):
            continue
        handle, flags, active = ctypes.c_int(), ctypes.c_uint(), ctypes.c_int()
        if (driver.cuDeviceGet(ctypes.byref(handle), ordinal) == 0
                and driver.cuDevicePrimaryCtxGetState(handle, ctypes.byref(flags), ctypes.byref(active)) == 0
                and active.value):
            driver.cuDevicePrimaryCtxReset_v2(handle)


class DirectWeightLoader:
    """Own b12x tensor views until direct transfers finish.

    Recipe subclasses select tensors and validate model-specific quantization.
    Source views may partition weight rows or columns without CPU weight copies.
    Compressed scale payloads are gathered and indexed on CUDA one layer at a time.
    """

    def __init__(
        self, manifest: IndexedCheckpoint, device: torch.device | str,
        *, prefixes: Sequence[str] | None = None,
        progress=None,
        progress_label: str = "Load weights",
    ):
        self.manifest = manifest
        self.device = device
        self.prefixes = None if prefixes is None else tuple(prefixes)
        if self.prefixes is not None and not self.prefixes:
            raise ValueError("prefixes must be nonempty when supplied")
        self.progress, self.progress_label = progress, progress_label
        self._display = None
        self._load_stats = None
        self._session = None
        self._sources = {}
        self._stack = None

    @cached_property
    def quantization(self):
        return quantized_layers(json.loads((self.manifest.metadata_path / "config.json").read_text()))

    def load_linear(self, prefix, *, rows=None, columns=None):
        """Load one projection using its checkpoint quantization declaration."""
        self._require_open()
        return self.load_linear_group((prefix,), row_ranges=(rows,), column_range=columns)

    def _load_nvfp4_linear(self, prefix, *, rows=None, columns=None):
        import torch
        from b12x._lib.intrinsics.reference import swizzle_block_scale
        from b12x.gemm.blockscaled import pack_weight

        source = self._sources[f"{prefix}.weight"]
        scale = self._sources[f"{prefix}.weight_scale"]
        scalar = self._sources[f"{prefix}.weight_scale_2"]
        if source.ndim != 2 or source.dtype != torch.uint8:
            raise ValueError(f"{prefix}: NVFP4 weight must be a packed uint8 matrix")
        n, packed_k = source.shape
        k = packed_k * 2
        if (k % 16 or scale.ndim != 2 or scale.dtype != torch.float8_e4m3fn
                or scale.shape[0] < n or scale.shape[1] < k // 16):
            raise ValueError(f"{prefix}: NVFP4 block scale geometry or dtype mismatch")
        if (scalar.dtype != torch.float32 or scalar.numel() != 1
                or not bool((torch.isfinite(scalar) & (scalar > 0)).all())):
            raise ValueError(f"{prefix}.weight_scale_2 must be a positive finite FP32 scalar")
        r0, r1 = _aligned_range(rows, n, 1)
        c0, c1 = _aligned_range(columns, k, 16)
        logical_k = c1 - c0
        stored_k = (logical_k + 127) // 128 * 128
        values = torch.zeros((r1 - r0, stored_k // 2), dtype=torch.uint8, device=self.device)
        scales = torch.zeros((r1 - r0, stored_k // 16), dtype=torch.float8_e4m3fn, device=self.device)
        gain = torch.empty(scalar.shape, dtype=torch.float32, device=self.device)
        self._copy(values[:, :logical_k // 2], source[r0:r1, c0 // 2:c1 // 2])
        self._copy(scales[:, :logical_k // 16], scale[r0:r1, c0 // 16:c1 // 16])
        self._copy(gain, scalar)
        activation_scale = None
        if self.quantization[prefix]["quant_algo"] == "NVFP4":
            activation = self._sources[f"{prefix}.input_scale"]
            if (activation.dtype != torch.float32 or activation.numel() != 1
                    or not bool((torch.isfinite(activation) & (activation > 0)).all())):
                raise ValueError(f"{prefix}.input_scale must be a positive finite FP32 scalar")
            reciprocal = activation.reciprocal()
            if not bool(torch.isfinite(reciprocal).all()):
                raise ValueError(f"{prefix}.input_scale reciprocal must remain finite")
            activation_scale = torch.empty_like(activation, device=self.device)
            self._copy(activation_scale, reciprocal)
        self._session.flush()
        weight = pack_weight(values, swizzle_block_scale(scales), recipe="nvfp4",
                             global_scale=gain, global_scale_kind="multiplier",
                             activation_global_scale=activation_scale)
        return replace(weight, in_features=logical_k)

    def _selected_names(self):
        return {name for name in self.manifest.tensors
                if self.prefixes is None or name.startswith(self.prefixes)}

    def _load_tensor_fp8_linear(self, prefix, *, rows=None, columns=None):
        import torch
        from b12x.gemm.blockscaled import pack_weight
        from ignition.components.tensor_fp8 import TensorFP8Weight

        source = self._sources[f"{prefix}.weight"]
        if source.ndim != 2 or source.dtype != torch.float8_e4m3fn:
            raise ValueError(f"{prefix}: tensor-FP8 weight must be an E4M3 matrix")
        r0, r1 = _aligned_range(rows, source.shape[0], 1)
        c0, c1 = _aligned_range(columns, source.shape[1], 32)
        values = torch.empty((r1 - r0, c1 - c0), dtype=source.dtype, device=self.device)
        self._copy(values, source[r0:r1, c0:c1])
        scales = []
        for suffix in ("weight_scale", "input_scale"):
            scalar = self._sources[f"{prefix}.{suffix}"]
            if scalar.dtype != torch.float32 or scalar.numel() != 1:
                raise ValueError(f"{prefix}.{suffix} must be a positive finite FP32 scalar")
            scale = torch.empty(scalar.shape, dtype=torch.float32, device=self.device)
            self._copy(scale, scalar)
            scales.append(scale)
        self._session.flush()
        for suffix, scale in zip(("weight_scale", "input_scale"), scales, strict=True):
            if not bool((torch.isfinite(scale) & (scale > 0)).all()):
                raise ValueError(f"{prefix}.{suffix} must be a positive finite FP32 scalar")
        weight_scale, input_scale = scales
        packed = pack_weight(values, weight_scale * input_scale, recipe="tensor_fp8")
        return TensorFP8Weight(packed, input_scale)

    @contextmanager
    def _loading_display(self):
        from b12x.loader import CheckpointDisplay

        with phase(self.progress, self.progress_label) as stage, CheckpointDisplay(
            console=stage.console, transient=True,
        ) as display:
            self._display = display
            started = monotonic()
            stats = {}
            yield stats
            display.complete(dict(ranks=1, payload_bytes=stats["payload_bytes"],
                                  physical_bytes=stats["physical_bytes"],
                                  load_seconds=monotonic() - started))

    def __enter__(self):
        import torch
        from b12x.loader import DirectWeightSession

        if self._stack is not None:
            raise RuntimeError("weight loader context is already open")
        device = torch.device(self.device)
        if device.type != "cuda":
            raise ValueError(f"{type(self).__name__} requires a CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        stack = ExitStack()
        try:
            if self.progress is not None:
                self._load_stats = stack.enter_context(self._loading_display())
            # Scalar calibration checks do not amortize intra-op thread dispatch.
            stack.callback(torch.set_num_threads, torch.get_num_threads())
            torch.set_num_threads(1)
            selected = self._selected_names()
            shards = {self.manifest.tensors[name].shard for name in selected}
            from ignition.preparation.checkpoint_source import source_for

            source = source_for(self.manifest.snapshot)
            if source is None:
                read_mode = _checkpoint_read_mode(self.manifest.snapshot / shard for shard in shards)
                self._session = stack.enter_context(DirectWeightSession(device=self.device.index, read_mode=read_mode))
            else:
                from ignition.preparation.remote_weights import RemoteWeightSession

                self._session = stack.enter_context(RemoteWeightSession(source, self.manifest, device=self.device.index))
            if self._display is not None:
                self._display.source(len(shards))

            def paths():
                for shard in sorted(shards):
                    if self._display is not None:
                        self._display.file(shard)
                    yield self.manifest.snapshot / shard
                    if self._display is not None:
                        self._display.advance(self._session.payload_bytes)

            # b12x validates source view mutations through tensor version counters.
            with torch.inference_mode(False):
                self._sources = dict(self._session.weights(
                    paths(),
                    index_path=self.manifest.index_path,
                    skip=lambda name: name not in selected,
                    needs_values=lambda entry: entry.name.endswith((
                        ".weight_scale_2", ".input_scale", ".weight_global_scale", ".input_global_scale",
                    )),
                ))
            if self._sources.keys() != selected:
                raise ValueError("direct loader tensor coverage differs from the inspected checkpoint")
            if self._display is not None:
                self._display.phase("execute")
            self._stack = stack
            return self
        except BaseException:
            stack.__exit__(*sys.exc_info())
            self._session = None
            self._display = self._load_stats = None
            self._sources.clear()
            raise

    def __exit__(self, *exc):
        try:
            if exc[0] is None and self._display is not None:
                try:
                    self._load_stats.update(self._session.stats())
                except BaseException:
                    self._stack.__exit__(*sys.exc_info())
                    raise
            return self._stack.__exit__(*exc)
        finally:
            self._stack = None
            self._session = None
            self._display = self._load_stats = None
            self._sources.clear()
            _release_unused_contexts(self.device.index)

    def _require_open(self):
        if self._session is None:
            raise RuntimeError(f"weight loading requires an open {type(self).__name__} context")

    def _copy(self, destination, source, *, numeric=True):
        from ignition.preparation.tensor_metadata import record_numeric_tensor

        if not self._session(destination, source):
            raise RuntimeError("b12x rejected the direct weight transfer")
        if numeric:
            record_numeric_tensor(destination)
        if self._display is not None:
            self._display.update(self._session.payload_bytes)

    def load_tensor(self, name: str, *, dtype: torch.dtype | None = None, extent=None) -> torch.Tensor:
        """Transfer a selected resident weight or declared scalar to the GPU."""
        target = self._queue_tensor(name, dtype=dtype, extent=extent)
        self._session.flush()
        return target

    def load_tensors(self, names: Sequence[str], *, extents=None) -> tuple[torch.Tensor, ...]:
        """Transfer independent weights together, returning ready GPU tensors."""
        self._require_open()
        if extents is None:
            extents = (None,) * len(names)
        targets = tuple(self._queue_tensor(name, extent=extent)
                        for name, extent in zip(names, extents, strict=True))
        self._session.flush()
        return targets

    def _queue_tensor(self, name, *, dtype=None, extent=None):
        import torch

        self._require_open()
        source = self._sources[name]
        if extent is not None:
            source = source[extent]
        target = torch.empty(source.shape, dtype=source.dtype if dtype is None else dtype, device=self.device)
        self._copy(target, source)
        return target

    def load_linear_group(self, prefixes, *, row_ranges=None, column_range=None):
        """Pack compatible projection rows without changing their quantization."""
        self._require_open()
        prefixes = tuple(prefixes)
        fmt = projection_group_format(self.quantization, prefixes)
        import torch
        from b12x.gemm.blockscaled import pack_weight
        if row_ranges is None:
            row_ranges = (None,) * len(prefixes)
        if fmt == "tensor_fp8":
            (rows,) = tuple(row_ranges)
            return self._load_tensor_fp8_linear(prefixes[0], rows=rows, columns=column_range)
        if fmt == "nvfp4":
            (rows,) = tuple(row_ranges)
            return self._load_nvfp4_linear(prefixes[0], rows=rows, columns=column_range)
        if fmt in ("iq2_xxs", "q8_0"):
            from b12x._lib.quant.block_codec import block_codec

            codec = block_codec(fmt)
            sources = []
            for prefix, row_range in zip(prefixes, row_ranges, strict=True):
                source = self._sources[f"{prefix}.weight"]
                if (source.ndim != 3 or source.dtype != torch.uint8
                        or source.shape[2] != codec.block_bytes):
                    raise ValueError(f"{prefix}: invalid {fmt} block payload")
                r0, r1 = _aligned_range(row_range, source.shape[0], 8)
                c0, c1 = _aligned_range(column_range, source.shape[1] * codec.block_weights, codec.block_weights)
                sources.append(source[r0:r1, c0 // codec.block_weights:c1 // codec.block_weights])
            columns = sources[0].shape[1]
            if any(source.shape[1] != columns for source in sources):
                raise ValueError("packed projections must share input width")
            values = torch.empty((sum(s.shape[0] for s in sources), columns, codec.block_bytes),
                                 dtype=torch.uint8, device=self.device)
            offset = 0
            for source in sources:
                self._copy(values[offset:offset + source.shape[0]], source)
                offset += source.shape[0]
            self._session.flush()
            return pack_weight(values, recipe=fmt)
        fp4 = fmt == "mxfp4"
        pack = 2 if fp4 else 1
        dtype = torch.uint8 if fp4 else torch.float8_e4m3fn
        weights, block_scales = [], []
        for prefix, row_range in zip(prefixes, row_ranges, strict=True):
            weight = self._sources[f"{prefix}.weight"]
            scale = self._sources[f"{prefix}.weight_scale"]
            if weight.ndim != 2 or weight.dtype != dtype:
                raise ValueError(f"{prefix}: {fmt.upper()} weight must be a {dtype} matrix")
            rows, stored_columns = weight.shape
            columns = stored_columns * pack
            r0, r1 = _aligned_range(row_range, rows, 1)
            c0, c1 = _aligned_range(column_range, columns, 32)
            if (scale.ndim != 2 or scale.dtype not in (torch.uint8, torch.float8_e8m0fnu)
                    or scale.shape[0] < rows or scale.shape[1] < columns // 32):
                raise ValueError(f"{prefix}: {fmt.upper()} block scale geometry mismatch")
            weights.append(weight[r0:r1, c0 // pack:c1 // pack])
            block_scales.append(scale[r0:r1, c0 // 32:c1 // 32].view(torch.uint8))
        columns = weights[0].shape[1] * pack
        if any(weight.shape[1] * pack != columns for weight in weights):
            raise ValueError("packed projections must share input width")
        rows = sum(weight.shape[0] for weight in weights)
        stored_columns = (columns + 255) // 256 * 256 if fp4 else columns
        values = torch.zeros((rows, stored_columns // pack), dtype=dtype, device=self.device)
        scales = torch.full((rows, stored_columns // 32), 127, dtype=torch.uint8, device=self.device)
        offset = 0
        for weight, scale in zip(weights, block_scales, strict=True):
            end = offset + weight.shape[0]
            self._copy(values[offset:end, :columns // pack], weight)
            self._copy(scales[offset:end, :columns // 32], scale)
            offset = end
        self._session.flush()
        if fp4:
            from b12x._lib.intrinsics.reference import swizzle_block_scale

            return replace(pack_weight(values, swizzle_block_scale(scales), recipe="mxfp4"),
                           in_features=columns)
        return pack_weight(values, scales, recipe="mxfp8")

    def load_block_quant_experts(self, prefixes, *, hidden_size, intermediate_size,
                                 intermediate_range=None, projection_names=("up_proj", "down_proj"),
                                 activation="relu2"):
        """Retile ungated GGML expert blocks without expanding or requantizing weights."""
        import torch
        from b12x._lib.quant.block_codec import block_codec
        from b12x.moe import fused_moe as moe
        from ignition.preparation.quantization import expert_format

        self._require_open()
        prefixes = tuple(prefixes)
        if not prefixes or activation != "relu2":
            raise ValueError("block-quantized expert loading requires nonempty ungated ReLU-squared experts")
        up, down = projection_names
        formats = {expert_format(self.quantization, f"{p}.{n}")
                   for p in prefixes for n in projection_names}
        if len(formats) != 1:
            raise ValueError("expert projections must share their block encoding")
        fmt, mode = formats.pop()
        if fmt not in ("iq2_xxs", "q8_0"):
            raise ValueError("expected IQ2_XXS or Q8_0 experts")
        codec = block_codec(fmt)
        first, last = _aligned_range(intermediate_range, intermediate_size, codec.block_weights)
        local = last - first
        if hidden_size % codec.block_weights:
            raise ValueError("expert hidden width must contain complete quantization blocks")
        w1 = torch.empty((len(prefixes), local, hidden_size // codec.block_weights, codec.block_bytes),
                         dtype=torch.uint8, device=self.device)
        w2 = torch.empty((len(prefixes), hidden_size, local // codec.block_weights, codec.block_bytes),
                         dtype=torch.uint8, device=self.device)
        for index, prefix in enumerate(prefixes):
            for projection, target, rows, columns in (
                    (up, w1[index], intermediate_size, hidden_size),
                    (down, w2[index], hidden_size, intermediate_size)):
                source = self._sources[f"{prefix}.{projection}.weight"]
                if source.dtype != torch.uint8 or source.shape != (rows, columns // codec.block_weights, codec.block_bytes):
                    raise ValueError(f"{prefix}.{projection}: invalid {fmt} expert geometry")
                source = source[:, first // codec.block_weights:last // codec.block_weights] if projection == down else source[first:last]
                self._copy(target, source)
        self._session.flush()
        plan = moe.plan_weights(source=moe.PackedSource(format=fmt),
            activation=moe.ActivationSpec(mode=mode, nonlinearity=activation, io_dtype=torch.bfloat16),
            geometry=moe.MoEGeometry(num_experts=len(prefixes), hidden_size=hidden_size, intermediate_size=local))
        return moe.prepare_weights(plan=plan, weights=moe.BlockQuantWeights(w1, w2, codec=fmt))

    def load_bf16_experts_as_nvfp4(
        self, prefix, *, num_experts, hidden_size, intermediate_size,
        intermediate_range=None, swiglu_limit=None, swiglu_clamp_after_silu=False,
        scale_compression=None, scale_scratch=None,
    ):
        """Quantize stacked BF16 expert shards once for NVFP4 W4A16 execution.

        Gate and up share one global scale per expert shard; down has its own.
        Staging holds one expert. Zero intermediate channels align packed W4A16
        tiles without changing the live channels or their TP partition.
        CSF compression requires two caller-owned scale expansion buffers.
        """
        import torch
        from flashinfer import fp4_quantize
        from b12x.moe import fused_moe as moe
        from ignition.preparation.tensor_metadata import record_numeric_tensor

        self._require_open()
        sources = {}
        for projection in ("up", "gate", "down"):
            name = f"{prefix}.{projection}_proj.weight"
            source = self._sources[name]
            shape = ((num_experts, hidden_size, intermediate_size) if projection == "down"
                     else (num_experts, intermediate_size, hidden_size))
            if source.dtype != torch.bfloat16 or source.shape != shape:
                raise ValueError(f"{name}: storage disagrees with BF16 expert geometry")
            sources[projection] = source
        if hidden_size % 128:
            raise ValueError("NVFP4 W4A16 experts require hidden size divisible by 128")
        first, last = _aligned_range(intermediate_range, intermediate_size, 16)
        local = last - first
        padded = (local + 127) // 128 * 128
        plan = moe.plan_weights(
            source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
            activation=moe.ActivationSpec(mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16,
                swiglu_limit=swiglu_limit, swiglu_clamp_after_silu=swiglu_clamp_after_silu),
            geometry=moe.MoEGeometry(num_experts=num_experts, hidden_size=hidden_size,
                                     intermediate_size=padded),
            constraints=moe.WeightPlanConstraints(scale_compression=scale_compression))
        w13 = torch.empty((num_experts, 2 * padded, hidden_size // 2), dtype=torch.uint8, device=self.device)
        w2 = torch.empty((num_experts, hidden_size, padded // 2), dtype=torch.uint8, device=self.device)
        s13 = torch.empty((num_experts, 2 * padded, hidden_size // 16), dtype=torch.float8_e4m3fn, device=self.device)
        s2 = torch.empty((num_experts, hidden_size, padded // 16), dtype=torch.float8_e4m3fn, device=self.device)
        g13, g2 = (torch.empty(num_experts, dtype=torch.float32, device=self.device) for _ in range(2))
        up_gate = torch.zeros((2 * padded, hidden_size), dtype=torch.bfloat16, device=self.device)
        down = torch.zeros((hidden_size, padded), dtype=torch.bfloat16, device=self.device)
        down_stage = torch.empty((hidden_size, local), dtype=torch.bfloat16, device=self.device)
        stream = torch.cuda.current_stream(self.device)
        stream.synchronize()
        for expert in range(num_experts):
            self._copy(up_gate[:local], sources["up"][expert, first:last])
            self._copy(up_gate[padded:padded + local], sources["gate"][expert, first:last])
            self._copy(down_stage, sources["down"][expert, :, first:last])
            self._session.flush()
            down[:, :local].copy_(down_stage)
            for source, values, scales, gains in ((up_gate, w13, s13, g13), (down, w2, s2, g2)):
                amax = source.abs().amax().float()
                if not bool(torch.isfinite(amax)):
                    raise ValueError(f"{prefix}: expert {expert} contains nonfinite BF16 weights")
                gain = torch.where(amax > 0, (amax / (448 * 6)).clamp_min(torch.finfo(torch.float32).tiny), 1.)
                packed, blocks = fp4_quantize(source, gain.reciprocal().reshape(1),
                    sf_vec_size=16, sf_use_ue8m0=False, is_sf_swizzled_layout=True,
                    nvfp4_4over6=None)
                values[expert].copy_(packed)
                scales[expert].copy_(blocks.view(torch.float8_e4m3fn))
                gains[expert].copy_(gain)
            # Direct transfers may use another stream; finish quantization before reusing staging.
            stream.synchronize()
        del up_gate, down, down_stage, source, packed, blocks
        for tensor in (w13, w2, s13, s2, g13, g2):
            record_numeric_tensor(tensor)
        return moe.prepare_weights(plan=plan, weights=moe.PackedWeights(
            w13=w13, w2=w2, w13_block_scales=s13, w2_block_scales=s2,
            w13_global_scales=g13, w2_global_scales=g2), scale_scratch=scale_scratch)

    def load_fp4_experts(self, prefixes, *, hidden_size, intermediate_size, format, activation_mode,
                         intermediate_range=None, scale_vectors=None, swiglu_limit=None,
                         swiglu_clamp_after_silu=False,
                         scale_compression=None, scale_scratch=None,
                         projection_names=("up_proj", "gate_proj", "down_proj"), scale_name="weight_scale",
                         weight_name="weight", activation="silu", w13_layout="w13", shared_columns=None):
        """Prepare NVFP4 or MXFP4 experts, optionally compressing their block scales.

        The first two projection names fill the first and second w13 halves in that
        order; ``w13_layout`` declares which halves hold up and gate. CSF compression
        requires two caller-owned expansion buffers. Serial routed layers may share
        those buffers; overlapping calls must not. With ``shared_columns``, each
        expert's down projection, whose intermediate columns partition across ranks,
        is read once per host (``SharedColumnReads``).
        """
        import torch
        from b12x.moe import fused_moe as moe
        from b12x._lib.intrinsics.reference import swizzle_block_scale
        from .sm103 import is_sm103
        from ignition.preparation.tensor_metadata import record_numeric_tensor

        self._require_open()
        prefixes = tuple(prefixes)
        sm103 = is_sm103(self.device)
        if sm103 and (scale_compression is not None or scale_scratch is not None):
            raise ValueError("SM103 FP4 experts do not support load-time scale compression")
        up_name, gate_name, down_name = projection_names
        if format not in ("nvfp4", "mxfp4"):
            raise ValueError("ordinary FP4 experts require NVFP4 or MXFP4")
        group = 16 if format == "nvfp4" else 32
        scale_dtype = torch.float8_e4m3fn if format == "nvfp4" else torch.uint8
        first, last = _aligned_range(intermediate_range, intermediate_size, group)
        local, count = last - first, len(prefixes)
        w13 = torch.empty((count, 2 * local, hidden_size // 2), dtype=torch.uint8, device=self.device)
        w2 = torch.empty((count, hidden_size, local // 2), dtype=torch.uint8, device=self.device)
        s13 = torch.empty((count, 2 * local, hidden_size // group), dtype=scale_dtype, device=self.device)
        s2 = torch.empty((count, hidden_size, local // group), dtype=scale_dtype, device=self.device)
        globals13, globals2 = [], []
        shared = shared_columns
        if shared is not None:
            down_bytes = hidden_size * (intermediate_size // 2)
            slot_bytes = fp4_down_staging_bytes(hidden_size, intermediate_size, group)
            if -(-count // shared.world) * slot_bytes > shared.capacity:
                raise ValueError("shared column staging cannot hold this rank's down projections")
        for index, prefix in enumerate(prefixes):
            scales = {}
            for projection, values, blocks in (
                (up_name, w13[index, :local], s13[index, :local]),
                (gate_name, w13[index, local:], s13[index, local:]),
                (down_name, w2[index], s2[index]),
            ):
                source = self._sources[f"{prefix}.{projection}.{weight_name}"]
                scale = self._sources[f"{prefix}.{projection}.{scale_name}"]
                if format == "mxfp4":
                    if source.dtype == torch.int8:
                        source = source.view(torch.uint8)
                    if scale.dtype == torch.float8_e8m0fnu:
                        scale = scale.view(torch.uint8)
                rows, columns = ((hidden_size, intermediate_size) if projection == down_name
                                 else (intermediate_size, hidden_size))
                if source.dtype != torch.uint8 or source.shape != (rows, columns // 2):
                    raise ValueError(f"{prefix}.{projection}: packed FP4 weight geometry mismatch")
                if scale.shape != (rows, columns // group) or scale.dtype != scale_dtype:
                    raise ValueError(f"{prefix}.{projection}: {format.upper()} block scale geometry or dtype mismatch")
                if projection == down_name and shared is not None:
                    # The reader stages the whole projection; its rows are contiguous in the checkpoint.
                    if shared.reader(index) == shared.rank:
                        offset = index // shared.world * slot_bytes
                        self._copy(shared.view(shared.rank, offset, source.shape, torch.uint8), source, numeric=False)
                        self._copy(shared.view(shared.rank, offset + down_bytes, scale.shape, scale_dtype), scale,
                                   numeric=False)
                else:
                    self._copy(values, source[:, first // 2:last // 2] if projection == down_name
                               else source[first:last])
                    self._copy(blocks, scale[:, first // group:last // group] if projection == down_name
                               else scale[first:last])
                if format == "mxfp4":
                    continue
                value = self._sources[f"{prefix}.{projection}.weight_scale_2"]
                if value.dtype != torch.float32 or value.numel() != 1 or not bool(torch.isfinite(value) & (value > 0)):
                    raise ValueError(f"{prefix}.{projection}: expected a positive FP32 global weight scale")
                scales[projection] = value.item()
            if format == "mxfp4":
                continue
            if scales[gate_name] != scales[up_name]:
                raise ValueError(f"{prefix}: gate/up global scales differ")
            globals13.append(scales[up_name])
            globals2.append(scales[down_name])
        if shared is not None:
            # Every rank copies its intermediate columns of each staged down projection.
            self._session.flush()
            shared.exchange()
            for index in range(count):
                reader, offset = shared.reader(index), index // shared.world * slot_bytes
                down = shared.view(reader, offset, (hidden_size, intermediate_size // 2), torch.uint8)
                blocks = shared.view(reader, offset + down_bytes, (hidden_size, intermediate_size // group),
                                     scale_dtype)
                w2[index].copy_(down[:, first // 2:last // 2])
                s2[index].copy_(blocks[:, first // group:last // group])
            record_numeric_tensor(w2)
            record_numeric_tensor(s2)
            shared.exchange()
        if activation_mode == "a4":
            if scale_vectors is None or len(scale_vectors) != 4:
                raise ValueError("NVFP4 A4 experts require weight and activation scale vectors")
            vectors = scale_vectors
        elif format == "nvfp4":
            vectors = (torch.tensor(globals13, dtype=torch.float32), torch.tensor(globals2, dtype=torch.float32))
        else:
            vectors = (torch.ones(count, dtype=torch.float32), torch.ones(count, dtype=torch.float32))
        device_scales = []
        for values in vectors:
            if (values.shape != (count,) or values.dtype != torch.float32
                    or not bool((torch.isfinite(values) & (values > 0)).all())):
                raise ValueError("expert scales must be positive finite FP32 vectors")
            target = torch.empty(count, dtype=torch.float32, device=self.device)
            self._copy(target, values)
            device_scales.append(target)
        self._session.flush()
        if sm103:
            if swiglu_clamp_after_silu:
                raise ValueError("SM103 FP4 experts do not support post-SiLU clamping")
            return _sm103_fp4_experts(w13, s13, w2, s2, vectors, format=format,
                                      activation_mode=activation_mode, swiglu_limit=swiglu_limit)
        plan = moe.plan_weights(
            source=moe.PackedSource(format="modelopt_nvfp4" if format == "nvfp4" else "fp4_e8m0_k32",
                                    w13_layout=w13_layout),
            activation=moe.ActivationSpec(mode=activation_mode, nonlinearity=activation, io_dtype=torch.bfloat16,
                                          swiglu_limit=swiglu_limit,
                                          swiglu_clamp_after_silu=swiglu_clamp_after_silu),
            geometry=moe.MoEGeometry(num_experts=count, hidden_size=hidden_size, intermediate_size=local),
            constraints=moe.WeightPlanConstraints(scale_compression=scale_compression))
        gains = {} if not device_scales else dict(w13_global_scales=device_scales[0], w2_global_scales=device_scales[1])
        if activation_mode == "a4":
            gains.update(input_scale=device_scales[2], intermediate_scale=device_scales[3], immutable_input_scales=True)
        packed = moe.PackedWeights(
            w13=w13, w2=w2,
            w13_block_scales=swizzle_block_scale(s13) if format == "nvfp4" else s13,
            w2_block_scales=swizzle_block_scale(s2) if format == "nvfp4" else s2, **gains)
        del s13, s2, blocks
        return moe.prepare_weights(plan=plan, weights=packed, scale_scratch=scale_scratch)

    def load_mxfp4_experts(
        self, prefixes, *, hidden_size, intermediate_size, scale_scratch,
        intermediate_range=None, swiglu_limit=None, scale_compression=None,
        projection_names=("up_proj", "gate_proj", "down_proj"), scale_name="weight_scale",
    ):
        """Load MXFP4 storage, optionally compressing native scales losslessly.

        ``scale_scratch`` is a caller-owned dictionary keyed by expert geometry.
        Sharing it between calls requires serialized routed-expert execution;
        overlapping expert branches must provide separate dictionaries.
        ``scale_compression="csf"`` losslessly encodes native scales during loading;
        checkpoints that already contain CSF retain their encoded storage.
        """
        import torch

        self._require_open()
        if scale_compression not in (None, "csf"):
            raise ValueError("MXFP4 expert scale compression must be None or 'csf'")
        prefixes = tuple(prefixes)
        compressed = getattr(self.manifest, "compressed_scales", frozenset())
        encoded = [f"{prefix}.{projection}.{scale_name}" in compressed
                   for prefix in prefixes for projection in projection_names]
        geometry = dict(hidden_size=hidden_size, intermediate_size=intermediate_size,
                        intermediate_range=intermediate_range, swiglu_limit=swiglu_limit,
                        scale_name=scale_name)
        if not any(encoded) and scale_compression is None:
            return self.load_fp4_experts(prefixes, **geometry, format="mxfp4", activation_mode="a8",
                                         projection_names=projection_names)
        if any(encoded) and not all(encoded):
            raise ValueError("each routed expert group must use one complete MXFP4 scale encoding")
        first, last = _aligned_range(intermediate_range, intermediate_size, 32)
        local, count = last - first, len(prefixes)
        key = count, hidden_size, local
        if key not in scale_scratch:
            scale_scratch[key] = tuple(torch.empty(shape, dtype=torch.uint8, device=self.device)
                for shape in ((count, hidden_size // 32, 2 * local), (count, local // 32, hidden_size)))
        if not any(encoded):
            return self.load_fp4_experts(prefixes, **geometry, format="mxfp4", activation_mode="a8",
                projection_names=projection_names, scale_compression=scale_compression,
                scale_scratch=scale_scratch[key])
        up, gate, down = projection_names
        return self.load_mxfp4_csf_experts(prefixes, **geometry,
            scale_scratch=scale_scratch[key], projection_names=(gate, up, down))

    def load_mxfp4_csf_experts(
        self, prefixes, *, hidden_size, intermediate_size, scale_scratch,
        intermediate_range=None, swiglu_limit=None,
        projection_names=("gate_proj", "up_proj", "down_proj"), scale_name="weight_scale",
        weight_name="weight", activation="silu", activation_mode="a8",
    ):
        """Keep lossless MXFP4 scales compressed and expand into shared native scratch.

        Gate and up weights use b12x's w31 source order. The caller may share
        scale scratch between serialized expert layers, retaining both buffers
        until every captured graph using those layers has drained. Activations
        are quantized to MXFP8 (``a8``) or kept in BF16 (``a16``).
        """
        import torch
        from b12x.moe import fused_moe as moe

        self._require_open()
        prefixes = tuple(prefixes)
        if not prefixes or hidden_size <= 0 or hidden_size % 128:
            raise ValueError("MXFP4-CSF experts require nonempty prefixes and 128-aligned hidden width")
        first, last = _aligned_range(intermediate_range, intermediate_size, 32)
        local, count = last - first, len(prefixes)
        shapes = ((count, hidden_size // 32, 2 * local),
                  (count, local // 32, hidden_size))
        if len(scale_scratch) != 2:
            raise ValueError("MXFP4-CSF experts require two caller-owned scale buffers")
        for scratch, shape in zip(scale_scratch, shapes, strict=True):
            if (scratch.device != torch.device(self.device) or scratch.dtype != torch.uint8
                    or scratch.numel() < prod(shape) or not scratch.is_contiguous()):
                raise ValueError("MXFP4-CSF scale scratch shape, dtype, device, or layout mismatch")
        gate, up, down = projection_names
        w13 = torch.empty((count, 2 * local, hidden_size // 2), dtype=torch.uint8, device=self.device)
        w2 = torch.empty((count, hidden_size, local // 2), dtype=torch.uint8, device=self.device)
        for index, prefix in enumerate(prefixes):
            for projection, destination, rows, columns in (
                (gate, w13[index, :local], intermediate_size, hidden_size),
                (up, w13[index, local:], intermediate_size, hidden_size),
                (down, w2[index], hidden_size, intermediate_size),
            ):
                source = self._sources[f"{prefix}.{projection}.{weight_name}"]
                if source.dtype == torch.int8:
                    source = source.view(torch.uint8)
                if source.dtype != torch.uint8 or source.shape != (rows, columns // 2):
                    raise ValueError(f"{prefix}.{projection}: packed MXFP4 weight geometry mismatch")
                self._copy(destination, source[:, first // 2:last // 2]
                           if projection == down else source[first:last])
        scales13 = self._queue_csf_sources(
            [f"{prefix}.{projection}.{scale_name}" for prefix in prefixes for projection in (gate, up)],
            rows=intermediate_size, columns=hidden_size // 32, quantization="mxfp4",
            join=2, row_range=(first, last))
        scales2 = self._queue_csf_sources(
            [f"{prefix}.{down}.{scale_name}" for prefix in prefixes],
            rows=hidden_size, columns=intermediate_size // 32, quantization="mxfp4",
            column_range=(first // 32, last // 32))
        self._session.flush()
        plan = moe.plan_weights(
            source=moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w31"),
            activation=moe.ActivationSpec(mode=activation_mode, nonlinearity=activation, io_dtype=torch.bfloat16,
                                          swiglu_limit=swiglu_limit),
            geometry=moe.MoEGeometry(num_experts=count, hidden_size=hidden_size, intermediate_size=local))
        return moe.prepare_weights(plan=plan, weights=moe.Mxfp4CsfWeights(
            w13=w13, w2=w2, w13_scales=scales13, w2_scales=scales2,
            w13_scale_scratch=scale_scratch[0], w2_scale_scratch=scale_scratch[1]))

    def load_nvfp4_experts(
        self, prefixes, *, hidden_size, intermediate_size, scale_vectors,
        intermediate_range=None, scale_scratch=None, swiglu_limit=None, activation_mode="a4",
    ):
        """Prepare routed A4 experts from packed rank-local checkpoint views.

        ``scale_vectors`` supplies up/gate and down global multipliers followed
        by their activation reciprocal scales. Gate/up scalar compatibility is
        the recipe's responsibility. Without ``scale_scratch`` (and always on
        SM103) the CSF scales expand once here into persistent storage. With it,
        each call expands them into the scratch, which serialized layers can
        share; both buffers remain caller-owned through replay.
        """
        import torch
        from b12x.moe import fused_moe as moe

        self._require_open()
        prefixes = tuple(prefixes)
        if not prefixes or hidden_size <= 0 or hidden_size % 128:
            raise ValueError("NVFP4 experts require nonempty prefixes and 128-aligned hidden width")
        first, last = _aligned_range(intermediate_range, intermediate_size, 32)
        local = last - first
        count = len(prefixes)
        shapes = ((count, (2 * local + 127) // 128 * 128, hidden_size // 16),
                  (count, hidden_size, (local // 16 + 3) // 4 * 4))
        from .sm103 import is_sm103

        sm103 = is_sm103(self.device)
        # Expanding once at load spends the expanded scales' memory to keep the expansion,
        # its scratch writes and their rereads off every forward's critical path.
        expand = sm103 or scale_scratch is None
        for scratch, shape in zip(() if scale_scratch is None else scale_scratch,
                                  () if scale_scratch is None else shapes, strict=True):
            if (scratch.device != self.device or scratch.dtype != torch.float8_e4m3fn
                    or tuple(scratch.shape) != shape or not scratch.is_contiguous()):
                raise ValueError("expert scale scratch shape, dtype, device, or layout mismatch")
        if len(scale_vectors) != 4 or any(
            scale.device.type != "cpu" or scale.dtype != torch.float32 or scale.shape != (count,)
            or not bool((torch.isfinite(scale) & (scale > 0)).all()) for scale in scale_vectors
        ):
            raise ValueError("expert scales require four positive finite CPU float32 vectors")

        w13 = torch.empty((count, 2 * local, hidden_size // 2), dtype=torch.uint8, device=self.device)
        w2 = torch.empty((count, hidden_size, local // 2), dtype=torch.uint8, device=self.device)
        for index, prefix in enumerate(prefixes):
            for projection, destination, rows, columns, rr, cr in (
                ("up_proj", w13[index, :local], intermediate_size, hidden_size, (first, last), None),
                ("gate_proj", w13[index, local:], intermediate_size, hidden_size, (first, last), None),
                ("down_proj", w2[index], hidden_size, intermediate_size, None, (first, last)),
            ):
                source = self._sources[f"{prefix}.{projection}.weight"]
                if source.dtype != torch.uint8 or source.shape != (rows, columns // 2):
                    raise ValueError(f"{prefix}.{projection}: packed NVFP4 weight geometry mismatch")
                r0, r1 = (0, rows) if rr is None else rr
                c0, c1 = (0, columns) if cr is None else cr
                self._copy(destination, source[r0:r1, c0 // 2:c1 // 2])
        device_scales = []
        for scale in scale_vectors:
            target = torch.empty_like(scale, device=self.device)
            self._copy(target, scale)
            device_scales.append(target)
        scales13 = self._queue_csf_sources(
            [f"{prefix}.{projection}.weight_scale" for prefix in prefixes
             for projection in ("up_proj", "gate_proj")],
            rows=intermediate_size, columns=hidden_size // 16, join=2,
            row_range=(first, last))
        scales2 = self._queue_csf_sources(
            [f"{prefix}.down_proj.weight_scale" for prefix in prefixes],
            rows=hidden_size, columns=intermediate_size // 16,
            column_range=(first // 16, last // 16))
        self._session.flush()
        if self._display is not None:
            self._display.update(self._session.payload_bytes)
        if expand:
            from b12x._lib.quant._csf_prepare import expand_source_scales, validate_source

            # SM103's kernels take row-major scales; b12x takes the native swizzled F8_128x4 grid.
            native = ((count, 2 * local, hidden_size // 16), (count, hidden_size, local // 16))
            expanded = []
            for source, shape in zip((scales13, scales2), native if sm103 else shapes, strict=True):
                validate_source(source)
                expanded.append(torch.empty(shape, dtype=torch.uint8 if sm103 else torch.float8_e4m3fn,
                                            device=self.device))
                expand_source_scales(source, expanded[-1], swizzled=not sm103)
            if sm103:
                return _sm103_fp4_experts(w13, expanded[0], w2, expanded[1], scale_vectors, format="nvfp4",
                                          activation_mode=activation_mode, swiglu_limit=swiglu_limit)
            block_scales = expanded
        else:
            block_scales = scale_scratch
        plan = moe.plan_weights(
            source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
            activation=moe.ActivationSpec(mode=activation_mode, nonlinearity="silu", io_dtype=torch.bfloat16,
                                          swiglu_limit=swiglu_limit),
            geometry=moe.MoEGeometry(num_experts=count, hidden_size=hidden_size, intermediate_size=local),
        )
        packed = moe.PackedWeights(
            w13=w13, w2=w2, w13_block_scales=block_scales[0], w2_block_scales=block_scales[1],
            w13_global_scales=device_scales[0], w2_global_scales=device_scales[1],
            input_scale=device_scales[2], intermediate_scale=device_scales[3],
            immutable_input_scales=True,
        )
        if expand:
            return moe.prepare_weights(plan=plan, weights=packed)
        return moe.prepare_weights(plan=plan, weights=moe.Nvfp4CsfWeights(
            packed=packed, w13_scales=scales13, w2_scales=scales2))

    def _queue_csf_sources(self, prefixes, *, rows, columns, quantization="nvfp4", **selection):
        from b12x.loader._csf_source import stage_csf_sources
        from ignition.preparation.tensor_metadata import record_numeric_tensor

        encoding = "csf_32x2_v1" if f"{prefixes[0]}.csf32_bases" in self._sources else "csf"
        suffixes = (tuple(f"csf32_{field}" for field in
                    ("bases", "codes", "page_bases", "offsets", "exceptions"))
                    if encoding == "csf_32x2_v1" else (f"{quantization}_csf_fixed", f"{quantization}_csf_exceptions"))
        components = [tuple(self._sources[f"{prefix}.{suffix}"] for suffix in suffixes) for prefix in prefixes]
        copy = self._session
        if encoding == "csf" and quantization == "mxfp4":
            from types import SimpleNamespace

            # Lossless MXFP4 checkpoints store each slab stream flat; this view changes only metadata.
            components = [(fixed.reshape(rows // 16, 16 * (1 + (columns + 7) // 8)), exceptions)
                          for fixed, exceptions in components]
            # Direct transfers validate the original checkpoint geometry. Only the destination
            # needs the slab shape; source byte streams remain whole, unmodified flat tensors.
            copy = SimpleNamespace(concatenate=lambda destination, sources: self._session.concatenate(
                destination, [source.reshape(-1) for source in sources]))
        result = stage_csf_sources(
            components, rows=rows, columns=columns, quantization=quantization, device=self.device,
            encoding=encoding, copy=copy, **selection)
        for tensor in (*result.components, result.starts):
            record_numeric_tensor(tensor)
        return result


def _sm103_fp4_experts(w13, s13, w2, s2, scale_vectors, *, format, activation_mode, swiglu_limit):
    """Prepare packed NVFP4 experts with row-major scales for FlashInfer's SM103 MoE kernels."""
    from .moe_sm103 import prepare_cute_w4a16_experts, prepare_trtllm_nvfp4_experts

    if format != "nvfp4":
        raise ValueError("SM103 routed experts support NVFP4 storage")
    if activation_mode == "a4":
        return prepare_trtllm_nvfp4_experts(w13, s13, w2, s2, scale_vectors, swiglu_limit=swiglu_limit)
    if activation_mode == "a16":
        return prepare_cute_w4a16_experts(w13, s13, w2, s2, *scale_vectors[:2], swiglu_limit=swiglu_limit)
    raise ValueError(f"SM103 routed experts do not support activation mode {activation_mode!r}")


def _aligned_range(selected, extent, alignment):
    start, end = (0, extent) if selected is None else selected
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= extent
            or start % alignment or end % alignment):
        raise ValueError("weight partition must be aligned, nonempty and in range")
    return start, end

