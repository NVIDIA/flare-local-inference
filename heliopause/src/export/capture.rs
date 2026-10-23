//! Capture prepared CUDA graph declarations and their reachable storage.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use libloading::Library;
use prost::Message;

use super::{
    Allocation, Argument, CaptureRequest, ExportResult, FilePart, FileSource, Kernel, RankOutput,
    Relocation, cubin, invalid,
    reachability::{Reachability, merge_ranges, subtract_ranges},
    recorder::Recorder,
    wire,
};

type Handle = *mut c_void;
macro_rules! status {
    ($cuda:expr, $name:literal, ($($ty:ty),*), $($arg:expr),* $(,)?) => {{
        let function = unsafe { $cuda.symbol::<unsafe extern "C" fn($($ty),*) -> c_int>(concat!($name, "\0").as_bytes())? };
        unsafe { function($($arg),*) }
    }};
    ($cuda:expr, $name:literal, ()) => {{
        let function = unsafe { $cuda.symbol::<unsafe extern "C" fn() -> c_int>(concat!($name, "\0").as_bytes())? };
        unsafe { function() }
    }};
}
macro_rules! cu {
    ($cuda:expr, $name:literal, ($($ty:ty),*), $($arg:expr),* $(,)?) => {{
        let status = status!($cuda, $name, ($($ty),*), $($arg),*);
        $cuda.check($name, status)
    }};
    ($cuda:expr, $name:literal, ()) => {{ $cuda.check($name, status!($cuda, $name, ())) }};
}

struct Cuda {
    library: Library,
}
impl Cuda {
    fn new() -> ExportResult<Self> {
        #[cfg(windows)]
        let library =
            unsafe { libloading::os::windows::Library::load_with_flags("nvcuda.dll", 0x800)? }
                .into();
        #[cfg(not(windows))]
        let library = unsafe { Library::new("libcuda.so.1")? };
        Ok(Self { library })
    }
    unsafe fn symbol<T: Copy>(&self, name: &[u8]) -> ExportResult<T> {
        Ok(*unsafe { self.library.get::<T>(name)? })
    }
    fn check(&self, name: &str, status: c_int) -> ExportResult<()> {
        if status == 0 {
            return Ok(());
        }
        let mut message = std::ptr::null();
        let text = unsafe {
            self.symbol::<unsafe extern "C" fn(c_int, *mut *const c_char) -> c_int>(
                b"cuGetErrorString\0",
            )
        };
        if let Ok(function) = text {
            unsafe {
                function(status, &mut message);
            }
        }
        let message = if message.is_null() {
            "unknown error".into()
        } else {
            unsafe { CStr::from_ptr(message) }.to_string_lossy()
        };
        Err(invalid(format!("{name}: CUDA {status}: {message}")))
    }
    fn current(&self) -> ExportResult<usize> {
        let mut context = std::ptr::null_mut();
        cu!(self, "cuCtxGetCurrent", (*mut Handle), &mut context)?;
        Ok(context as usize)
    }
    fn synchronize(&self) -> ExportResult<()> {
        cu!(self, "cuCtxSynchronize", ())
    }
    fn is_device(&self, address: u64) -> ExportResult<bool> {
        let mut memory = 0i32;
        cu!(
            self,
            "cuPointerGetAttribute",
            (*mut c_void, c_int, u64),
            (&mut memory as *mut i32).cast(),
            2,
            address
        )?;
        Ok(memory == 2)
    }
}

struct ContextGuard<'a> {
    cuda: &'a Cuda,
    previous: usize,
}
impl<'a> ContextGuard<'a> {
    fn enter(cuda: &'a Cuda, context: usize) -> ExportResult<Self> {
        if context == 0 {
            return Err(invalid("export requires a live CUDA context"));
        }
        let previous = cuda.current()?;
        cu!(cuda, "cuCtxSetCurrent", (Handle), context as Handle)?;
        Ok(Self { cuda, previous })
    }
}
impl Drop for ContextGuard<'_> {
    fn drop(&mut self) {
        let _ = (|| -> ExportResult<()> {
            cu!(
                self.cuda,
                "cuCtxSetCurrent",
                (Handle),
                self.previous as Handle
            )
        })();
    }
}

#[repr(C)]
#[derive(Default)]
struct KernelParams {
    function: Handle,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    params: *mut Handle,
    extra: *mut Handle,
    kernel: Handle,
    context: Handle,
}
#[repr(C)]
#[derive(Default)]
struct MemsetParams {
    destination: u64,
    pitch: usize,
    value: u32,
    element_size: u32,
    width: usize,
    height: usize,
}
#[repr(C)]
#[derive(Default)]
struct CopyEndpoint {
    x: usize,
    y: usize,
    z: usize,
    lod: usize,
    memory: c_int,
    host: Handle,
    device: u64,
    array: Handle,
    reserved: Handle,
    pitch: usize,
    height: usize,
}
#[repr(C)]
#[derive(Default)]
struct CopyParams {
    source: CopyEndpoint,
    destination: CopyEndpoint,
    width: usize,
    height: usize,
    depth: usize,
}

fn kernel_params(cuda: &Cuda, node: Handle) -> ExportResult<(KernelParams, usize)> {
    let mut params = KernelParams::default();
    cu!(
        cuda,
        "cuGraphKernelNodeGetParams_v2",
        (Handle, *mut KernelParams),
        node,
        &mut params
    )?;
    let mut function = params.function;
    if function.is_null() {
        cu!(
            cuda,
            "cuKernelGetFunction",
            (*mut Handle, Handle),
            &mut function,
            params.kernel
        )?;
    }
    if function.is_null() {
        return Err(invalid("kernel node has no CUDA function"));
    }
    Ok((params, function as usize))
}

fn nodes(cuda: &Cuda, graph: usize) -> ExportResult<Vec<Handle>> {
    let mut count = 0usize;
    cu!(
        cuda,
        "cuGraphGetNodes",
        (Handle, *mut Handle, *mut usize),
        graph as Handle,
        std::ptr::null_mut(),
        &mut count
    )?;
    let mut nodes = vec![std::ptr::null_mut(); count];
    cu!(
        cuda,
        "cuGraphGetNodes",
        (Handle, *mut Handle, *mut usize),
        graph as Handle,
        nodes.as_mut_ptr(),
        &mut count
    )?;
    nodes.truncate(count);
    Ok(nodes)
}

fn node_kind(cuda: &Cuda, node: Handle) -> ExportResult<c_int> {
    let mut kind = 0;
    cu!(
        cuda,
        "cuGraphNodeGetType",
        (Handle, *mut c_int),
        node,
        &mut kind
    )?;
    Ok(kind)
}

fn reference(
    allocations: &[Allocation],
    address: u64,
    nbytes: u64,
) -> ExportResult<wire::Reference> {
    let end = address
        .checked_add(nbytes)
        .ok_or_else(|| invalid("CUDA span address overflow"))?;
    let mut matches = allocations
        .iter()
        .filter(|a| a.address <= address && end <= a.address + a.nbytes);
    let allocation = matches
        .next()
        .ok_or_else(|| invalid(format!("unregistered CUDA span {address:#x}+{nbytes}")))?;
    if matches.next().is_some() {
        return Err(invalid("CUDA span belongs to overlapping allocations"));
    }
    Ok(wire::Reference {
        allocation: allocation.name.clone(),
        offset: address - allocation.address,
    })
}

fn address(allocations: &[Allocation], reference: &super::Reference) -> ExportResult<u64> {
    let allocation = allocations
        .iter()
        .find(|a| a.name == reference.allocation)
        .ok_or_else(|| {
            invalid(format!(
                "unregistered graph allocation: {}",
                reference.allocation
            ))
        })?;
    if reference.offset >= allocation.nbytes {
        return Err(invalid("allocation reference offset is out of bounds"));
    }
    Ok(allocation.address + reference.offset)
}
fn relocation(value: &Relocation) -> ExportResult<wire::Relocation> {
    let kind = match value.kind.as_str() {
        "pointer" => 0,
        "tensor_map" => 1,
        _ => return Err(invalid("unsupported relocation kind")),
    };
    Ok(wire::Relocation {
        offset: value.offset,
        target: Some(wire::Reference {
            allocation: value.target.allocation.clone(),
            offset: value.target.offset,
        }),
        kind,
    })
}
fn check_relocations(
    allocations: &[Allocation],
    size: u64,
    relocations: &[Relocation],
) -> ExportResult<()> {
    let mut relocations: Vec<_> = relocations.iter().collect();
    relocations.sort_by_key(|r| r.offset);
    let mut end = 0;
    for binding in relocations {
        let width = match binding.kind.as_str() {
            "pointer" => 8,
            "tensor_map" => 128,
            _ => return Err(invalid("unsupported relocation kind")),
        };
        let stop = binding
            .offset
            .checked_add(width)
            .ok_or_else(|| invalid("relocation extent overflow"))?;
        if binding.offset < end || stop > size {
            return Err(invalid("invalid or overlapping relocation fields"));
        }
        address(allocations, &binding.target)?;
        end = stop;
    }
    Ok(())
}
fn check_bindings(
    cuda: &Cuda,
    allocations: &[Allocation],
    data: &[u8],
    relocations: &[Relocation],
) -> ExportResult<()> {
    for binding in relocations {
        let target = address(allocations, &binding.target)?;
        let offset = binding.offset as usize;
        if binding.kind == "pointer" {
            if data.get(offset..offset + 8) != Some(target.to_le_bytes().as_slice()) {
                return Err(invalid(format!(
                    "pointer binding does not match {}",
                    binding.target.allocation
                )));
            }
        } else {
            #[repr(C, align(64))]
            struct TensorMap([u8; 128]);
            let original = data
                .get(offset..offset + 128)
                .ok_or_else(|| invalid("tensor-map binding outside argument"))?;
            let mut map = TensorMap(original.try_into().unwrap());
            cu!(
                cuda,
                "cuTensorMapReplaceAddress",
                (*mut c_void, *mut c_void),
                map.0.as_mut_ptr().cast(),
                target as usize as Handle
            )?;
            if map.0.as_slice() != original {
                return Err(invalid(format!(
                    "tensor-map binding does not match {}",
                    binding.target.allocation
                )));
            }
        }
    }
    Ok(())
}

unsafe fn argument_data(
    params: &KernelParams,
    layout: &[(usize, usize)],
) -> ExportResult<Vec<Vec<u8>>> {
    if !params.params.is_null() {
        if !params.extra.is_null() {
            return Err(invalid(
                "kernel specifies both parameter-array and packed arguments",
            ));
        }
        return layout
            .iter()
            .enumerate()
            .map(|(index, &(_, size))| {
                let pointer = unsafe { *params.params.add(index) };
                if pointer.is_null() {
                    return Err(invalid("kernel parameter array contains a null argument"));
                }
                Ok(unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), size) }.to_vec())
            })
            .collect();
    }
    if params.extra.is_null() {
        return if layout.is_empty() {
            Ok(vec![])
        } else {
            Err(invalid("kernel has parameters but no launch arguments"))
        };
    }
    let mut buffer: *const u8 = std::ptr::null();
    let mut size_pointer: *const usize = std::ptr::null();
    let mut terminated = false;
    let mut seen = [false; 3];
    for index in (0..5).step_by(2) {
        let tag = unsafe { *params.extra.add(index) } as usize;
        if tag == 0 {
            terminated = true;
            break;
        }
        if index == 4 || tag > 2 || seen[tag] {
            return Err(invalid("invalid packed kernel argument options"));
        }
        seen[tag] = true;
        let value = unsafe { *params.extra.add(index + 1) };
        match tag {
            1 => buffer = value as *const u8,
            2 => size_pointer = value as *const usize,
            _ => return Err(invalid("invalid packed kernel argument options")),
        }
    }
    if !terminated {
        return Err(invalid("packed kernel argument options lack a terminator"));
    }
    if size_pointer.is_null() {
        return Err(invalid("packed kernel arguments lack a buffer size"));
    }
    let nbytes = unsafe { *size_pointer };
    if nbytes > 0 && buffer.is_null() {
        return Err(invalid("packed kernel arguments lack a buffer"));
    }
    layout
        .iter()
        .map(|&(offset, size)| {
            if offset.checked_add(size).is_none_or(|end| end > nbytes) {
                return Err(invalid(
                    "kernel parameter extends beyond its packed argument buffer",
                ));
            }
            if size == 0 {
                return Ok(vec![]);
            }
            Ok(unsafe { std::slice::from_raw_parts(buffer.add(offset), size) }.to_vec())
        })
        .collect()
}

fn function_layout(cuda: &Cuda, function: usize) -> ExportResult<Vec<(usize, usize)>> {
    let mut layout = Vec::new();
    for index in 0..4096usize {
        let (mut offset, mut size) = (0usize, 0usize);
        let status = status!(
            cuda,
            "cuFuncGetParamInfo",
            (Handle, usize, *mut usize, *mut usize),
            function as Handle,
            index,
            &mut offset,
            &mut size
        );
        if status == 1 {
            return Ok(layout);
        }
        cuda.check("cuFuncGetParamInfo", status)?;
        if size == 0 {
            return Err(invalid("kernel arguments must have positive sizes"));
        }
        layout.push((offset, size));
    }
    Err(invalid("kernel exceeds supported parameter count"))
}

fn kernel_node(
    cuda: &Cuda,
    node: Handle,
    allocations: &[Allocation],
    kernels: &[Kernel],
    layouts: &[Vec<(usize, usize)>],
    indices: &HashMap<usize, usize>,
) -> ExportResult<wire::Kernel> {
    let (params, function) = kernel_params(cuda, node)?;
    let index = *indices
        .get(&function)
        .ok_or_else(|| invalid("no explicit argument bindings for kernel"))?;
    let kernel = &kernels[index];
    let arguments = unsafe { argument_data(&params, &layouts[index])? }
        .into_iter()
        .zip(&kernel.arguments)
        .map(|(data, argument)| {
            check_bindings(cuda, allocations, &data, &argument.relocations)?;
            Ok(wire::Blob {
                data,
                relocations: argument
                    .relocations
                    .iter()
                    .map(relocation)
                    .collect::<ExportResult<_>>()?,
            })
        })
        .collect::<ExportResult<Vec<_>>>()?;
    let mut attributes = Vec::new();
    for id in [1, 2, 4, 5, 8, 9, 10, 11, 13, 14] {
        let mut value = [0u64; 8];
        let status = status!(
            cuda,
            "cuGraphKernelNodeGetAttribute",
            (Handle, c_int, *mut c_void),
            node,
            id,
            value.as_mut_ptr().cast()
        );
        if id == 11 && status == 1 {
            continue;
        }
        cuda.check(&format!("cuGraphKernelNodeGetAttribute({id})"), status)?;
        let any = value.iter().any(|&v| v != 0);
        if id == 13 {
            if any {
                return Err(invalid(
                    "device-updatable kernel nodes require explicit export support",
                ));
            }
            continue;
        }
        if id == 14 && !any {
            continue;
        }
        let relocations = if id == 1 && value[1] != 0 {
            vec![wire::Relocation {
                offset: 0,
                target: Some(reference(allocations, value[0], value[1])?),
                kind: 0,
            }]
        } else {
            vec![]
        };
        attributes.push(wire::Attribute {
            id,
            value: Some(wire::Blob {
                data: value.into_iter().flat_map(u64::to_ne_bytes).collect(),
                relocations,
            }),
        });
    }
    Ok(wire::Kernel {
        function: index as u64,
        grid: params.grid.to_vec(),
        block: params.block.to_vec(),
        shared: params.shared,
        arguments,
        attributes,
    })
}

fn capture_graph(
    cuda: &Cuda,
    graph: &super::GraphHandle,
    recorder: Option<&Recorder>,
    cupti: Option<&Library>,
    allocations: &[Allocation],
    kernels: &[Kernel],
    layouts: &[Vec<(usize, usize)>],
    indices: &HashMap<usize, usize>,
    events: &BTreeMap<String, usize>,
) -> ExportResult<(Vec<wire::Node>, Vec<wire::Edge>, u64)> {
    let mut ordered = nodes(cuda, graph.graph)?
        .into_iter()
        .map(|node| {
            let id = if let Some(recorder) = recorder {
                recorder.graph_node_id(node as usize)?
            } else {
                let library = cupti.ok_or_else(|| invalid("graph node ordering requires CUPTI"))?;
                let function = unsafe {
                    library.get::<unsafe extern "C" fn(Handle, *mut u64) -> c_int>(
                        b"cuptiGetGraphNodeId\0",
                    )?
                };
                let mut id = 0u64;
                let status = unsafe { function(node, &mut id) };
                if status != 0 {
                    return Err(invalid(format!("cuptiGetGraphNodeId: CUPTI {status}")));
                }
                id
            };
            Ok((id, node))
        })
        .collect::<ExportResult<Vec<_>>>()?;
    ordered.sort_by_key(|&(id, _)| id);
    let node_indices: HashMap<_, _> = ordered
        .iter()
        .enumerate()
        .map(|(index, &(_, node))| (node as usize, index as u64))
        .collect();
    let mut result = Vec::new();
    for (_, node) in ordered {
        use wire::node::Operation;
        let operation = match node_kind(cuda, node)? {
            0 => Operation::Kernel(kernel_node(
                cuda,
                node,
                allocations,
                kernels,
                layouts,
                indices,
            )?),
            1 => {
                let mut params = CopyParams::default();
                cu!(
                    cuda,
                    "cuGraphMemcpyNodeGetParams",
                    (Handle, *mut CopyParams),
                    node,
                    &mut params
                )?;
                if params.height != 1 || params.depth != 1 {
                    return Err(invalid(
                        "only contiguous device-to-device graph copies are supported",
                    ));
                }
                for endpoint in [&params.source, &params.destination] {
                    if ![2, 4].contains(&endpoint.memory)
                        || endpoint.y != 0
                        || endpoint.z != 0
                        || endpoint.lod != 0
                        || (endpoint.memory == 4 && !cuda.is_device(endpoint.device)?)
                    {
                        return Err(invalid(
                            "only contiguous device-to-device graph copies are supported",
                        ));
                    }
                }
                Operation::Copy(wire::Copy {
                    source: Some(reference(
                        allocations,
                        params
                            .source
                            .device
                            .checked_add(params.source.x as u64)
                            .ok_or_else(|| invalid("copy source address overflow"))?,
                        params.width as u64,
                    )?),
                    destination: Some(reference(
                        allocations,
                        params
                            .destination
                            .device
                            .checked_add(params.destination.x as u64)
                            .ok_or_else(|| invalid("copy destination address overflow"))?,
                        params.width as u64,
                    )?),
                    nbytes: params.width as u64,
                })
            }
            2 => {
                let mut params = MemsetParams::default();
                cu!(
                    cuda,
                    "cuGraphMemsetNodeGetParams",
                    (Handle, *mut MemsetParams),
                    node,
                    &mut params
                )?;
                let extent = params
                    .height
                    .checked_sub(1)
                    .and_then(|h| h.checked_mul(params.pitch))
                    .and_then(|n| {
                        params
                            .width
                            .checked_mul(params.element_size as usize)
                            .and_then(|w| n.checked_add(w))
                    })
                    .ok_or_else(|| invalid("invalid graph memset extent"))?;
                Operation::Memset(wire::Memset {
                    destination: Some(reference(allocations, params.destination, extent as u64)?),
                    pitch: params.pitch as u64,
                    value: params.value,
                    element_size: params.element_size,
                    width: params.width as u64,
                    height: params.height as u64,
                })
            }
            5 => Operation::Empty(wire::Empty {}),
            6 => {
                let mut event = std::ptr::null_mut();
                cu!(
                    cuda,
                    "cuGraphEventWaitNodeGetEvent",
                    (Handle, *mut Handle),
                    node,
                    &mut event
                )?;
                let name = events
                    .iter()
                    .find(|(_, value)| **value == event as usize)
                    .map(|(name, _)| name.clone())
                    .ok_or_else(|| {
                        invalid(format!(
                            "graph {}: undeclared external readiness event",
                            graph.name
                        ))
                    })?;
                Operation::EventWait(wire::EventWait { event: name })
            }
            kind => {
                return Err(invalid(format!(
                    "graph {}: unsupported CUDA graph node type {kind}",
                    graph.name
                )));
            }
        };
        result.push(wire::Node {
            operation: Some(operation),
        });
    }
    let mut count = 0usize;
    cu!(
        cuda,
        "cuGraphGetEdges_v2",
        (Handle, *mut Handle, *mut Handle, *mut [u8; 8], *mut usize),
        graph.graph as Handle,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        &mut count
    )?;
    let mut sources = vec![std::ptr::null_mut(); count];
    let mut destinations = sources.clone();
    let mut data = vec![[0u8; 8]; count];
    if count > 0 {
        cu!(
            cuda,
            "cuGraphGetEdges_v2",
            (Handle, *mut Handle, *mut Handle, *mut [u8; 8], *mut usize),
            graph.graph as Handle,
            sources.as_mut_ptr(),
            destinations.as_mut_ptr(),
            data.as_mut_ptr(),
            &mut count
        )?;
    }
    let edges = sources
        .into_iter()
        .zip(destinations)
        .zip(data)
        .take(count)
        .map(|((source, destination), data)| {
            Ok(wire::Edge {
                source: *node_indices
                    .get(&(source as usize))
                    .ok_or_else(|| invalid("graph edge has unregistered source"))?,
                destination: *node_indices
                    .get(&(destination as usize))
                    .ok_or_else(|| invalid("graph edge has unregistered destination"))?,
                data: data.to_vec(),
            })
        })
        .collect::<ExportResult<Vec<_>>>()?;
    let mut flags = 0u64;
    cu!(
        cuda,
        "cuGraphExecGetFlags",
        (Handle, *mut u64),
        graph.executable as Handle,
        &mut flags
    )?;
    Ok((result, edges, flags))
}

struct DownloadSlot {
    pointer: Handle,
    event: Handle,
    offset: u64,
    nbytes: usize,
}
struct Downloader<'a> {
    cuda: &'a Cuda,
    stream: Handle,
    slots: Vec<DownloadSlot>,
    drained: bool,
    downloaded: u64,
}
impl<'a> Downloader<'a> {
    const CHUNK: usize = 8 << 20;
    fn new(cuda: &'a Cuda) -> ExportResult<Self> {
        let mut this = Self {
            cuda,
            stream: std::ptr::null_mut(),
            slots: Vec::new(),
            drained: true,
            downloaded: 0,
        };
        cu!(
            cuda,
            "cuStreamCreate",
            (*mut Handle, c_uint),
            &mut this.stream,
            1
        )?;
        for _ in 0..2 {
            let mut pointer = std::ptr::null_mut();
            cu!(
                cuda,
                "cuMemHostAlloc",
                (*mut Handle, usize, c_uint),
                &mut pointer,
                Self::CHUNK,
                0
            )?;
            this.slots.push(DownloadSlot {
                pointer,
                event: std::ptr::null_mut(),
                offset: 0,
                nbytes: 0,
            });
            let slot = this.slots.last_mut().unwrap();
            cu!(
                cuda,
                "cuEventCreate",
                (*mut Handle, c_uint),
                &mut slot.event,
                2
            )?;
        }
        Ok(this)
    }
    fn drain(&mut self) -> ExportResult<()> {
        if self.stream.is_null() {
            return Ok(());
        }
        cu!(self.cuda, "cuStreamSynchronize", (Handle), self.stream)?;
        self.drained = true;
        Ok(())
    }
    fn read(&mut self, address: u64, memory: &str, output: &mut [u8]) -> ExportResult<()> {
        if memory == "mapped_host" {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    address as usize as *const u8,
                    output.as_mut_ptr(),
                    output.len(),
                );
            }
            return Ok(());
        }
        for (index, chunk) in output.chunks_mut(Self::CHUNK).enumerate() {
            self.drained = false;
            self.downloaded += chunk.len() as u64;
            let pointer = self.slots[0].pointer;
            cu!(
                self.cuda,
                "cuMemcpyDtoHAsync_v2",
                (Handle, u64, usize, Handle),
                pointer,
                address + (index * Self::CHUNK) as u64,
                chunk.len(),
                self.stream
            )?;
            self.drain()?;
            chunk.copy_from_slice(unsafe {
                std::slice::from_raw_parts(pointer.cast::<u8>(), chunk.len())
            });
        }
        Ok(())
    }
    fn for_each(
        &mut self,
        allocation: &Allocation,
        spans: &[(u64, u64)],
        mut visit: impl FnMut(u64, &[u8]) -> ExportResult<()>,
    ) -> ExportResult<()> {
        let chunks = spans.iter().flat_map(|&(start, size)| {
            (0..size).step_by(Self::CHUNK).map(move |relative| {
                (
                    start + relative,
                    (size - relative).min(Self::CHUNK as u64) as usize,
                )
            })
        });
        if allocation.memory == "mapped_host" {
            for (offset, size) in chunks {
                visit(offset, unsafe {
                    std::slice::from_raw_parts(
                        (allocation.address + offset) as usize as *const u8,
                        size,
                    )
                })?;
            }
            return Ok(());
        }
        let mut pending = std::collections::VecDeque::new();
        for (offset, size) in chunks {
            if pending.len() == self.slots.len() {
                let index = pending.pop_front().unwrap();
                self.consume(index, &mut visit)?;
            }
            let index = if pending.is_empty() {
                0
            } else {
                1 - pending[0]
            };
            let slot = &mut self.slots[index];
            slot.offset = offset;
            slot.nbytes = size;
            self.drained = false;
            self.downloaded += size as u64;
            cu!(
                self.cuda,
                "cuMemcpyDtoHAsync_v2",
                (Handle, u64, usize, Handle),
                slot.pointer,
                allocation.address + offset,
                size,
                self.stream
            )?;
            cu!(
                self.cuda,
                "cuEventRecord",
                (Handle, Handle),
                slot.event,
                self.stream
            )?;
            pending.push_back(index);
        }
        while let Some(index) = pending.pop_front() {
            self.consume(index, &mut visit)?;
        }
        self.drain()
    }
    fn consume(
        &mut self,
        index: usize,
        visit: &mut impl FnMut(u64, &[u8]) -> ExportResult<()>,
    ) -> ExportResult<()> {
        let slot = &self.slots[index];
        cu!(self.cuda, "cuEventSynchronize", (Handle), slot.event)?;
        visit(slot.offset, unsafe {
            std::slice::from_raw_parts(slot.pointer.cast::<u8>(), slot.nbytes)
        })
    }
}
impl Drop for Downloader<'_> {
    fn drop(&mut self) {
        if !self.drained && self.drain().is_err() {
            return;
        }
        for slot in &self.slots {
            if !slot.event.is_null() {
                let _ = (|| -> ExportResult<()> {
                    cu!(self.cuda, "cuEventDestroy_v2", (Handle), slot.event)
                })();
            }
            let _ = (|| -> ExportResult<()> {
                cu!(self.cuda, "cuMemFreeHost", (Handle), slot.pointer)
            })();
        }
        if !self.stream.is_null() {
            let _ = (|| -> ExportResult<()> {
                cu!(self.cuda, "cuStreamDestroy_v2", (Handle), self.stream)
            })();
        }
    }
}

fn live_pieces(allocation: &Allocation, inactive: &[(u64, u64)]) -> Vec<Allocation> {
    let (begin, end, granule) = (
        allocation.address,
        allocation.address + allocation.nbytes,
        allocation.alignment,
    );
    let mut cursor = begin;
    let mut pieces = Vec::new();
    for &(start, stop) in inactive {
        if stop <= begin {
            continue;
        }
        if start >= end {
            break;
        }
        let start = start.max(begin).div_ceil(granule) * granule;
        let stop = stop.min(end) / granule * granule;
        if start < stop {
            if cursor < start {
                pieces.push((cursor, start));
            }
            cursor = cursor.max(stop);
        }
    }
    if cursor < end {
        pieces.push((cursor, end));
    }
    if pieces.is_empty() || pieces == [(begin, end)] {
        return vec![allocation.clone()];
    }
    let count = pieces.len();
    pieces
        .into_iter()
        .enumerate()
        .map(|(index, (start, stop))| {
            let mut piece = allocation.clone();
            if count > 1 {
                piece.name = format!("{}.{}", allocation.name, index);
            }
            piece.address = start;
            piece.nbytes = stop - start;
            piece
        })
        .collect()
}

fn without_zero_ranges(allocation: &mut Allocation, zero: &[(u64, u64)]) {
    if allocation.initialize == "zero" {
        return;
    }
    let mut kept = Vec::new();
    for (offset, size) in allocation
        .snapshot_spans
        .clone()
        .unwrap_or_else(|| vec![(0, allocation.nbytes)])
    {
        for (start, end) in subtract_ranges(
            allocation.address + offset,
            allocation.address + offset + size,
            zero,
        ) {
            kept.push((start - allocation.address, end - start));
        }
    }
    if kept.is_empty() {
        allocation.initialize = "zero".into();
        allocation.snapshot_spans = None;
    } else {
        allocation.snapshot_spans = Some(kept);
    }
}

fn external_allocations(mut spans: Vec<(u64, u64)>, granule: u64) -> ExportResult<Vec<Allocation>> {
    if !granule.is_power_of_two() {
        return Err(invalid("allocation granularity must be a power of two"));
    }
    spans.sort_unstable();
    let mut groups: Vec<(u64, u64, Vec<(u64, u64)>)> = Vec::new();
    for (address, nbytes) in spans {
        if address == 0 || nbytes == 0 {
            return Err(invalid(
                "external allocation requires nonnull address and positive size",
            ));
        }
        let start = address & !(granule - 1);
        let end = address
            .checked_add(nbytes)
            .and_then(|n| n.checked_add(granule - 1))
            .map(|n| n & !(granule - 1))
            .ok_or_else(|| invalid("external allocation address overflow"))?;
        if let Some(group) = groups.last_mut().filter(|group| start <= group.1) {
            group.1 = group.1.max(end);
            group.2.push((address, nbytes));
        } else {
            groups.push((start, end, vec![(address, nbytes)]));
        }
    }
    groups
        .into_iter()
        .enumerate()
        .map(|(index, (start, end, spans))| {
            let spans = merge_ranges(
                spans
                    .into_iter()
                    .map(|(address, size)| (address, address + size))
                    .collect(),
            );
            Ok(Allocation {
                name: format!("external-{index}"),
                address: start,
                nbytes: end - start,
                alignment: granule,
                initialize: "snapshot".into(),
                relocations: vec![],
                preserve_address: true,
                memory: "device".into(),
                snapshot_spans: Some(
                    spans
                        .into_iter()
                        .map(|(address, stop)| (address - start, stop - address))
                        .collect(),
                ),
                sizing: None,
            })
        })
        .collect()
}

fn storage_ranges(ranges: &[(u64, u64)]) -> ExportResult<Vec<(u64, u64)>> {
    let intervals = ranges
        .iter()
        .map(|&(address, size)| {
            if address == 0 || size == 0 {
                return Err(invalid(
                    "storage ranges require a nonnull address and positive size",
                ));
            }
            address
                .checked_add(size)
                .map(|end| (address, end))
                .ok_or_else(|| invalid("storage range address overflow"))
        })
        .collect::<ExportResult<_>>()?;
    Ok(merge_ranges(intervals))
}

fn prepare_allocations(
    cuda: &Cuda,
    request: &CaptureRequest,
    recorder: Option<&Recorder>,
) -> ExportResult<(Vec<Allocation>, Vec<(u64, u64)>)> {
    let mut allocations = request.allocations.clone();
    let mut external = Vec::new();
    if recorder.is_none() && (request.discover_allocations || !request.peer_mappings.is_empty()) {
        return Err(invalid(
            "allocation discovery and peer mapping validation require a native module recorder",
        ));
    }
    if let Some(recorder) =
        recorder.filter(|_| request.discover_allocations || !request.peer_mappings.is_empty())
    {
        let recorded_allocations = recorder.recorded_allocations()?;
        for &(address, nbytes) in &request.peer_mappings {
            if !recorded_allocations
                .iter()
                .any(|a| a.address == address && a.nbytes == nbytes && a.memory == "device")
            {
                return Err(invalid(format!(
                    "peer mapping {address:#x}+{nbytes} differs from the recorded allocation"
                )));
            }
        }
        for recorded in recorded_allocations
            .into_iter()
            .filter(|_| request.discover_allocations)
        {
            if recorded.memory != "device" {
                continue;
            }
            let end = recorded
                .address
                .checked_add(recorded.nbytes)
                .ok_or_else(|| invalid("recorded allocation address overflow"))?;
            if allocations
                .iter()
                .any(|a| a.address <= recorded.address && end <= a.address + a.nbytes)
            {
                continue;
            }
            let mut context = std::ptr::null_mut();
            let mut memory = 0i32;
            cu!(
                cuda,
                "cuPointerGetAttribute",
                (*mut c_void, c_int, u64),
                (&mut context as *mut Handle).cast(),
                1,
                recorded.address
            )?;
            cu!(
                cuda,
                "cuPointerGetAttribute",
                (*mut c_void, c_int, u64),
                (&mut memory as *mut i32).cast(),
                2,
                recorded.address
            )?;
            if memory != 2 || context as usize != request.context {
                return Err(invalid(format!(
                    "external allocation {:#x}+{} is not device memory in the serving context (memory {memory}, context {:?}, recorded context {:?}, serving context {:#x})",
                    recorded.address, recorded.nbytes, context, recorded.context, request.context
                )));
            }
            external.push((recorded.address, recorded.nbytes));
        }
    }
    allocations.extend(external_allocations(external.clone(), request.granularity)?);
    let inactive = merge_ranges(request.inactive.clone());
    let peer: HashSet<_> = request.peer_addresses.iter().copied().collect();
    allocations = allocations
        .into_iter()
        .flat_map(|a| {
            if a.memory == "device"
                && a.sizing.is_none()
                && !peer.contains(&a.address)
                && a.snapshot_spans.is_none()
            {
                live_pieces(&a, &inactive)
            } else {
                vec![a]
            }
        })
        .collect();
    let zero = storage_ranges(&request.zero_ranges)?;
    for allocation in &mut allocations {
        without_zero_ranges(allocation, &zero);
    }
    Ok((allocations, external))
}

fn validate_allocations(allocations: &[Allocation]) -> ExportResult<()> {
    if allocations.is_empty() {
        return Err(invalid("runtime package requires allocations and kernels"));
    }
    let mut names = HashSet::new();
    let mut intervals = Vec::new();
    for a in allocations {
        if a.name.is_empty() || a.address == 0 || a.nbytes == 0 || !names.insert(&a.name) {
            return Err(invalid(
                "allocations require a unique name, nonnull address, and positive size",
            ));
        }
        let end = a
            .address
            .checked_add(a.nbytes)
            .ok_or_else(|| invalid("allocation address overflow"))?;
        if !a.alignment.is_power_of_two() || a.address % a.alignment != 0 {
            return Err(invalid(
                "allocation alignment must be a satisfied power of two",
            ));
        }
        if !["snapshot", "zero"].contains(&a.initialize.as_str()) {
            return Err(invalid("allocation initializer must be snapshot or zero"));
        }
        if !["device", "mapped_host", "peer"].contains(&a.memory.as_str()) {
            return Err(invalid("unsupported allocation memory type"));
        }
        if a.memory == "peer"
            && (!a.preserve_address || a.initialize != "zero" || !a.relocations.is_empty())
        {
            return Err(invalid(
                "peer allocations name another rank's storage at its recorded address",
            ));
        }
        if a.memory == "mapped_host" && !a.preserve_address {
            return Err(invalid(
                "mapped host allocations require a recorded address",
            ));
        }
        if let Some(spans) = &a.snapshot_spans {
            if a.initialize != "snapshot" {
                return Err(invalid("snapshot spans require a snapshot initializer"));
            }
            let mut cursor = 0;
            for &(offset, size) in spans {
                let stop = offset
                    .checked_add(size)
                    .ok_or_else(|| invalid("snapshot span overflow"))?;
                if offset < cursor || size == 0 || stop > a.nbytes {
                    return Err(invalid(
                        "snapshot spans must be ordered, disjoint, and within the allocation",
                    ));
                }
                cursor = stop;
            }
        }
        check_relocations(allocations, a.nbytes, &a.relocations)?;
        intervals.push((a.address, end));
    }
    intervals.sort_unstable();
    if intervals.windows(2).any(|w| w[0].1 > w[1].0) {
        return Err(invalid(
            "register shared backing storage once, with offset references for views",
        ));
    }
    Ok(())
}

struct PackedDownload<'a, 'b> {
    downloader: &'a mut Downloader<'b>,
    allocation: &'a Allocation,
    spans: &'a [(u64, u64)],
    size: u64,
    pending: std::collections::VecDeque<usize>,
    next: u64,
}
impl PackedDownload<'_, '_> {
    fn queue(&mut self, index: usize, offset: u64) -> ExportResult<()> {
        let count = (self.size - offset).min(Downloader::CHUNK as u64) as usize;
        let slot = &mut self.downloader.slots[index];
        slot.offset = offset;
        slot.nbytes = count;
        let mut logical = 0;
        let mut copied = 0usize;
        for &(start, length) in self.spans {
            if logical + length <= offset {
                logical += length;
                continue;
            }
            let relative = offset.saturating_sub(logical);
            let bytes = (length - relative).min((count - copied) as u64) as usize;
            self.downloader.drained = false;
            self.downloader.downloaded += bytes as u64;
            cu!(
                self.downloader.cuda,
                "cuMemcpyDtoHAsync_v2",
                (Handle, u64, usize, Handle),
                slot.pointer.cast::<u8>().wrapping_add(copied).cast(),
                self.allocation.address + start + relative,
                bytes,
                self.downloader.stream
            )?;
            copied += bytes;
            logical += length;
            if copied == count {
                break;
            }
        }
        if copied != count {
            return Err(invalid(
                "packed allocation payload extends beyond stored spans",
            ));
        }
        cu!(
            self.downloader.cuda,
            "cuEventRecord",
            (Handle, Handle),
            slot.event,
            self.downloader.stream
        )?;
        self.pending.push_back(index);
        self.next = offset + count as u64;
        Ok(())
    }
    fn fill(&mut self, offset: u64, output: &mut [u8]) -> ExportResult<()> {
        if self.allocation.memory == "mapped_host" {
            let mut logical = 0;
            let mut copied = 0;
            for &(start, length) in self.spans {
                if logical + length <= offset {
                    logical += length;
                    continue;
                }
                let relative = offset.saturating_sub(logical);
                let count = (length - relative).min((output.len() - copied) as u64) as usize;
                self.downloader.read(
                    self.allocation.address + start + relative,
                    "mapped_host",
                    &mut output[copied..copied + count],
                )?;
                copied += count;
                logical += length;
                if copied == output.len() {
                    return Ok(());
                }
            }
            return Err(invalid(
                "packed host allocation payload extends beyond stored spans",
            ));
        }
        if offset == 0 {
            self.downloader.drain()?;
            self.pending.clear();
            self.next = 0;
            self.queue(0, 0)?;
            if self.next < self.size {
                self.queue(1, self.next)?;
            }
        }
        let index = self
            .pending
            .pop_front()
            .ok_or_else(|| invalid("nonsequential allocation payload download"))?;
        let slot = &self.downloader.slots[index];
        if slot.offset != offset || slot.nbytes != output.len() {
            return Err(invalid("nonsequential allocation payload download"));
        }
        cu!(
            self.downloader.cuda,
            "cuEventSynchronize",
            (Handle),
            slot.event
        )?;
        output.copy_from_slice(unsafe {
            std::slice::from_raw_parts(slot.pointer.cast::<u8>(), slot.nbytes)
        });
        if self.next < self.size {
            self.queue(index, self.next)?;
        }
        Ok(())
    }
}

struct AllocationCommit {
    allocation: Allocation,
    spans: Vec<(u64, u64)>,
    prepared: super::blob::PreparedBlob,
}

fn commit_allocations(
    context: usize,
    store: &super::blob::BlobStore,
    jobs: std::sync::mpsc::Receiver<AllocationCommit>,
) -> ExportResult<u64> {
    let cuda = Cuda::new()?;
    let _context = ContextGuard::enter(&cuda, context)?;
    let mut downloader = Downloader::new(&cuda)?;
    let result = (|| -> ExportResult<()> {
        for job in jobs {
            let size = job.spans.iter().map(|&(_, n)| n).sum();
            let mut reader = PackedDownload {
                downloader: &mut downloader,
                allocation: &job.allocation,
                spans: &job.spans,
                size,
                pending: std::collections::VecDeque::new(),
                next: 0,
            };
            store.commit(job.prepared, |offset, output| reader.fill(offset, output))?;
            reader.downloader.drain()?;
        }
        Ok(())
    })();
    let drained = downloader.drain().and_then(|_| cuda.synchronize());
    result?;
    drained?;
    Ok(downloader.downloaded)
}

fn with_allocation_commits<T>(
    context: usize,
    store: Option<&super::blob::BlobStore>,
    produce: impl FnOnce(Option<&std::sync::mpsc::SyncSender<AllocationCommit>>) -> ExportResult<T>,
) -> ExportResult<(T, u64)> {
    std::thread::scope(|scope| {
        let (commits, worker) = if let Some(store) = store {
            let (sender, receiver) = std::sync::mpsc::sync_channel(0);
            (
                Some(sender),
                Some(scope.spawn(move || commit_allocations(context, store, receiver))),
            )
        } else {
            (None, None)
        };
        let produced = produce(commits.as_ref());
        drop(commits);
        let committed = match worker {
            Some(worker) => worker
                .join()
                .map_err(|_| invalid("allocation commit worker panicked"))?,
            None => Ok(0),
        };
        let downloaded = committed?;
        Ok((produced?, downloaded))
    })
}

fn write_allocation(
    downloader: &mut Downloader<'_>,
    allocation: &Allocation,
    path: &Path,
    reach: &mut Reachability,
    store: Option<&super::blob::BlobStore>,
    commits: Option<&std::sync::mpsc::SyncSender<AllocationCommit>>,
    required: bool,
    phase: &helios_core::startup::Phase,
    scanned: &mut u64,
) -> ExportResult<(Option<FileSource>, Vec<(u64, u64)>)> {
    if allocation.initialize == "zero" {
        return Ok((None, vec![]));
    }
    let mut stored: Vec<(u64, u64)> = Vec::new();
    let mut file = if store.is_none() {
        Some(BufWriter::new(File::create(path)?))
    } else {
        None
    };
    let spans = allocation
        .snapshot_spans
        .clone()
        .unwrap_or_else(|| vec![(0, allocation.nbytes)]);
    let mut written_since_flush = 0usize;
    downloader.for_each(allocation, &spans, |offset, data| {
        reach.contents(allocation, offset, data)?;
        *scanned += data.len() as u64;
        phase.advance_bytes(data.len() as u64);
        for (block, bytes) in data.chunks(64 << 10).enumerate() {
            if !bytes.iter().any(|&b| b != 0) {
                continue;
            }
            let begin = offset + (block * (64 << 10)) as u64;
            if let Some(last) = stored.last_mut().filter(|last| last.0 + last.1 == begin) {
                last.1 += bytes.len() as u64;
            } else {
                stored.push((begin, bytes.len() as u64));
            }
            if let Some(file) = &mut file {
                file.write_all(bytes)?;
                written_since_flush += bytes.len();
            }
        }
        if written_since_flush >= 256 << 20 {
            if let Some(file) = &mut file {
                flush_payload(file)?;
            }
            written_since_flush = 0;
        }
        Ok(())
    })?;
    let size = stored.iter().map(|&(_, n)| n).sum();
    if let Some(file) = &mut file {
        flush_payload(file)?;
    }
    if stored.is_empty() {
        drop(file);
        if path.exists() {
            fs::remove_file(path)?;
        }
        return Ok((None, stored));
    }
    let source = if let Some(store) = store {
        let mut reader = PackedDownload {
            downloader,
            allocation,
            spans: &stored,
            size,
            pending: std::collections::VecDeque::new(),
            next: 0,
        };
        let prepared = store.prepare(size, required, None, |offset, output| {
            reader.fill(offset, output)
        })?;
        reader.downloader.drain()?;
        let name = match prepared {
            super::blob::PreparedBlob::Reused(name) => name,
            prepared => {
                let name = match &prepared {
                    super::blob::PreparedBlob::Write { name, .. } => name.clone(),
                    _ => unreachable!(),
                };
                if let Some(commits) = commits {
                    commits
                        .send(AllocationCommit {
                            allocation: allocation.clone(),
                            spans: stored.clone(),
                            prepared,
                        })
                        .map_err(|_| invalid("allocation commit worker stopped"))?;
                    name
                } else {
                    store.commit(prepared, |offset, output| reader.fill(offset, output))?
                }
            }
        };
        FileSource {
            path: store.root.join(name),
            offset: 4096,
            nbytes: size,
        }
    } else {
        FileSource {
            path: path.to_path_buf(),
            offset: 0,
            nbytes: size,
        }
    };
    Ok((Some(source), stored))
}

fn flush_payload(output: &mut BufWriter<File>) -> ExportResult<()> {
    output.flush()?;
    output.get_ref().sync_data()?;
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        let status = unsafe {
            libc::posix_fadvise(
                output.get_ref().as_raw_fd(),
                0,
                0,
                libc::POSIX_FADV_DONTNEED,
            )
        };
        if status != 0 {
            return Err(std::io::Error::from_raw_os_error(status).into());
        }
    }
    Ok(())
}

fn stage_directory(path: &Path) -> ExportResult<PathBuf> {
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let stem = path
        .file_name()
        .ok_or_else(|| invalid("runtime package path requires a filename"))?
        .to_string_lossy();
    loop {
        let stage = parent.join(format!(
            ".{stem}-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        match fs::create_dir(&stage) {
            Ok(()) => return Ok(stage),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
}
struct Stage {
    path: PathBuf,
    keep: bool,
}
impl Drop for Stage {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}
fn write_record(output: &mut impl Write, message: &impl Message) -> ExportResult<()> {
    let data = message.encode_to_vec();
    if data.len() > 1 << 20 {
        return Err(invalid("CUDA graph record exceeds 1 MiB"));
    }
    output.write_all(&(data.len() as u32).to_le_bytes())?;
    output.write_all(&data)?;
    Ok(())
}
fn append_resource(output: &mut RankOutput, name: String, source: FileSource) -> ExportResult<()> {
    let cursor = output
        .payload
        .iter()
        .map(|part| part.offset + part.source.nbytes)
        .max()
        .unwrap_or(0)
        .checked_add(4095)
        .map(|n| n & !4095)
        .ok_or_else(|| invalid("payload offset overflow"))?;
    output.index.ranks[0].resources.insert(
        name,
        wire::Span {
            offset: cursor,
            nbytes: source.nbytes,
            filename: String::new(),
        },
    );
    output.payload.push(FilePart {
        offset: cursor,
        source,
    });
    Ok(())
}
fn file_source(path: PathBuf) -> ExportResult<FileSource> {
    let nbytes = fs::metadata(&path)?.len();
    Ok(FileSource {
        path,
        offset: 0,
        nbytes,
    })
}

pub(crate) fn capture_rank(
    request: &CaptureRequest,
    recorder: Option<&Recorder>,
) -> ExportResult<RankOutput> {
    if request.world_size == 0 {
        return Err(invalid("export rank count must be positive"));
    }
    if request.path.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            &*request.path.to_string_lossy(),
        )
        .into());
    }
    if request.graphs.is_empty()
        || request
            .graphs
            .iter()
            .any(|g| g.name.is_empty() || g.graph == 0 || g.executable == 0)
    {
        return Err(invalid(
            "named live graph templates and executables are required",
        ));
    }
    if request
        .graphs
        .iter()
        .map(|g| &g.name)
        .collect::<HashSet<_>>()
        .len()
        != request.graphs.len()
    {
        return Err(invalid("graph names must be unique"));
    }
    if request
        .events
        .iter()
        .any(|(name, &handle)| name.is_empty() || handle == 0)
        || request.events.values().collect::<HashSet<_>>().len() != request.events.len()
    {
        return Err(invalid(
            "external events require unique names and live handles",
        ));
    }
    if request.transports.len() > 1 {
        return Err(invalid("a rank exports at most one RoCE transport"));
    }
    if request.sources && request.publisher.is_some() {
        return Err(invalid("a signed package cannot reference staged sources"));
    }
    if request.options.no_weights && request.options.cache.is_none() {
        return Err(invalid(
            "export without weight writes requires a package cache",
        ));
    }
    // Validate serving defaults before reading graph handles or storage.
    super::binding::validate_metadata(request)?;
    let progress = if request.rank == 0 {
        helios_core::startup::StartupProgress::stderr_rows()
    } else {
        helios_core::startup::StartupProgress::silent()
    };
    let clear_phase = progress.phase("Clear disposable storage", None);
    let cuda = Cuda::new()?;
    let _context = ContextGuard::enter(&cuda, request.context)?;
    cuda.synchronize()?;
    if !request.allocations.is_empty() {
        validate_allocations(&request.allocations)?;
    }
    let (allocations, external) = prepare_allocations(&cuda, request, recorder)?;
    validate_allocations(&allocations)?;
    let dynamic: Vec<_> = allocations
        .iter()
        .filter(|a| a.sizing.is_some())
        .map(|a| (a.address, a.address + a.nbytes))
        .collect();
    let clear_ranges: Vec<_> = request
        .zero_ranges
        .iter()
        .copied()
        .filter(|&(address, _)| {
            !dynamic
                .iter()
                .any(|&(start, end)| start <= address && address < end)
        })
        .collect();
    let zero = storage_ranges(&clear_ranges)?;
    for &(start, end) in &zero {
        cu!(
            &cuda,
            "cuMemsetD8_v2",
            (u64, u8, usize),
            start,
            0,
            (end - start) as usize
        )?;
    }
    cuda.synchronize()?;
    clear_phase.finish(format!(
        "{} bytes",
        zero.iter().map(|&(start, end)| end - start).sum::<u64>()
    ));

    let inspect_phase = progress.phase(
        "Inspect CUDA graph functions",
        Some(request.graphs.len() as u64),
    );
    let mut kernels = request.kernels.clone();
    let mut modules: BTreeMap<String, Arc<[u8]>> = request
        .modules
        .iter()
        .map(|(name, path)| Ok((name.clone(), Arc::from(fs::read(path)?))))
        .collect::<ExportResult<_>>()?;
    for kernel in &kernels {
        if !modules.contains_key(&kernel.module) {
            let recorder = recorder.ok_or_else(|| {
                invalid("kernel requires a registered module image or native module recorder")
            })?;
            modules.insert(kernel.module.clone(), recorder.image_for(kernel.function)?);
        }
    }
    let mut module_names = HashMap::new();
    let mut indices: HashMap<_, _> = kernels
        .iter()
        .enumerate()
        .map(|(index, kernel)| (kernel.function, index))
        .collect();
    if indices.len() != kernels.len() {
        return Err(invalid("kernel function registrations must be unique"));
    }
    for graph in &request.graphs {
        for node in nodes(&cuda, graph.graph)? {
            if node_kind(&cuda, node)? != 0 {
                continue;
            }
            let (_, function) = kernel_params(&cuda, node)?;
            if indices.contains_key(&function) {
                continue;
            }
            if !request.kernels.is_empty() {
                return Err(invalid("no explicit argument bindings for kernel"));
            }
            let recorder = recorder.ok_or_else(|| {
                invalid("automatic kernel discovery requires a native module recorder")
            })?;
            let mut module = std::ptr::null_mut();
            cu!(
                &cuda,
                "cuFuncGetModule",
                (*mut Handle, Handle),
                &mut module,
                function as Handle
            )?;
            let count = module_names.len();
            let name = module_names
                .entry(module as usize)
                .or_insert_with(|| format!("module-{count}"))
                .clone();
            if !modules.contains_key(&name) {
                modules.insert(name.clone(), recorder.image_for(function)?);
            }
            let arguments = function_layout(&cuda, function)?
                .iter()
                .map(|&(_, size)| Argument {
                    size,
                    relocations: vec![],
                    numeric: vec![],
                })
                .collect();
            indices.insert(function, kernels.len());
            kernels.push(Kernel {
                function,
                module: name,
                arguments,
            });
        }
        inspect_phase.advance(1);
    }
    if kernels.is_empty() {
        return Err(invalid("runtime package requires allocations and kernels"));
    }
    if modules.values().any(|image| !image.starts_with(b"\x7fELF")) {
        return Err(invalid("module images must be compiled cubins"));
    }
    let mut layouts = Vec::new();
    let mut functions = Vec::new();
    for kernel in &mut kernels {
        if kernel.function == 0 || !modules.contains_key(&kernel.module) {
            return Err(invalid(
                "kernel requires a live function and registered module image",
            ));
        }
        let layout = function_layout(&cuda, kernel.function)?;
        if layout
            .iter()
            .map(|&(_, size)| size)
            .ne(kernel.arguments.iter().map(|a| a.size))
        {
            return Err(invalid(
                "argument layout does not cover exactly the kernel's parameters",
            ));
        }
        if let Some(typed) = request.parameter_layouts.get(&kernel.function) {
            if typed
                .iter()
                .map(|&(size, _)| size)
                .ne(kernel.arguments.iter().map(|a| a.size))
            {
                return Err(invalid(
                    "compiler parameter metadata differs from the CUDA function layout",
                ));
            }
            for (argument, &(_, numeric)) in kernel.arguments.iter_mut().zip(typed) {
                if numeric {
                    argument.numeric = vec![(0, argument.size)];
                }
            }
        }
        for argument in &kernel.arguments {
            check_relocations(&allocations, argument.size as u64, &argument.relocations)?;
            let mut end = 0;
            for &(start, stop) in &argument.numeric {
                if start < end || stop <= start || stop > argument.size {
                    return Err(invalid(
                        "numeric argument fields must be ordered, disjoint, and within the parameter",
                    ));
                }
                if argument.relocations.iter().any(|r| {
                    start < r.offset as usize + if r.kind == "pointer" { 8 } else { 128 }
                        && (r.offset as usize) < stop
                }) {
                    return Err(invalid("numeric argument fields overlap a pointer binding"));
                }
                end = stop;
            }
        }
        let mut name = std::ptr::null();
        cu!(
            &cuda,
            "cuFuncGetName",
            (*mut *const c_char, Handle),
            &mut name,
            kernel.function as Handle
        )?;
        if name.is_null() {
            return Err(invalid("CUDA function has no symbol name"));
        }
        let symbol = unsafe { CStr::from_ptr(name) }.to_str()?.to_owned();
        let mut attributes = Vec::new();
        for id in [8, 9, 14, 15] {
            let mut value = 0;
            cu!(
                &cuda,
                "cuFuncGetAttribute",
                (*mut c_int, c_int, Handle),
                &mut value,
                id,
                kernel.function as Handle
            )?;
            attributes.push(wire::FunctionAttribute { id, value });
        }
        functions.push(wire::Function {
            module: kernel.module.clone(),
            symbol,
            parameter_sizes: kernel.arguments.iter().map(|a| a.size as u64).collect(),
            attributes,
        });
        layouts.push(layout);
    }
    inspect_phase.finish(format!(
        "{} functions, {} modules",
        kernels.len(),
        modules.len()
    ));
    let mut device = 0;
    cu!(&cuda, "cuCtxGetDevice", (*mut c_int), &mut device)?;
    let mut architecture = Vec::new();
    for attribute in [75, 76] {
        let mut value = 0;
        cu!(
            &cuda,
            "cuDeviceGetAttribute",
            (*mut c_int, c_int, c_int),
            &mut value,
            attribute,
            device
        )?;
        architecture.push(value);
    }
    let (mut flags, mut cache) = (0u32, 0i32);
    cu!(&cuda, "cuCtxGetFlags", (*mut c_uint), &mut flags)?;
    cu!(&cuda, "cuCtxGetCacheConfig", (*mut c_int), &mut cache)?;
    let mut limits = Vec::new();
    for id in 0..7 {
        let mut value = 0usize;
        let status = status!(&cuda, "cuCtxGetLimit", (*mut usize, c_int), &mut value, id);
        if status == 215 {
            continue;
        }
        cuda.check(&format!("cuCtxGetLimit({id})"), status)?;
        limits.push(wire::Limit {
            id: id as u64,
            value: value as u64,
        });
    }
    let mut stage = Stage {
        path: stage_directory(&request.path)?,
        keep: false,
    };
    let mut output = RankOutput {
        index: wire::ModelIndex {
            version: 4,
            ranks: vec![wire::BundleIndex {
                architecture,
                functions,
                context: Some(wire::Context {
                    flags,
                    cache,
                    limits,
                }),
                events: request.events.keys().cloned().collect(),
                ..Default::default()
            }],
            ..Default::default()
        },
        payload: vec![],
        allocations: vec![],
        module_globals: vec![],
        external,
        staging: Some(stage.path.clone()),
    };
    let mut roots = request.roots.clone();
    roots.extend(
        allocations
            .iter()
            .filter(|a| request.peer_addresses.contains(&a.address))
            .map(|a| a.name.clone()),
    );
    for transport in &request.transports {
        for part in ["region", "counters"] {
            let pointer = transport[part]["address"]
                .as_u64()
                .ok_or_else(|| invalid("transport storage address is missing"))?;
            roots.push(reference(&allocations, pointer, 1)?.allocation);
        }
    }
    let mut reach = Reachability::new(&allocations, &request.numeric, &request.inactive, &roots)?;
    if let Some(execution) = &request.execution {
        let execution = super::binding::execution(execution, &allocations)?;
        for buffer in &execution.buffers {
            reach.reference(
                &buffer
                    .reference
                    .as_ref()
                    .ok_or_else(|| invalid("execution buffer lacks allocation reference"))?
                    .allocation,
            )?;
        }
        output.index.ranks[0].execution = Some(execution);
    }
    let standalone_cupti =
        if recorder.is_none() {
            Some(unsafe {
                Library::new(request.cupti_path.as_ref().ok_or_else(|| {
                    invalid("standalone graph export requires a CUPTI library path")
                })?)?
            })
        } else {
            None
        };
    let mut other_resources = Vec::new();
    let graph_phase = progress.phase("Serialize CUDA graphs", Some(request.graphs.len() as u64));
    let mut graph_bytes = 0u64;
    for (index, graph) in request.graphs.iter().enumerate() {
        let (nodes, edges, flags) = capture_graph(
            &cuda,
            graph,
            recorder,
            standalone_cupti.as_ref(),
            &allocations,
            &kernels,
            &layouts,
            &indices,
            &request.events,
        )?;
        reach.graph(&nodes, &kernels)?;
        let file = format!("graph-{index}.bin");
        let path = stage.path.join(&file);
        let mut writer = BufWriter::new(File::create(&path)?);
        for node in &nodes {
            write_record(&mut writer, node)?;
        }
        for edge in &edges {
            write_record(&mut writer, edge)?;
        }
        writer.flush()?;
        let functions: BTreeSet<_> = nodes
            .iter()
            .filter_map(|n| match &n.operation {
                Some(wire::node::Operation::Kernel(k)) => Some(k.function),
                _ => None,
            })
            .collect();
        output.index.ranks[0].graphs.push(wire::GraphIndex {
            name: graph.name.clone(),
            flags,
            resource: file.clone(),
            node_count: nodes.len() as u64,
            edge_count: edges.len() as u64,
            functions: functions.into_iter().collect(),
        });
        let source = file_source(path)?;
        graph_bytes += source.nbytes;
        other_resources.push((file, source));
        graph_phase.advance(1);
    }
    graph_phase.finish(format!(
        "{} graphs, {graph_bytes} bytes",
        request.graphs.len()
    ));
    let module_phase = progress.phase("Capture compiled modules and globals", None);
    let mut module_bytes = 0u64;
    let mut downloader = Downloader::new(&cuda)?;
    for (index, (name, image)) in modules.into_iter().enumerate() {
        let file = format!("module-{index}.cubin");
        let (exported, symbols) = cubin::redact_symbols(&cubin::strip_debug(&image)?)?;
        module_bytes += exported.len() as u64;
        module_phase.advance_bytes(exported.len() as u64);
        fs::write(stage.path.join(&file), exported)?;
        for function in &mut output.index.ranks[0].functions {
            if function.module == name {
                function.symbol = symbols
                    .get(&function.symbol)
                    .unwrap_or(&function.symbol)
                    .clone();
            }
        }
        let kernel = kernels
            .iter()
            .find(|k| k.module == name)
            .ok_or_else(|| invalid("registered module has no function"))?;
        let mut module = std::ptr::null_mut();
        cu!(
            &cuda,
            "cuFuncGetModule",
            (*mut Handle, Handle),
            &mut module,
            kernel.function as Handle
        )?;
        let mut globals = Vec::new();
        for symbol in cubin::global_symbols(&image)? {
            let symbol_c = std::ffi::CString::new(symbol.clone())?;
            let (mut address, mut size) = (0u64, 0usize);
            cu!(
                &cuda,
                "cuModuleGetGlobal_v2",
                (*mut u64, *mut usize, Handle, *const c_char),
                &mut address,
                &mut size,
                module,
                symbol_c.as_ptr()
            )?;
            output
                .module_globals
                .push((address, size as u64, format!("{name}.{symbol}")));
            let data_file = format!("module-{index}-global-{}.bin", globals.len());
            let path = stage.path.join(&data_file);
            let mut writer = BufWriter::new(File::create(&path)?);
            let global = Allocation {
                name: format!("module-global:{name}.{symbol}"),
                address,
                nbytes: size as u64,
                alignment: 1,
                initialize: "snapshot".into(),
                relocations: vec![],
                preserve_address: true,
                memory: "device".into(),
                snapshot_spans: None,
                sizing: None,
            };
            downloader.for_each(&global, &[(0, size as u64)], |offset, bytes| {
                let mut redacted = Vec::new();
                let data = if symbol.starts_with("assertFile_") {
                    if offset + bytes.len() as u64 == size as u64 && bytes.last() != Some(&0) {
                        return Err(invalid(
                            "CUDA assertion filename must be a terminated string",
                        ));
                    }
                    redacted.resize(bytes.len(), 0);
                    if offset == 0 {
                        let length = 6.min(size.saturating_sub(1));
                        redacted[..length].copy_from_slice(&b"kernel"[..length]);
                    }
                    redacted.as_slice()
                } else {
                    bytes
                };
                reach.opaque_contents(&global.name, offset, data)?;
                writer.write_all(data)?;
                module_bytes += data.len() as u64;
                module_phase.advance_bytes(data.len() as u64);
                Ok(())
            })?;
            writer.flush()?;
            other_resources.push((data_file.clone(), file_source(path)?));
            globals.push(wire::Global {
                symbol: symbols.get(&symbol).unwrap_or(&symbol).clone(),
                file: data_file,
                nbytes: size as u64,
                relocations: vec![],
            });
        }
        output.index.ranks[0].modules.push(wire::Module {
            name,
            file: file.clone(),
            globals,
        });
        other_resources.push((file.clone(), file_source(stage.path.join(file))?));
    }
    module_phase.finish(format!(
        "{module_bytes} bytes, {} global bytes downloaded",
        downloader.downloaded
    ));
    let storage_phase = progress.phase(
        "Scan and copy reachable storage",
        Some(
            allocations
                .iter()
                .filter(|a| a.initialize == "snapshot")
                .map(|a| {
                    a.snapshot_spans
                        .as_ref()
                        .map(|spans| spans.iter().map(|&(_, size)| size).sum())
                        .unwrap_or(a.nbytes)
                })
                .sum(),
        ),
    );
    let storage_download_start = downloader.downloaded;
    let mut scanned = 0u64;
    let mut stored_bytes = 0u64;
    let budget = super::io::IoBudget::for_capture(request.world_size)?;
    let store = request
        .options
        .cache
        .as_ref()
        .map(|root| super::blob::BlobStore::new(root, budget))
        .transpose()?;
    let (_, committed_downloads) = with_allocation_commits(
        request.context,
        store.as_ref(),
        |commits| {
            while let Some(allocation) = reach.take()? {
                if allocation.memory == "device"
                    && allocation.preserve_address
                    && allocation.address < request.device_address_floor
                {
                    return Err(invalid(format!(
                        "retained device allocation {:?} at {:#x} is below the restorable address floor {:#x}",
                        allocation.name, allocation.address, request.device_address_floor
                    )));
                }
                let file = format!("storage-{}.bin", output.allocations.len());
                let (source, spans) = write_allocation(
                    &mut downloader,
                    &allocation,
                    &stage.path.join(&file),
                    &mut reach,
                    store.as_ref(),
                    commits,
                    request.options.no_weights,
                    &storage_phase,
                    &mut scanned,
                )?;
                stored_bytes += source.as_ref().map(|s| s.nbytes).unwrap_or(0);
                for binding in &allocation.relocations {
                    let width = if binding.kind == "pointer" { 8 } else { 128 };
                    let mut data = vec![0; width];
                    downloader.read(
                        allocation.address + binding.offset,
                        &allocation.memory,
                        &mut data,
                    )?;
                    let mut relative = binding.clone();
                    relative.offset = 0;
                    check_bindings(&cuda, &allocations, &data, &[relative])?;
                }
                if source.is_none()
                    && allocation
                        .relocations
                        .iter()
                        .any(|r| r.kind == "tensor_map")
                {
                    return Err(invalid(
                        "tensor-map storage requires a snapshot initializer",
                    ));
                }
                let sizing = allocation
                    .sizing
                    .as_ref()
                    .map(|value| super::binding::message("Sizing", value))
                    .transpose()?;
                output.index.ranks[0].allocations.push(wire::Allocation {
                    name: allocation.name.clone(),
                    nbytes: if allocation.sizing.is_some() {
                        0
                    } else {
                        allocation.nbytes
                    },
                    alignment: allocation.alignment,
                    file: source.as_ref().map(|_| file.clone()),
                    data_spans: spans
                        .into_iter()
                        .map(|(offset, nbytes)| wire::Span {
                            offset,
                            nbytes,
                            filename: String::new(),
                        })
                        .collect(),
                    relocations: allocation
                        .relocations
                        .iter()
                        .map(relocation)
                        .collect::<ExportResult<_>>()?,
                    address: allocation.preserve_address.then_some(allocation.address),
                    memory: match allocation.memory.as_str() {
                        "mapped_host" => 1,
                        "peer" => 2,
                        _ => 0,
                    },
                    sizing,
                });
                if let Some(source) = source {
                    append_resource(&mut output, file, source)?;
                }
                output.allocations.push(allocation);
            }
            Ok(())
        },
    )?;
    downloader.drain()?;
    cuda.synchronize()?;
    storage_phase.finish(format!("{scanned} bytes scanned, {} bytes downloaded, {stored_bytes} bytes stored; {} allocations omitted", downloader.downloaded - storage_download_start + committed_downloads, allocations.len() - output.allocations.len()));
    drop(store);
    drop(downloader);
    let granule = request.granularity.max(1 << 30);
    let reservations = if request.reservations.is_empty() {
        merge_ranges(
            output
                .allocations
                .iter()
                .filter(|a| a.memory == "device")
                .map(|a| {
                    (
                        a.address & !(granule - 1),
                        (a.address + a.nbytes).div_ceil(granule) * granule,
                    )
                })
                .collect(),
        )
    } else {
        request
            .reservations
            .iter()
            .filter(|&&(start, size)| {
                output
                    .allocations
                    .iter()
                    .any(|a| a.memory == "device" && start <= a.address && a.address < start + size)
            })
            .map(|&(start, size)| (start, start + size))
            .collect()
    };
    output.index.ranks[0].reservations = reservations
        .into_iter()
        .map(|(address, end)| wire::Reservation {
            address,
            nbytes: end - address,
        })
        .collect();
    other_resources.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, source) in other_resources {
        append_resource(&mut output, name, source)?;
    }
    super::binding::finish_rank(request, &mut output)?;
    if let Some(recorder) = recorder {
        recorder.check_error()?;
    }
    stage.keep = true;
    Ok(output)
}

pub(crate) fn reset_peer_storage(context: usize, ranges: &[(u64, u64)]) -> ExportResult<()> {
    let ranges = storage_ranges(ranges)?;
    let cuda = Cuda::new()?;
    let _context = ContextGuard::enter(&cuda, context)?;
    cuda.synchronize()?;
    for (start, end) in ranges {
        cu!(
            &cuda,
            "cuMemsetD8_v2",
            (u64, u8, usize),
            start,
            0,
            (end - start) as usize
        )?;
    }
    cuda.synchronize()
}

pub(crate) fn drain_context(context: usize) -> ExportResult<()> {
    let cuda = Cuda::new()?;
    let _context = ContextGuard::enter(&cuda, context)?;
    cuda.synchronize()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn allocation(name: &str, address: u64, nbytes: u64) -> Allocation {
        Allocation {
            name: name.into(),
            address,
            nbytes,
            alignment: 256,
            initialize: "snapshot".into(),
            relocations: vec![],
            preserve_address: true,
            memory: "device".into(),
            snapshot_spans: None,
            sizing: None,
        }
    }
    #[test]
    #[ignore = "requires an explicitly assigned CUDA device"]
    fn allocation_commit_worker_binds_context_reuses_cache_and_rejects_mutation() -> ExportResult<()>
    {
        let cuda = Cuda::new()?;
        cu!(&cuda, "cuInit", (u32), 0)?;
        let previous = cuda.current()?;
        let mut context = std::ptr::null_mut();
        cu!(
            &cuda,
            "cuCtxCreate_v2",
            (*mut Handle, u32, c_int),
            &mut context,
            0,
            0
        )?;
        let size = (2 * Downloader::CHUNK + 4096) as u64;
        let mut addresses = Vec::new();
        let result = (|| -> ExportResult<()> {
            for value in [3u8, 7] {
                let mut address = 0u64;
                cu!(
                    &cuda,
                    "cuMemAlloc_v2",
                    (*mut u64, usize),
                    &mut address,
                    size as usize
                )?;
                addresses.push(address);
                cu!(
                    &cuda,
                    "cuMemsetD8_v2",
                    (u64, u8, usize),
                    address,
                    value,
                    size as usize
                )?;
            }
            cuda.synchronize()?;
            let allocations: Vec<_> = addresses
                .iter()
                .enumerate()
                .map(|(index, &address)| allocation(&format!("device-{index}"), address, size))
                .collect();
            let root = tempfile::tempdir()?;
            let store = super::super::blob::BlobStore::new(
                root.path(),
                super::super::io::IoBudget::for_capture(1)?,
            )?;
            let progress = helios_core::startup::StartupProgress::silent();
            let phase = progress.phase("Copy device allocations", None);
            let mut downloader = Downloader::new(&cuda)?;
            for pass in 0..2 {
                let mut reach = Reachability::new(&allocations, &[], &[], &[])?;
                let mut scanned = 0;
                let (_, downloaded) =
                    with_allocation_commits(context as usize, Some(&store), |commits| {
                        for allocation in &allocations {
                            let (source, spans) = write_allocation(
                                &mut downloader,
                                allocation,
                                &root.path().join(&allocation.name),
                                &mut reach,
                                Some(&store),
                                commits,
                                false,
                                &phase,
                                &mut scanned,
                            )?;
                            assert_eq!(source.unwrap().nbytes, size);
                            assert_eq!(spans, vec![(0, size)]);
                        }
                        Ok(())
                    })?;
                assert_eq!(scanned, size * 2);
                assert_eq!(downloaded, if pass == 0 { size * 2 } else { 0 });
                assert_eq!(store.written.load(Ordering::Relaxed), size * 2);
            }
            cu!(
                &cuda,
                "cuMemsetD8_v2",
                (u64, u8, usize),
                allocations[0].address,
                11,
                size as usize
            )?;
            cuda.synchronize()?;
            let mut reach = Reachability::new(&allocations, &[], &[], &[])?;
            let mut scanned = 0;
            let stopped: ExportResult<((), u64)> =
                with_allocation_commits(context as usize, Some(&store), |commits| {
                    write_allocation(
                        &mut downloader,
                        &allocations[0],
                        &root.path().join("stopped"),
                        &mut reach,
                        Some(&store),
                        commits,
                        false,
                        &phase,
                        &mut scanned,
                    )?;
                    Err(invalid("producer stopped"))
                });
            assert_eq!(stopped.unwrap_err().to_string(), "producer stopped");
            assert_eq!(store.written.load(Ordering::Relaxed), size * 3);
            let allocation = &allocations[0];
            let spans = [(0, size)];
            let mut reader = PackedDownload {
                downloader: &mut downloader,
                allocation,
                spans: &spans,
                size,
                pending: std::collections::VecDeque::new(),
                next: 0,
            };
            let other = tempfile::tempdir()?;
            let empty_store =
                super::super::blob::BlobStore::new(other.path(), store.budget.clone())?;
            let prepared = empty_store.prepare(size, false, None, |offset, output| {
                reader.fill(offset, output)
            })?;
            reader.downloader.drain()?;
            cu!(
                &cuda,
                "cuMemsetD8_v2",
                (u64, u8, usize),
                allocation.address,
                9,
                size as usize
            )?;
            cuda.synchronize()?;
            let error = with_allocation_commits(context as usize, Some(&empty_store), |commits| {
                commits.unwrap().send(AllocationCommit {
                    allocation: allocation.clone(),
                    spans: spans.to_vec(),
                    prepared,
                })?;
                Ok(())
            })
            .unwrap_err();
            assert!(
                error.to_string().contains("payload changed during export"),
                "{error}"
            );
            assert_eq!(empty_store.written.load(Ordering::Relaxed), 0);
            assert_eq!(fs::read_dir(&empty_store.root)?.count(), 0);
            downloader.drain()?;
            Ok(())
        })();
        for address in addresses {
            cu!(&cuda, "cuMemFree_v2", (u64), address)?;
        }
        cu!(&cuda, "cuCtxDestroy_v2", (Handle), context)?;
        cu!(&cuda, "cuCtxSetCurrent", (Handle), previous as Handle)?;
        result
    }

    #[test]
    fn packed_arguments_honor_abi_offsets_and_parameterless_launches() {
        let mut bytes = [0u8; 24];
        bytes[0..4].copy_from_slice(&17u32.to_le_bytes());
        bytes[8..16].copy_from_slice(&0x1020304050607080u64.to_le_bytes());
        bytes[16] = 9;
        let mut size = bytes.len();
        let mut options = [
            1usize as Handle,
            bytes.as_mut_ptr().cast(),
            2usize as Handle,
            (&mut size as *mut usize).cast(),
            std::ptr::null_mut(),
        ];
        let params = KernelParams {
            extra: options.as_mut_ptr(),
            ..Default::default()
        };
        let data = unsafe { argument_data(&params, &[(0, 4), (8, 8), (16, 1)]) }.unwrap();
        assert_eq!(
            data,
            vec![
                17u32.to_le_bytes().to_vec(),
                0x1020304050607080u64.to_le_bytes().to_vec(),
                vec![9]
            ]
        );
        assert!(
            unsafe { argument_data(&KernelParams::default(), &[]) }
                .unwrap()
                .is_empty()
        );
        assert!(unsafe { argument_data(&KernelParams::default(), &[(0, 4)]) }.is_err());
    }
    #[test]
    fn argument_options_reject_duplicate_unknown_missing_and_out_of_bounds_fields() {
        let mut bytes = [1u8; 16];
        let mut size = bytes.len();
        let mut duplicate = [
            1usize as Handle,
            bytes.as_mut_ptr().cast(),
            1usize as Handle,
            bytes.as_mut_ptr().cast(),
            std::ptr::null_mut(),
        ];
        let mut params = KernelParams {
            extra: duplicate.as_mut_ptr(),
            ..Default::default()
        };
        assert!(unsafe { argument_data(&params, &[]) }.is_err());
        let mut unknown = [3usize as Handle, std::ptr::null_mut(), std::ptr::null_mut()];
        params.extra = unknown.as_mut_ptr();
        assert!(unsafe { argument_data(&params, &[]) }.is_err());
        let mut no_size = [
            1usize as Handle,
            bytes.as_mut_ptr().cast(),
            std::ptr::null_mut(),
        ];
        params.extra = no_size.as_mut_ptr();
        assert!(unsafe { argument_data(&params, &[]) }.is_err());
        let mut good = [
            1usize as Handle,
            bytes.as_mut_ptr().cast(),
            2usize as Handle,
            (&mut size as *mut usize).cast(),
            std::ptr::null_mut(),
        ];
        params.extra = good.as_mut_ptr();
        assert!(unsafe { argument_data(&params, &[(12, 8)]) }.is_err());
        let mut pointers = [bytes.as_mut_ptr().cast()];
        params.params = pointers.as_mut_ptr();
        assert!(unsafe { argument_data(&params, &[(0, 4)]) }.is_err());
        params.extra = std::ptr::null_mut();
        let mut null_pointers = [std::ptr::null_mut()];
        params.params = null_pointers.as_mut_ptr();
        assert!(unsafe { argument_data(&params, &[(0, 4)]) }.is_err());
    }
    #[test]
    fn external_pages_preserve_only_live_byte_spans() {
        let spans = external_allocations(
            vec![
                (0x1500, 0x100),
                (0x1a00, 0x200),
                (0x2100, 0x80),
                (0x5100, 0x80),
            ],
            0x1000,
        )
        .unwrap();
        assert_eq!(spans.len(), 2);
        assert_eq!((spans[0].address, spans[0].nbytes), (0x1000, 0x2000));
        assert_eq!(
            spans[0].snapshot_spans,
            Some(vec![(0x500, 0x100), (0xa00, 0x200), (0x1100, 0x80)])
        );
        assert_eq!(spans[1].snapshot_spans, Some(vec![(0x100, 0x80)]));
    }
    #[test]
    fn trim_excludes_only_wholly_inactive_granules() {
        let a = allocation("arena", 0x1000, 0x1000);
        let pieces = live_pieces(&a, &[(0x1080, 0x1380), (0x1600, 0x1a00), (0x1f00, 0x2000)]);
        assert_eq!(
            pieces
                .iter()
                .map(|a| (a.name.as_str(), a.address, a.nbytes))
                .collect::<Vec<_>>(),
            vec![
                ("arena.0", 0x1000, 0x100),
                ("arena.1", 0x1300, 0x300),
                ("arena.2", 0x1a00, 0x500)
            ]
        );
        assert_eq!(live_pieces(&a, &[(0x1000, 0x2000)])[0].nbytes, a.nbytes);
    }
    #[test]
    fn zero_mask_preserves_neighboring_bytes_and_preexisting_holes() {
        let mut a = allocation("arena", 0x1000, 0x1000);
        a.snapshot_spans = Some(vec![(0, 0x300), (0x500, 0x500)]);
        without_zero_ranges(&mut a, &[(0x1100, 0x1200), (0x1600, 0x1800)]);
        assert_eq!(
            a.snapshot_spans,
            Some(vec![
                (0, 0x100),
                (0x200, 0x100),
                (0x500, 0x100),
                (0x800, 0x200)
            ])
        );
        without_zero_ranges(&mut a, &[(0x1000, 0x2000)]);
        assert_eq!(a.initialize, "zero");
        assert_eq!(a.snapshot_spans, None);
    }
    #[test]
    fn allocation_validation_rejects_shared_backing_and_invalid_peer_contracts() {
        assert!(
            validate_allocations(&[
                allocation("a", 0x1000, 0x1000),
                allocation("b", 0x1800, 0x100)
            ])
            .is_err()
        );
        assert!(
            validate_allocations(&[
                allocation("a", 0x1000, 0x100),
                allocation("a", 0x2000, 0x100)
            ])
            .is_err()
        );
        let mut peer = allocation("peer", 0x3000, 0x1000);
        peer.memory = "peer".into();
        assert!(validate_allocations(&[peer.clone()]).is_err());
        peer.initialize = "zero".into();
        validate_allocations(&[peer.clone()]).unwrap();
        peer.preserve_address = false;
        assert!(validate_allocations(&[peer]).is_err());
    }
    #[test]
    fn references_cover_exact_spans_and_relocations_do_not_overlap() {
        let allocations = [
            allocation("a", 0x1000, 0x100),
            allocation("b", 0x2000, 0x100),
        ];
        assert_eq!(reference(&allocations, 0x1080, 0x80).unwrap().offset, 0x80);
        assert!(reference(&allocations, 0x1080, 0x81).is_err());
        let pointer = Relocation {
            offset: 0,
            target: super::super::Reference {
                allocation: "b".into(),
                offset: 1,
            },
            kind: "pointer".into(),
        };
        check_relocations(&allocations, 8, &[pointer.clone()]).unwrap();
        assert!(check_relocations(&allocations, 7, &[pointer.clone()]).is_err());
        let mut overlap = pointer.clone();
        overlap.offset = 4;
        assert!(check_relocations(&allocations, 16, &[pointer.clone(), overlap]).is_err());
        let mut outside = pointer;
        outside.target.offset = 0x100;
        assert!(check_relocations(&allocations, 8, &[outside]).is_err());
    }
    #[test]
    fn disposable_storage_requires_nonnull_nonempty_ranges() {
        assert!(storage_ranges(&[(0, 1)]).is_err());
        assert!(storage_ranges(&[(0x1000, 0)]).is_err());
        assert!(storage_ranges(&[(u64::MAX, 1)]).is_err());
        assert_eq!(
            storage_ranges(&[(0x1200, 0x100), (0x1000, 0x200)]).unwrap(),
            vec![(0x1000, 0x1300)]
        );
    }

    #[test]
    fn graph_records_round_trip_native_protobuf_and_enforce_record_bound() {
        let node = wire::Node {
            operation: Some(wire::node::Operation::Copy(wire::Copy {
                source: Some(wire::Reference {
                    allocation: "a".into(),
                    offset: 0,
                }),
                destination: Some(wire::Reference {
                    allocation: "b".into(),
                    offset: 8,
                }),
                nbytes: 128,
            })),
        };
        let mut bytes = Vec::new();
        write_record(&mut bytes, &node).unwrap();
        let nbytes = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
        assert_eq!(nbytes, bytes.len() - 4);
        let decoded = wire::Node::decode(&bytes[4..]).unwrap();
        assert_eq!(decoded, node);
        assert!(
            write_record(
                &mut Vec::new(),
                &wire::Blob {
                    data: vec![0; 1 << 20],
                    relocations: vec![]
                }
            )
            .is_err()
        );
    }
}
