//! Native ownership and reconstruction of prepared CUDA graph packages.

mod archive;
mod cache_memory;
#[cfg(test)]
mod cooperative_tests;
mod describe;
pub use describe::InspectionOptions;
mod device_memory;
pub mod driver;
mod engram;
#[cfg(test)]
mod event_tests;
mod format;
#[cfg(any(windows, test))]
mod host;
mod limits;
mod metadata;
pub use metadata::unverified_metadata;
mod memory;
mod model;
mod payload;
mod protection;
mod protobuf;
pub use protobuf::wire;
pub fn validate_sampling_defaults(value: &wire::DefaultSamplingParameters) -> crate::Result<()> {
    protobuf::sampling_defaults(value.clone()).map(|_| ())
}
pub const WIRE_DESCRIPTOR: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/runtime_package_descriptor.bin"));
mod publishers;
#[cfg(any(test, feature = "test-tools"))]
pub mod test_support;
pub use protection::{
    verify_package_metadata, AuthorizedPackage, PackageClaim, PackageLicense, Publisher,
    VerifiedPackage,
};
mod selection;
mod serving;
#[cfg(test)]
mod shared_memory_tests;
pub use archive::{PackageFile, package_companions};
pub use limits::{KvCacheLimit, MemoryLimit};
pub use model::{
    CacheAgreement, ClusterRank, NativeModel, RankNeeds, RoceSpec, package_cluster, rank_needs,
    rank_ranges, reserve_bytes,
};
pub use selection::LoadOptions;

use std::collections::HashMap;
use std::ffi::CString;
use std::ptr::{null, null_mut};
use std::sync::Arc;

use crate::{
    BufferSpan, GraphBundle, Result,
    startup::{Phase, StartupProgress},
};
use archive::Archive;
use driver::{
    Access, AllocationProperties, CopyEndpoint, CopyParams, Driver, Handle, KernelParams, Location,
    MemsetParams, TensorMap,
};
use format::{Blob, Memory, Node, Package, Reference, Relocation, RelocationKind, invalid};

struct Storage {
    base: u64,
    address: u64,
    nbytes: usize,
    memory: Memory,
    fixed: bool,
    mappings: Vec<(u64, usize)>,
}

/// Owns a CUDA context and every allocation, module, and graph loaded into it.
/// Borrowed handles remain valid until this owner is dropped, after replay drains.
pub struct NativeResources {
    driver: Arc<Driver>,
    context: Handle,
    device: i32,
    allocations: HashMap<String, Storage>,
    modules: Vec<Handle>,
    functions: HashMap<usize, Handle>,
    templates: Vec<Handle>,
    executables: Vec<Handle>,
    names: Vec<String>,
    reservations: Vec<(u64, usize)>,
    execution: Option<protobuf::wire::Execution>,
    // An integrated GPU (GB10) registers host memory without pinning it.
    integrated: bool,
    events: HashMap<String, Handle>,
    cache: Option<crate::offload::CacheGeometry>,
    #[cfg(windows)]
    host_mappings: Vec<(u64, usize)>,
}

// SAFETY: CUDA contexts may move between threads. Resource operations bind the
// owning context, and mutable ownership serializes loading and destruction.
unsafe impl Send for NativeResources {}

fn clamp_cooperative_grid(mut grid: [u32; 3], capacity: u64) -> Result<[u32; 3]> {
    let axis = (0..3).max_by_key(|&axis| grid[axis]).unwrap();
    let plane = u64::from(grid[(axis + 1) % 3]) * u64::from(grid[(axis + 2) % 3]);
    let max_workers = capacity / plane;
    if max_workers == 0 {
        return Err(invalid(format!(
            "cooperative kernel grid {grid:?} cannot fit {capacity} resident CTAs by reducing one dimension",
        )));
    }
    grid[axis] = u64::from(grid[axis]).min(max_workers) as u32;
    Ok(grid)
}

impl NativeResources {
    /// Load compiled code and persistent storage, then bind and instantiate graphs.
    ///
    /// # Safety
    /// The package must be trusted. Its kernels and declared bindings must obey
    /// CUDA's memory and execution contracts, including any mapped host access.
    /// Cooperative kernels must tolerate fewer worker CTAs in the largest grid dimension.
    /// This method makes the owner's context current on the calling thread.
    pub unsafe fn load(authorized: AuthorizedPackage, ordinal: i32) -> Result<Self> {
        let AuthorizedPackage { archive, mut index } = authorized;
        if index.ranks.len() != 1 || !index.peer_memory.is_empty() || index.engram.is_some() {
            return Err(invalid("resource loader requires one independent rank"));
        }
        let package = protobuf::decode(index.ranks.remove(0))?;
        archive.validate(&package)?;
        package.validate()?;
        if package.allocations.iter().any(|a| a.sizing.is_some()) {
            return Err(invalid(
                "runtime-sized storage requires a model memory plan",
            ));
        }
        let phase = StartupProgress::silent().phase("Load model", None);
        let mut owner = unsafe { Self::initialize(&package, ordinal)? };
        unsafe {
            owner.reserve(&package)?;
            owner.allocate(&package, &phase)?;
            owner.load_payload(&archive, &package, &phase, 1)?;
            owner.prepare_kernels(&archive, &package, &phase)?;
            owner.prepare_graphs(&archive, &package, &phase)?;
        }
        phase.finish("");
        Ok(owner)
    }

    unsafe fn initialize(package: &Package, ordinal: i32) -> Result<Self> {
        let driver = Arc::new(Driver::load()?);
        // SAFETY: Driver signatures are checked against cuda.h; output storage is
        // valid, and CUDA initialization precedes all context/resource operations.
        unsafe {
            driver.check("cuInit", (driver.init)(0))?;
            let mut device = 0;
            driver.check("cuDeviceGet", (driver.device)(&mut device, ordinal))?;
            for (attribute, expected) in [75, 76].into_iter().zip(package.architecture) {
                let mut value = 0;
                driver.check(
                    "cuDeviceGetAttribute",
                    (driver.device_attribute)(&mut value, attribute, device),
                )?;
                if value != expected {
                    return Err(invalid("package GPU architecture does not match device"));
                }
            }
            let mut integrated = 0;
            driver.check(
                "cuDeviceGetAttribute",
                (driver.device_attribute)(&mut integrated, 18, device),
            )?;
            let mut context = null_mut();
            driver.check(
                "cuCtxCreate_v2",
                (driver.context_create)(&mut context, package.context.flags, device),
            )?;
            let owner = Self {
                driver,
                context,
                device,
                allocations: HashMap::new(),
                modules: vec![],
                functions: HashMap::new(),
                templates: vec![],
                executables: vec![],
                names: vec![],
                reservations: vec![],
                execution: package.execution.clone(),
                integrated: integrated != 0,
                events: HashMap::new(),
                cache: None,
                #[cfg(windows)]
                host_mappings: Vec::new(),
            };
            for [limit, value] in &package.context.limits {
                owner.driver.check(
                    "cuCtxSetLimit",
                    (owner.driver.context_limit)(*limit as i32, *value),
                )?;
            }
            owner.driver.check(
                "cuCtxSetCacheConfig",
                (owner.driver.context_cache)(package.context.cache),
            )?;
            Ok(owner)
        }
    }

    fn address(&self, reference: &Reference) -> u64 {
        self.allocations[&reference.allocation].address + reference.offset as u64
    }

    fn relocate(&self, data: &mut [u8], relocation: &Relocation, offset: usize) -> Result<()> {
        let address = self.address(&relocation.target);
        match relocation.kind {
            RelocationKind::Pointer => {
                data[offset..offset + 8].copy_from_slice(&address.to_le_bytes())
            }
            RelocationKind::TensorMap => {
                let mut descriptor = TensorMap(data[offset..offset + 128].try_into().unwrap());
                // SAFETY: TensorMap has CUDA's required 64-byte alignment. The
                // descriptor and replacement allocation were validated before use.
                self.driver.check("cuTensorMapReplaceAddress", unsafe {
                    (self.driver.tensor_map_replace)(&mut descriptor, address as Handle)
                })?;
                data[offset..offset + 128].copy_from_slice(&descriptor.0);
            }
        }
        Ok(())
    }

    fn bind(&self, blob: &Blob) -> Result<Vec<u8>> {
        let mut data = blob.data.clone();
        for relocation in &blob.relocations {
            self.relocate(&mut data, relocation, relocation.offset)?;
        }
        Ok(data)
    }

    unsafe fn reserve(&mut self, package: &Package) -> Result<()> {
        let driver = &self.driver;
        unsafe {
            driver.check("cuCtxSetCurrent", (driver.context_set)(self.context))?;
            for reservation in &package.reservations {
                let mut address = 0;
                driver.check(
                    "cuMemAddressReserve",
                    (driver.reserve)(&mut address, reservation.nbytes, 0, reservation.address, 0),
                )?;
                self.reservations.push((address, reservation.nbytes));
                if address != reservation.address {
                    return Err(invalid(format!(
                        "CUDA address collision: requested {:#x}, reserved {address:#x}",
                        reservation.address
                    )));
                }
            }
        }
        Ok(())
    }

    unsafe fn allocate(&mut self, package: &Package, phase: &Phase) -> Result<()> {
        #[cfg(windows)]
        if self.host_mappings.is_empty() {
            unsafe { self.allocate_host(package)? };
        }
        let driver = self.driver.clone();
        // SAFETY: Package validation bounds all indexes and relocation fields.
        // Every CUDA resource is immediately retained by self for error cleanup.
        unsafe {
            driver.check("cuCtxSetCurrent", (driver.context_set)(self.context))?;
            let footprint = package
                .allocations
                .iter()
                .filter(|allocation| {
                    matches!(allocation.memory, Memory::Device)
                        && !self
                            .allocations
                            .get(&allocation.name)
                            .is_some_and(|a| a.nbytes > 0)
                })
                .try_fold(0usize, |total, allocation| {
                    total.checked_add(allocation.nbytes)
                })
                .ok_or_else(|| invalid("package device footprint overflows address space"))?;
            let free = self.available_memory()?;
            if footprint > free {
                return Err(invalid(format!(
                    "package needs {footprint} device bytes; only {free} are free"
                )));
            }
            for allocation in &package.allocations {
                if self
                    .allocations
                    .get(&allocation.name)
                    .is_some_and(|a| a.nbytes > 0)
                {
                    continue;
                }
                if allocation.sizing.is_some() {
                    self.allocations.insert(
                        allocation.name.clone(),
                        Storage {
                            base: 0,
                            address: allocation.address.unwrap(),
                            nbytes: 0,
                            memory: allocation.memory,
                            fixed: true,
                            mappings: Vec::new(),
                        },
                    );
                    continue;
                }
                let mut base = 0;
                let size = allocation.nbytes + allocation.alignment - 1;
                let address = if let Some(address) = allocation.address {
                    match allocation.memory {
                        Memory::Device => {
                            let properties = AllocationProperties {
                                kind: 1,
                                handles: 0,
                                location: Location {
                                    kind: 1,
                                    device: self.device,
                                },
                                windows: null_mut(),
                                flags: [0; 8],
                            };
                            let mut physical = 0;
                            driver.check(
                                "cuMemCreate",
                                (driver.physical_create)(
                                    &mut physical,
                                    allocation.nbytes,
                                    &properties,
                                    0,
                                ),
                            )?;
                            let mapped = (driver.map)(address, allocation.nbytes, 0, physical, 0);
                            (driver.physical_release)(physical);
                            driver.check("cuMemMap", mapped)?;
                        }
                        Memory::MappedHost => {
                            #[cfg(unix)]
                            {
                                let mapped = libc::mmap(
                                    address as Handle,
                                    allocation.nbytes,
                                    libc::PROT_READ | libc::PROT_WRITE,
                                    libc::MAP_PRIVATE
                                        | libc::MAP_ANONYMOUS
                                        | libc::MAP_FIXED_NOREPLACE,
                                    -1,
                                    0,
                                );
                                if mapped != address as Handle {
                                    return Err(invalid(format!(
                                        "host address collision at {address:#x}: {}",
                                        std::io::Error::last_os_error()
                                    )));
                                }
                                // Registration does not pin host memory for an integrated GPU, and a
                                // swapped-out page turns each GPU access during replay into a fault.
                                if self.integrated && libc::mlock(mapped, allocation.nbytes) != 0 {
                                    let error = std::io::Error::last_os_error();
                                    libc::munmap(mapped, allocation.nbytes);
                                    return Err(invalid(format!(
                                        "cannot lock {} bytes of mapped host storage: {error}; raise RLIMIT_MEMLOCK",
                                        allocation.nbytes
                                    )));
                                }
                                let status = (driver.host_register)(mapped, allocation.nbytes, 2);
                                if status != 0 {
                                    libc::munmap(mapped, allocation.nbytes);
                                }
                                driver.check("cuMemHostRegister_v2", status)?;
                            }
                        }
                        // The owning rank maps peer storage and grants this rank access.
                        Memory::Peer => {}
                    }
                    address
                } else {
                    driver.check("cuMemAlloc_v2", (driver.allocate)(&mut base, size))?;
                    (base + allocation.alignment as u64 - 1) & !(allocation.alignment as u64 - 1)
                };
                self.allocations.insert(
                    allocation.name.clone(),
                    Storage {
                        base,
                        address,
                        nbytes: allocation.nbytes,
                        memory: allocation.memory,
                        fixed: allocation.address.is_some(),
                        mappings: if allocation.address.is_some()
                            && matches!(allocation.memory, Memory::Device)
                        {
                            vec![(address, allocation.nbytes)]
                        } else {
                            Vec::new()
                        },
                    },
                );
                if allocation.address.is_some() {
                    match allocation.memory {
                        Memory::Device => {
                            let access = Access {
                                location: Location {
                                    kind: 1,
                                    device: self.device,
                                },
                                flags: 3,
                            };
                            driver.check(
                                "cuMemSetAccess",
                                (driver.access)(address, allocation.nbytes, &access, 1),
                            )?;
                        }
                        Memory::MappedHost => {
                            #[cfg(unix)]
                            {
                                let mut gpu = 0;
                                driver.check(
                                    "cuMemHostGetDevicePointer_v2",
                                    (driver.host_device)(&mut gpu, address as Handle, 0),
                                )?;
                                if gpu != address {
                                    return Err(invalid(
                                        "mapped host address differs from recorded CUDA address",
                                    ));
                                }
                            }
                        }
                        Memory::Peer => {}
                    }
                }
                if allocation.stored_bytes() < allocation.nbytes {
                    match allocation.memory {
                        Memory::MappedHost => {
                            std::ptr::write_bytes(address as *mut u8, 0, allocation.nbytes)
                        }
                        Memory::Device => driver.check(
                            "cuMemsetD8_v2",
                            (driver.zero)(address, 0, allocation.nbytes),
                        )?,
                        Memory::Peer => {}
                    }
                }
                phase.advance(allocation.nbytes as u64);
            }
            driver.check("cuCtxSynchronize", (driver.context_sync)())?;
        }
        Ok(())
    }

    unsafe fn load_payload(
        &mut self,
        archive: &Archive,
        package: &Package,
        phase: &Phase,
        ranks: usize,
    ) -> Result<()> {
        let driver = self.driver.clone();
        unsafe {
            driver.check("cuCtxSetCurrent", (driver.context_set)(self.context))?;
            payload::load(self, archive, package, phase, ranks)?;
            for allocation in &package.allocations {
                for relocation in &allocation.relocations {
                    let address =
                        self.allocations[&allocation.name].address + relocation.offset as u64;
                    let mut bytes = vec![0u8; relocation.width()];
                    driver.check(
                        "cuMemcpyDtoH_v2",
                        (driver.download)(bytes.as_mut_ptr().cast(), address, bytes.len()),
                    )?;
                    self.relocate(&mut bytes, relocation, 0)?;
                    driver.check(
                        "cuMemcpyHtoD_v2",
                        (driver.upload)(address, bytes.as_ptr().cast(), bytes.len()),
                    )?;
                }
            }
        }
        Ok(())
    }

    unsafe fn prepare_kernels(
        &mut self,
        archive: &Archive,
        package: &Package,
        phase: &Phase,
    ) -> Result<()> {
        let driver = self.driver.clone();
        unsafe {
            driver.check("cuCtxSetCurrent", (driver.context_set)(self.context))?;
            let mut modules = HashMap::new();
            for module in &package.modules {
                let image = archive.read(package, &module.file)?;
                if !image.starts_with(b"\x7fELF") {
                    return Err(invalid("module image is not a compiled cubin"));
                }
                let mut handle = null_mut();
                driver.check(
                    "cuModuleLoadData",
                    (driver.module_load)(&mut handle, image.as_ptr().cast()),
                )?;
                self.modules.push(handle);
                modules.insert(&module.name, handle);
                for global in &module.globals {
                    let symbol = CString::new(global.symbol.as_str()).unwrap();
                    let (mut address, mut size) = (0, 0);
                    driver.check(
                        "cuModuleGetGlobal_v2",
                        (driver.module_global)(&mut address, &mut size, handle, symbol.as_ptr()),
                    )?;
                    let mut contents = archive.read(package, &global.file)?;
                    if size != global.nbytes || contents.len() != size {
                        return Err(invalid(format!(
                            "module global size mismatch for {}",
                            global.symbol
                        )));
                    }
                    for relocation in &global.relocations {
                        self.relocate(&mut contents, relocation, relocation.offset)?;
                    }
                    driver.check(
                        "cuMemcpyHtoD_v2",
                        (driver.upload)(address, contents.as_ptr().cast(), size),
                    )?;
                }
                phase.advance(1);
            }
            for function in &package.functions {
                let symbol = CString::new(function.symbol.as_str()).unwrap();
                let mut handle = null_mut();
                driver.check(
                    "cuModuleGetFunction",
                    (driver.function)(&mut handle, modules[&function.module], symbol.as_ptr()),
                )?;
                driver.check("cuFuncLoad", (driver.function_load)(handle))?;
                for (index, expected) in function.parameter_sizes.iter().enumerate() {
                    let (mut offset, mut size) = (0, 0);
                    driver.check(
                        "cuFuncGetParamInfo",
                        (driver.parameter_info)(handle, index, &mut offset, &mut size),
                    )?;
                    if size != *expected {
                        return Err(invalid(
                            "loaded kernel parameter layout differs from package",
                        ));
                    }
                }
                let (mut offset, mut size) = (0, 0);
                let status = (driver.parameter_info)(
                    handle,
                    function.parameter_sizes.len(),
                    &mut offset,
                    &mut size,
                );
                if status != 1 {
                    return Err(invalid(
                        "loaded kernel parameter count differs from package",
                    ));
                }
                for [attribute, value] in &function.attributes {
                    driver.check(
                        "cuFuncSetAttribute",
                        (driver.function_attribute)(handle, *attribute, *value),
                    )?;
                }
                self.functions.insert(function.id, handle);
                phase.advance(1);
            }
        }
        Ok(())
    }

    unsafe fn prepare_graphs(
        &mut self,
        archive: &Archive,
        package: &Package,
        phase: &Phase,
    ) -> Result<()> {
        let driver = self.driver.clone();
        unsafe {
            driver.check("cuCtxSetCurrent", (driver.context_set)(self.context))?;
            for name in &package.events {
                let mut event = null_mut();
                driver.check("cuEventCreate", (driver.event_create)(&mut event, 2))?;
                self.events.insert(name.clone(), event);
            }
            let mut dynamic_shared = HashMap::new();
            for &function in self.functions.values() {
                let mut bytes = 0;
                driver.check(
                    "cuFuncGetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
                    (driver.function_get_attribute)(&mut bytes, 8, function),
                )?;
                dynamic_shared.insert(function as usize, bytes as u32);
            }
            for graph in &package.graphs {
                let mut handle = null_mut();
                driver.check("cuGraphCreate", (driver.graph_create)(&mut handle, 0))?;
                self.templates.push(handle);
                let mut nodes = Vec::with_capacity(graph.node_count);
                let mut records = archive.records(package, &graph.resource)?;
                for _ in 0..graph.node_count {
                    let node = protobuf::node(records.read()?)?;
                    package.validate_node(&node)?;
                    if let Node::Kernel { function, .. } = &node {
                        if !graph.functions.contains(function) {
                            return Err(invalid(
                                "graph kernel is absent from its dependency index",
                            ));
                        }
                    }
                    let node = &node;
                    let mut result = null_mut();
                    match node {
                        Node::Kernel {
                            function,
                            grid,
                            block,
                            shared,
                            arguments,
                            attributes,
                        } => {
                            let function_handle = self.functions[function];
                            let allowance =
                                dynamic_shared.get_mut(&(function_handle as usize)).unwrap();
                            // Captured nodes retain their launch requirements even if later
                            // preparation lowers the function's dynamic shared-memory allowance.
                            if *shared > *allowance {
                                let bytes = i32::try_from(*shared).map_err(|_| {
                                    invalid(
                                        "kernel shared-memory size exceeds CUDA's attribute range",
                                    )
                                })?;
                                driver.check(
                                    "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
                                    (driver.function_attribute)(function_handle, 8, bytes),
                                )?;
                                *allowance = *shared;
                            }
                            let mut storage = arguments
                                .iter()
                                .map(|a| self.bind(a))
                                .collect::<Result<Vec<_>>>()?;
                            let mut pointers: Vec<_> =
                                storage.iter_mut().map(|a| a.as_mut_ptr().cast()).collect();
                            let mut params = KernelParams {
                                function: function_handle,
                                grid: *grid,
                                block: *block,
                                shared: *shared,
                                arguments: pointers.as_mut_ptr(),
                                extra: null_mut(),
                                kernel: null_mut(),
                                context: null_mut(),
                            };
                            let status =
                                (driver.graph_kernel)(&mut result, handle, null(), 0, &params);
                            if status != 0 {
                                let symbol = package
                                    .functions
                                    .iter()
                                    .find(|entry| entry.id == *function)
                                    .map(|entry| entry.symbol.as_str())
                                    .unwrap_or("unknown");
                                driver.check(
                                    &format!(
                                        "cuGraphAddKernelNode_v2 for graph {} node {} kernel {symbol} (grid {grid:?}, block {block:?}, shared {shared})",
                                        graph.name, nodes.len()
                                    ),
                                    status,
                                )?;
                            }
                            for attribute in attributes {
                                let bytes = self.bind(&attribute.value)?;
                                let mut value = [0u64; 8];
                                std::ptr::copy_nonoverlapping(
                                    bytes.as_ptr(),
                                    value.as_mut_ptr().cast(),
                                    64,
                                );
                                let mut status = (driver.graph_attribute)(
                                    result,
                                    attribute.id,
                                    value.as_ptr().cast(),
                                );
                                if attribute.id == 2 && value[0] as u32 != 0 && status == 720 {
                                    self.clamp_cooperative_launch(&mut params)?;
                                    driver.check(
                                        "cuGraphKernelNodeSetParams_v2",
                                        (driver.graph_kernel_params)(result, &params),
                                    )?;
                                    status = (driver.graph_attribute)(
                                        result,
                                        attribute.id,
                                        value.as_ptr().cast(),
                                    );
                                }
                                driver.check("cuGraphKernelNodeSetAttribute", status)?;
                            }
                        }
                        Node::Memset {
                            destination,
                            pitch,
                            value,
                            element_size,
                            width,
                            height,
                        } => {
                            let params = MemsetParams {
                                destination: self.address(destination),
                                pitch: *pitch,
                                value: *value,
                                element_size: *element_size,
                                width: *width,
                                height: *height,
                            };
                            driver.check(
                                "cuGraphAddMemsetNode",
                                (driver.graph_memset)(
                                    &mut result,
                                    handle,
                                    null(),
                                    0,
                                    &params,
                                    self.context,
                                ),
                            )?;
                        }
                        Node::Empty => {
                            driver.check(
                                "cuGraphAddEmptyNode",
                                (driver.graph_empty)(&mut result, handle, null(), 0),
                            )?;
                        }
                        Node::EventWait { event } => {
                            driver.check(
                                "cuGraphAddEventWaitNode",
                                (driver.graph_event_wait)(
                                    &mut result,
                                    handle,
                                    null(),
                                    0,
                                    self.events[event],
                                ),
                            )?;
                        }
                        Node::Copy {
                            source,
                            destination,
                            nbytes,
                        } => {
                            let endpoint = |reference| CopyEndpoint {
                                memory_type: 2,
                                device: self.address(reference),
                                pitch: *nbytes,
                                height: 1,
                                ..CopyEndpoint::default()
                            };
                            let params = CopyParams {
                                source: endpoint(source),
                                destination: endpoint(destination),
                                width: *nbytes,
                                height: 1,
                                depth: 1,
                            };
                            driver.check(
                                "cuGraphAddMemcpyNode",
                                (driver.graph_copy)(
                                    &mut result,
                                    handle,
                                    null(),
                                    0,
                                    &params,
                                    self.context,
                                ),
                            )?;
                        }
                    }
                    nodes.push(result);
                    phase.advance(1);
                }
                let mut dependencies = format::Dependencies::new(graph.node_count);
                for begin in (0..graph.edge_count).step_by(4096) {
                    let mut sources = Vec::new();
                    let mut destinations = Vec::new();
                    let mut data = Vec::new();
                    for _ in begin..(begin + 4096).min(graph.edge_count) {
                        let edge = protobuf::edge(records.read()?)?;
                        dependencies.add(&edge)?;
                        sources.push(nodes[edge.source]);
                        destinations.push(nodes[edge.destination]);
                        data.push(edge.data);
                    }
                    driver.check(
                        "cuGraphAddDependencies_v2",
                        (driver.graph_edges)(
                            handle,
                            sources.as_ptr(),
                            destinations.as_ptr(),
                            data.as_ptr(),
                            data.len(),
                        ),
                    )?;
                }
                dependencies.finish()?;
                records.finish()?;
                let mut executable = null_mut();
                driver.check(
                    "cuGraphInstantiateWithFlags",
                    (driver.graph_instantiate)(&mut executable, handle, graph.flags),
                )?;
                self.executables.push(executable);
                self.names.push(graph.name.clone());
                driver.check(
                    "cuGraphUpload",
                    (driver.graph_upload)(executable, null_mut()),
                )?;
                driver.check("cuCtxSynchronize", (driver.context_sync)())?;
                driver.check("cuGraphDestroy", (driver.graph_destroy)(handle))?;
                self.templates.pop();
                phase.advance(1);
            }
            self.functions.clear();
            driver.check("cuCtxSynchronize", (driver.context_sync)())?;
        }
        Ok(())
    }

    unsafe fn clamp_cooperative_launch(&self, params: &mut KernelParams) -> Result<()> {
        let threads = params
            .block
            .iter()
            .try_fold(1i32, |n, &dim| n.checked_mul(i32::try_from(dim).ok()?))
            .ok_or_else(|| invalid("cooperative kernel block size overflow"))?;
        let (mut blocks_per_sm, mut sms) = (0, 0);
        unsafe {
            self.driver.check(
                "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                (self.driver.occupancy)(
                    &mut blocks_per_sm,
                    params.function,
                    threads,
                    params.shared as usize,
                ),
            )?;
            self.driver.check(
                "cuDeviceGetAttribute",
                (self.driver.device_attribute)(&mut sms, 16, self.device),
            )?;
        }
        if blocks_per_sm <= 0 || sms <= 0 {
            return Err(invalid("cooperative kernel has no resident CTA capacity"));
        }
        let capacity = blocks_per_sm as u64 * sms as u64;
        params.grid = clamp_cooperative_grid(params.grid, capacity)?;
        Ok(())
    }

    pub fn graph_names(&self) -> &[String] {
        &self.names
    }

    /// The rank's prefix checkpoint storage, present after cache allocation.
    pub fn cache_geometry(&self) -> Option<&crate::offload::CacheGeometry> {
        self.cache.as_ref()
    }

    pub(super) fn context_handle(&self) -> (usize, i32) {
        (self.context as usize, self.device)
    }

    pub fn graph_bundle(&self) -> GraphBundle {
        GraphBundle {
            context: self.context as usize,
            device: self.device,
            graph_execs: self.executables.iter().map(|v| *v as usize).collect(),
            buffers: self
                .allocations
                .values()
                .filter(|a| !matches!(a.memory, Memory::Peer))
                .map(|a| BufferSpan {
                    address: a.address as usize,
                    nbytes: a.nbytes,
                })
                .collect(),
        }
    }

    pub fn allocation(&self, name: &str) -> Result<BufferSpan> {
        let a = self
            .allocations
            .get(name)
            .ok_or_else(|| invalid(format!("unknown allocation {name:?}")))?;
        Ok(BufferSpan {
            address: a.address as usize,
            nbytes: a.nbytes,
        })
    }

    /// Copy an allocation after the caller has drained every replay that uses it.
    pub fn download(&self, name: &str) -> Result<Vec<u8>> {
        let a = self.allocation(name)?;
        let mut bytes = vec![0u8; a.nbytes];
        // SAFETY: The allocation remains owned by self and the output is sized to it.
        unsafe {
            self.driver
                .check("cuCtxSetCurrent", (self.driver.context_set)(self.context))?;
            self.driver
                .check("cuCtxSynchronize", (self.driver.context_sync)())?;
            self.driver.check(
                "cuMemcpyDtoH_v2",
                (self.driver.download)(bytes.as_mut_ptr().cast(), a.address as u64, bytes.len()),
            )?;
        }
        Ok(bytes)
    }

    /// Replace allocation contents after draining replay on the owned context.
    pub fn upload(&self, name: &str, bytes: &[u8]) -> Result<()> {
        let a = self.allocation(name)?;
        if bytes.len() != a.nbytes {
            return Err(invalid("upload size differs from allocation"));
        }
        // SAFETY: The owned allocation and input span have equal validated sizes.
        unsafe {
            self.driver
                .check("cuCtxSetCurrent", (self.driver.context_set)(self.context))?;
            self.driver
                .check("cuCtxSynchronize", (self.driver.context_sync)())?;
            self.driver.check(
                "cuMemcpyHtoD_v2",
                (self.driver.upload)(a.address as u64, bytes.as_ptr().cast(), bytes.len()),
            )?;
        }
        Ok(())
    }
}

impl Drop for NativeResources {
    fn drop(&mut self) {
        // SAFETY: This owner is the sole resource destroyer. On a failed drain,
        // retain the driver and all GPU resources until process termination.
        unsafe {
            let mut previous = null_mut();
            (self.driver.context_get)(&mut previous);
            if (self.driver.context_set)(self.context) != 0 || (self.driver.context_sync)() != 0 {
                std::mem::forget(self.driver.clone());
                return;
            }
            for executable in self.executables.iter().rev() {
                (self.driver.executable_destroy)(*executable);
            }
            for graph in self.templates.iter().rev() {
                (self.driver.graph_destroy)(*graph);
            }
            for event in self.events.values() {
                (self.driver.event_destroy)(*event);
            }
            for module in self.modules.iter().rev() {
                (self.driver.module_unload)(*module);
            }
            for allocation in self.allocations.values() {
                if allocation.nbytes == 0 {
                    continue;
                }
                match (allocation.memory, allocation.fixed) {
                    (Memory::Device, false) => {
                        (self.driver.free)(allocation.base);
                    }
                    (Memory::Device, true) => {
                        for &(address, bytes) in &allocation.mappings {
                            (self.driver.unmap)(address, bytes);
                        }
                    }
                    (Memory::MappedHost, _) => {
                        #[cfg(unix)]
                        {
                            (self.driver.host_unregister)(allocation.address as Handle);
                            libc::munmap(allocation.address as Handle, allocation.nbytes);
                        }
                    }
                    (Memory::Peer, _) => {}
                }
            }
            #[cfg(windows)]
            for &(address, bytes) in self.host_mappings.iter().rev() {
                (self.driver.unmap)(address, bytes);
            }
            for (address, size) in self.reservations.iter().rev() {
                (self.driver.release_address)(*address, *size);
            }
            (self.driver.context_destroy)(self.context);
            if previous != self.context {
                (self.driver.context_set)(previous);
            }
        }
    }
}
