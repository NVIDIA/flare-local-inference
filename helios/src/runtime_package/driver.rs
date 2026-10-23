use std::ffi::{CStr, c_char, c_int, c_uint, c_void};

use libloading::Library;

use crate::{CudaError, ExecutionError, Result};

pub type Handle = *mut c_void;

#[repr(C)]
pub struct KernelParams {
    pub function: Handle,
    pub grid: [c_uint; 3],
    pub block: [c_uint; 3],
    pub shared: c_uint,
    pub arguments: *mut Handle,
    pub extra: *mut Handle,
    pub kernel: Handle,
    pub context: Handle,
}

#[repr(C)]
pub struct MemsetParams {
    pub destination: u64,
    pub pitch: usize,
    pub value: c_uint,
    pub element_size: c_uint,
    pub width: usize,
    pub height: usize,
}

#[repr(C)]
#[derive(Default)]
pub struct CopyEndpoint {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub lod: usize,
    pub memory_type: c_int,
    pub host: Handle,
    pub device: u64,
    pub array: Handle,
    pub reserved: Handle,
    pub pitch: usize,
    pub height: usize,
}

#[repr(C)]
pub struct CopyParams {
    pub source: CopyEndpoint,
    pub destination: CopyEndpoint,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

#[repr(C, align(64))]
pub struct TensorMap(pub [u8; 128]);

#[repr(C)]
pub struct Location {
    pub kind: c_int,
    pub device: c_int,
}

#[repr(C)]
pub struct AllocationProperties {
    pub kind: c_int,
    pub handles: c_int,
    pub location: Location,
    pub windows: Handle,
    pub flags: [u8; 8],
}

#[repr(C)]
pub struct Access {
    pub location: Location,
    pub flags: c_int,
}

macro_rules! driver {
    ($($field:ident: $symbol:literal($($arg:ty),*) );* $(;)?) => {
        pub struct Driver {
            _library: Library,
            $(pub $field: unsafe extern "C" fn($($arg),*) -> c_int,)*
        }

        impl Driver {
            pub fn load() -> Result<Self> {
                // SAFETY: These signatures match the CUDA driver ABI; the library
                // remains loaded for the lifetime of all resolved entrypoints.
                unsafe {
                    let library = crate::os::cuda_library().map_err(|e| e.to_string())?;
                    $(let $field = *library.get(concat!($symbol, "\0").as_bytes())
                        .map_err(|e| e.to_string())?;)*
                    Ok(Self { _library: library, $($field,)* })
                }
            }
        }
    };
}

driver! {
    init: "cuInit"(c_uint);
    device: "cuDeviceGet"(*mut c_int, c_int);
    device_total_memory: "cuDeviceTotalMem_v2"(*mut usize, c_int);
    device_pci_bus_id: "cuDeviceGetPCIBusId"(*mut c_char, c_int, c_int);
    device_attribute: "cuDeviceGetAttribute"(*mut c_int, c_int, c_int);
    context_create: "cuCtxCreate_v2"(*mut Handle, c_uint, c_int);
    context_set: "cuCtxSetCurrent"(Handle);
    context_get: "cuCtxGetCurrent"(*mut Handle);
    context_sync: "cuCtxSynchronize"();
    context_destroy: "cuCtxDestroy_v2"(Handle);
    context_limit: "cuCtxSetLimit"(c_int, usize);
    context_cache: "cuCtxSetCacheConfig"(c_int);
    allocate: "cuMemAlloc_v2"(*mut u64, usize);
    memory_info: "cuMemGetInfo_v2"(*mut usize, *mut usize);
    free: "cuMemFree_v2"(u64);
    reserve: "cuMemAddressReserve"(*mut u64, usize, usize, u64, u64);
    release_address: "cuMemAddressFree"(u64, usize);
    physical_create: "cuMemCreate"(*mut u64, usize, *const AllocationProperties, u64);
    physical_release: "cuMemRelease"(u64);
    granularity: "cuMemGetAllocationGranularity"(*mut usize, *const AllocationProperties, c_int);
    map: "cuMemMap"(u64, usize, usize, u64, u64);
    unmap: "cuMemUnmap"(u64, usize);
    access: "cuMemSetAccess"(u64, usize, *const Access, usize);
    host_register: "cuMemHostRegister_v2"(Handle, usize, c_uint);
    host_unregister: "cuMemHostUnregister"(Handle);
    host_device: "cuMemHostGetDevicePointer_v2"(*mut u64, Handle, c_uint);
    host_allocate: "cuMemHostAlloc"(*mut Handle, usize, c_uint);
    host_free: "cuMemFreeHost"(Handle);
    upload: "cuMemcpyHtoD_v2"(u64, *const c_void, usize);
    upload_async: "cuMemcpyHtoDAsync_v2"(u64, *const c_void, usize, Handle);
    stream_create: "cuStreamCreate"(*mut Handle, c_uint);
    stream_sync: "cuStreamSynchronize"(Handle);
    stream_destroy: "cuStreamDestroy_v2"(Handle);
    event_create: "cuEventCreate"(*mut Handle, c_uint);
    event_record: "cuEventRecord"(Handle, Handle);
    event_sync: "cuEventSynchronize"(Handle);
    event_destroy: "cuEventDestroy_v2"(Handle);
    download: "cuMemcpyDtoH_v2"(*mut c_void, u64, usize);
    zero: "cuMemsetD8_v2"(u64, u8, usize);
    module_load: "cuModuleLoadData"(*mut Handle, *const c_void);
    module_global: "cuModuleGetGlobal_v2"(*mut u64, *mut usize, Handle, *const c_char);
    module_unload: "cuModuleUnload"(Handle);
    function: "cuModuleGetFunction"(*mut Handle, Handle, *const c_char);
    parameter_info: "cuFuncGetParamInfo"(Handle, usize, *mut usize, *mut usize);
    function_attribute: "cuFuncSetAttribute"(Handle, c_int, c_int);
    function_get_attribute: "cuFuncGetAttribute"(*mut c_int, c_int, Handle);
    function_load: "cuFuncLoad"(Handle);
    occupancy: "cuOccupancyMaxActiveBlocksPerMultiprocessor"(*mut c_int, Handle, c_int, usize);
    graph_upload: "cuGraphUpload"(Handle, Handle);
    graph_create: "cuGraphCreate"(*mut Handle, c_uint);
    graph_destroy: "cuGraphDestroy"(Handle);
    graph_kernel: "cuGraphAddKernelNode_v2"(*mut Handle, Handle, *const Handle, usize, *const KernelParams);
    graph_kernel_params: "cuGraphKernelNodeSetParams_v2"(Handle, *const KernelParams);
    graph_memset: "cuGraphAddMemsetNode"(*mut Handle, Handle, *const Handle, usize, *const MemsetParams, Handle);
    graph_copy: "cuGraphAddMemcpyNode"(*mut Handle, Handle, *const Handle, usize, *const CopyParams, Handle);
    graph_empty: "cuGraphAddEmptyNode"(*mut Handle, Handle, *const Handle, usize);
    graph_event_wait: "cuGraphAddEventWaitNode"(*mut Handle, Handle, *const Handle, usize, Handle);
    graph_attribute: "cuGraphKernelNodeSetAttribute"(Handle, c_int, *const c_void);
    graph_edges: "cuGraphAddDependencies_v2"(Handle, *const Handle, *const Handle, *const [u8; 8], usize);
    graph_instantiate: "cuGraphInstantiateWithFlags"(*mut Handle, Handle, u64);
    executable_destroy: "cuGraphExecDestroy"(Handle);
    tensor_map_replace: "cuTensorMapReplaceAddress"(*mut TensorMap, *mut c_void);
    error_name: "cuGetErrorName"(c_int, *mut *const c_char);
    error_string: "cuGetErrorString"(c_int, *mut *const c_char);
}

impl Driver {
    pub fn check(&self, operation: &str, status: c_int) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let text = |function: unsafe extern "C" fn(c_int, *mut *const c_char) -> c_int| {
            let mut value = std::ptr::null();
            // SAFETY: CUDA owns the returned string, which is copied before return.
            unsafe {
                if function(status, &mut value) == 0 && !value.is_null() {
                    CStr::from_ptr(value).to_string_lossy().into_owned()
                } else {
                    "unknown CUDA error".to_owned()
                }
            }
        };
        Err(ExecutionError::Driver(CudaError {
            operation: operation.into(),
            code: status,
            name: text(self.error_name),
            description: text(self.error_string),
        })
        .into())
    }
}
