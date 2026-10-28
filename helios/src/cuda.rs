use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::Arc;

use libloading::Library;

use crate::{CudaError, Error, ExecutionError, Failure, Result};

type Handle = *mut c_void;
type GetContext = unsafe extern "C" fn(*mut Handle) -> c_int;
type SetContext = unsafe extern "C" fn(Handle) -> c_int;
type GetDevice = unsafe extern "C" fn(*mut c_int) -> c_int;
type StreamCreate = unsafe extern "C" fn(*mut Handle, c_uint) -> c_int;
type StreamOperation = unsafe extern "C" fn(Handle) -> c_int;
type GraphLaunch = unsafe extern "C" fn(Handle, Handle) -> c_int;
type ErrorText = unsafe extern "C" fn(c_int, *mut *const c_char) -> c_int;
type HostAlloc = unsafe extern "C" fn(*mut Handle, usize, c_uint) -> c_int;
type HostFree = unsafe extern "C" fn(Handle) -> c_int;
type Upload = unsafe extern "C" fn(u64, *const c_void, usize, Handle) -> c_int;
type Download = unsafe extern "C" fn(Handle, u64, usize, Handle) -> c_int;
type Download2D = unsafe extern "C" fn(*const Copy2D, Handle) -> c_int;
type EventCreate = unsafe extern "C" fn(*mut Handle, c_uint) -> c_int;
type EventRecord = unsafe extern "C" fn(Handle, Handle) -> c_int;
type EventElapsed = unsafe extern "C" fn(*mut f32, Handle, Handle) -> c_int;
type StreamWait = unsafe extern "C" fn(Handle, Handle, c_uint) -> c_int;
type WaitValue = unsafe extern "C" fn(Handle, u64, u32, c_uint) -> c_int;
type HostPointer = unsafe extern "C" fn(*mut u64, Handle, c_uint) -> c_int;
type PointerAttribute = unsafe extern "C" fn(*mut c_void, c_int, u64) -> c_int;
type StreamCreatePriority = unsafe extern "C" fn(*mut Handle, c_uint, c_int) -> c_int;
type PriorityRange = unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int;
type HostRegister = unsafe extern "C" fn(Handle, usize, c_uint) -> c_int;
type HostUnregister = unsafe extern "C" fn(Handle) -> c_int;
#[cfg(feature = "runtime-package")]
type DeviceUuid = unsafe extern "C" fn(*mut [u8; 16], c_int) -> c_int;
type DeviceAttribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;
type AddressReserve = unsafe extern "C" fn(*mut u64, usize, usize, u64, u64) -> c_int;
type AddressFree = unsafe extern "C" fn(u64, usize) -> c_int;
type MemoryCreate =
    unsafe extern "C" fn(*mut u64, usize, *const AllocationProperties, u64) -> c_int;
type MemoryRelease = unsafe extern "C" fn(u64) -> c_int;
type MemoryMap = unsafe extern "C" fn(u64, usize, usize, u64, u64) -> c_int;
type MemoryUnmap = unsafe extern "C" fn(u64, usize) -> c_int;
type MemoryAccess = unsafe extern "C" fn(u64, usize, *const AccessDescriptor, usize) -> c_int;
type Granularity = unsafe extern "C" fn(*mut usize, *const AllocationProperties, c_int) -> c_int;
type MemcpyBatch = unsafe extern "C" fn(
    *const u64,
    *const u64,
    *const usize,
    usize,
    *const CopyAttributes,
    *const usize,
    usize,
    Handle,
) -> c_int;

/// `CUmemcpyAttributes`: source access order, source and destination location
/// hints as `CUmemLocation {type, id}`, and flags.
#[repr(C)]
struct CopyAttributes {
    source_order: c_int,
    source_location: [c_int; 2],
    destination_location: [c_int; 2],
    flags: c_uint,
}

/// `CUDA_MEMCPY2D`.
#[repr(C)]
struct Copy2D {
    source_x: usize,
    source_y: usize,
    source_type: c_uint,
    source_host: *const c_void,
    source_device: u64,
    source_array: Handle,
    source_pitch: usize,
    destination_x: usize,
    destination_y: usize,
    destination_type: c_uint,
    destination_host: Handle,
    destination_device: u64,
    destination_array: Handle,
    destination_pitch: usize,
    width: usize,
    height: usize,
}

/// `CUmemLocation`: a location type and, for devices and host NUMA nodes, its id.
#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryLocation {
    kind: c_int,
    id: c_int,
}

/// `CUmemAllocationProp`.
#[repr(C)]
struct AllocationProperties {
    kind: c_int,
    handle_types: c_int,
    location: MemoryLocation,
    windows_metadata: Handle,
    flags: [u8; 8],
}

/// `CUmemAccessDesc`.
#[repr(C)]
struct AccessDescriptor {
    location: MemoryLocation,
    flags: c_int,
}

/// `CU_MEMCPY_SRC_ACCESS_ORDER_STREAM`: the source is read in stream order.
const SOURCE_ORDER_STREAM: c_int = 1;
/// `CU_MEMHOSTREGISTER_PORTABLE`: the registration is visible to every context.
const HOST_REGISTER_PORTABLE: c_uint = 0x1;
/// `CU_DEVICE_ATTRIBUTE_INTEGRATED`.
const ATTRIBUTE_INTEGRATED: c_int = 18;
/// `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID`: the host NUMA node closest to the device, or -1.
const ATTRIBUTE_HOST_NUMA_ID: c_int = 134;
/// `CU_MEM_ALLOCATION_TYPE_PINNED`.
const ALLOCATION_PINNED: c_int = 1;
/// `CU_MEM_LOCATION_TYPE_DEVICE`.
const LOCATION_DEVICE: c_int = 1;
/// `CU_MEM_LOCATION_TYPE_HOST_NUMA`: host memory on the NUMA node named by the id.
const LOCATION_HOST_NUMA: c_int = 3;
/// `CU_MEM_ACCESS_FLAGS_PROT_READWRITE`.
const ACCESS_READ_WRITE: c_int = 3;
/// `CU_MEM_ALLOC_GRANULARITY_MINIMUM`.
const GRANULARITY_MINIMUM: c_int = 0;
/// `CU_STREAM_NON_BLOCKING`: the stream never synchronizes with the legacy default stream.
const STREAM_NON_BLOCKING: c_uint = 0x1;
/// `CU_EVENT_BLOCKING_SYNC`: a thread waiting on the event sleeps instead of spinning.
const EVENT_BLOCKING_SYNC: c_uint = 0x1;
/// `CU_EVENT_DISABLE_TIMING`: the event records no timestamp.
const EVENT_DISABLE_TIMING: c_uint = 0x2;

/// Most bytes one batched copy may carry.
pub const BATCH_BYTES: usize = 16 << 20;
/// Most entries one batched copy may carry.
pub const BATCH_ENTRIES: usize = 4096;

struct Driver {
    _library: Library,
    get_context: GetContext,
    set_context: SetContext,
    get_device: GetDevice,
    stream_create: StreamCreate,
    stream_synchronize: StreamOperation,
    stream_destroy: StreamOperation,
    graph_launch: GraphLaunch,
    error_name: ErrorText,
    error_string: ErrorText,
    host_alloc: HostAlloc,
    host_free: HostFree,
    upload: Upload,
    download: Download,
    download_2d: Download2D,
    event_create: EventCreate,
    event_record: EventRecord,
    event_elapsed: EventElapsed,
    event_query: StreamOperation,
    event_destroy: StreamOperation,
    stream_wait: StreamWait,
    wait_value: WaitValue,
    host_pointer: HostPointer,
    pointer_attribute: PointerAttribute,
    stream_create_priority: StreamCreatePriority,
    priority_range: PriorityRange,
    host_register: HostRegister,
    host_unregister: HostUnregister,
    device_attribute: DeviceAttribute,
    #[cfg(feature = "runtime-package")]
    device_uuid: DeviceUuid,
    event_synchronize: StreamOperation,
    memcpy_batch: MemcpyBatch,
    address_reserve: AddressReserve,
    address_free: AddressFree,
    memory_create: MemoryCreate,
    memory_release: MemoryRelease,
    memory_map: MemoryMap,
    memory_unmap: MemoryUnmap,
    memory_access: MemoryAccess,
    granularity: Granularity,
}

impl Driver {
    fn load() -> Result<Arc<Self>> {
        // SAFETY: The CUDA driver exports these ABI signatures. The retained
        // Library keeps every resolved function pointer valid for this Driver.
        unsafe {
            let library = crate::os::cuda_library().map_err(|e| {
                ExecutionError::State(Failure::caused_by(
                    format!("cannot load NVIDIA CUDA driver: {e}"),
                    e,
                ))
            })?;
            macro_rules! symbol {
                ($name:literal) => {
                    *library.get(concat!($name, "\0").as_bytes()).map_err(|e| {
                        ExecutionError::State(Failure::caused_by(
                            format!("cannot load CUDA entrypoint {}: {e}", $name),
                            e,
                        ))
                    })?
                };
            }
            Ok(Arc::new(Self {
                get_context: symbol!("cuCtxGetCurrent"),
                set_context: symbol!("cuCtxSetCurrent"),
                get_device: symbol!("cuCtxGetDevice"),
                stream_create: symbol!("cuStreamCreate"),
                stream_synchronize: symbol!("cuStreamSynchronize"),
                stream_destroy: symbol!("cuStreamDestroy_v2"),
                graph_launch: symbol!("cuGraphLaunch"),
                error_name: symbol!("cuGetErrorName"),
                error_string: symbol!("cuGetErrorString"),
                host_alloc: symbol!("cuMemHostAlloc"),
                host_free: symbol!("cuMemFreeHost"),
                upload: symbol!("cuMemcpyHtoDAsync_v2"),
                download: symbol!("cuMemcpyDtoHAsync_v2"),
                download_2d: symbol!("cuMemcpy2DAsync_v2"),
                event_create: symbol!("cuEventCreate"),
                event_record: symbol!("cuEventRecord"),
                event_elapsed: symbol!("cuEventElapsedTime"),
                event_query: symbol!("cuEventQuery"),
                event_destroy: symbol!("cuEventDestroy_v2"),
                stream_wait: symbol!("cuStreamWaitEvent"),
                wait_value: symbol!("cuStreamWaitValue32_v2"),
                host_pointer: symbol!("cuMemHostGetDevicePointer_v2"),
                pointer_attribute: symbol!("cuPointerGetAttribute"),
                stream_create_priority: symbol!("cuStreamCreateWithPriority"),
                priority_range: symbol!("cuCtxGetStreamPriorityRange"),
                host_register: symbol!("cuMemHostRegister_v2"),
                host_unregister: symbol!("cuMemHostUnregister"),
                device_attribute: symbol!("cuDeviceGetAttribute"),
                #[cfg(feature = "runtime-package")]
                device_uuid: symbol!("cuDeviceGetUuid"),
                event_synchronize: symbol!("cuEventSynchronize"),
                memcpy_batch: symbol!("cuMemcpyBatchAsync_v2"),
                address_reserve: symbol!("cuMemAddressReserve"),
                address_free: symbol!("cuMemAddressFree"),
                memory_create: symbol!("cuMemCreate"),
                memory_release: symbol!("cuMemRelease"),
                memory_map: symbol!("cuMemMap"),
                memory_unmap: symbol!("cuMemUnmap"),
                memory_access: symbol!("cuMemSetAccess"),
                granularity: symbol!("cuMemGetAllocationGranularity"),
                _library: library,
            }))
        }
    }

    fn check(&self, operation: &str, status: c_int) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let describe = |function: ErrorText| {
            let mut text = std::ptr::null();
            // SAFETY: Driver retains the CUDA library and its error-text functions.
            // CUDA owns the returned NUL-terminated string; it is copied immediately.
            unsafe {
                if function(status, &mut text) == 0 && !text.is_null() {
                    CStr::from_ptr(text).to_string_lossy().into_owned()
                } else {
                    "unknown CUDA error".to_owned()
                }
            }
        };
        Err(ExecutionError::Driver(CudaError {
            operation: operation.to_owned(),
            code: status,
            name: describe(self.error_name),
            description: describe(self.error_string),
        })
        .into())
    }

    fn current(&self) -> Result<(usize, i32)> {
        let mut context = std::ptr::null_mut();
        // SAFETY: The driver owns the ABI and the output pointer is valid for the call.
        self.check("cuCtxGetCurrent", unsafe {
            (self.get_context)(&mut context)
        })?;
        if context.is_null() {
            return Err(Error::execution(
                ExecutionError::State,
                "no current CUDA context; initialize the model's device before registration",
            ));
        }
        let mut device = -1;
        // SAFETY: A current context was verified and device is a valid output pointer.
        self.check("cuCtxGetDevice", unsafe { (self.get_device)(&mut device) })?;
        Ok((context as usize, device))
    }
}

pub fn current_context() -> Result<(usize, i32)> {
    Driver::load()?.current()
}

/// Whether the device owning `context` shares physical memory with the host.
pub(crate) fn device_integrated(context: usize) -> Result<bool> {
    let driver = Driver::load()?;
    // SAFETY: The context handle was produced by the driver and outlives this call.
    driver.check("cuCtxSetCurrent", unsafe {
        (driver.set_context)(context as Handle)
    })?;
    let (_, device) = driver.current()?;
    let mut value = 0;
    // SAFETY: `value` is a valid output pointer and `device` came from the current context.
    driver.check("cuDeviceGetAttribute", unsafe {
        (driver.device_attribute)(&mut value, ATTRIBUTE_INTEGRATED, device)
    })?;
    Ok(value != 0)
}

/// The UUID of the device with this ordinal, which names the GPU across restarts.
#[cfg(feature = "runtime-package")]
pub(crate) fn device_uuid(device: i32) -> Result<[u8; 16]> {
    let driver = Driver::load()?;
    let mut uuid = [0u8; 16];
    // SAFETY: `uuid` is a valid 16-byte output for the call.
    driver.check("cuDeviceGetUuid", unsafe {
        (driver.device_uuid)(&mut uuid, device)
    })?;
    Ok(uuid)
}

/// Page-locked registration of process memory, released on drop.
pub(crate) struct HostRegistration {
    driver: Arc<Driver>,
    address: usize,
    device_address: u64,
    nbytes: usize,
}

impl HostRegistration {
    pub(crate) fn address(&self) -> usize {
        self.address
    }
    pub(crate) fn nbytes(&self) -> usize {
        self.nbytes
    }
    pub(crate) fn device_address(&self) -> u64 {
        self.device_address
    }
}

impl Drop for HostRegistration {
    fn drop(&mut self) {
        // SAFETY: The registration was made by `register_host` on this range.
        unsafe {
            (self.driver.host_unregister)(self.address as Handle);
        }
    }
}

/// Register `nbytes` at `address`, which must be page aligned, as portable
/// page-locked memory with `context` current. The range stays mapped and
/// unchanged in size until the registration is dropped.
pub(crate) unsafe fn register_host(
    context: usize,
    address: usize,
    nbytes: usize,
) -> Result<HostRegistration> {
    let page = crate::os::page_size();
    if address == 0 || !address.is_multiple_of(page) || nbytes == 0 {
        return Err(Error::execution(
            ExecutionError::State,
            format!(
                "host registration requires a nonempty range at a {page}-byte aligned address, \
                 got {nbytes} bytes at {address:#x}"
            ),
        ));
    }
    let driver = Driver::load()?;
    // SAFETY: The context handle was produced by the driver, and the caller
    // guarantees the range stays mapped until the registration is dropped.
    unsafe {
        driver.check("cuCtxSetCurrent", (driver.set_context)(context as Handle))?;
        driver.check(
            "cuMemHostRegister_v2",
            (driver.host_register)(address as Handle, nbytes, HOST_REGISTER_PORTABLE),
        )?;
    }
    // Own the registration before querying its device alias so a failed query
    // unregisters the range. Registered memory can have disjoint CPU and GPU
    // addresses, including on Windows WDDM.
    let mut registration = HostRegistration {
        driver,
        address,
        device_address: 0,
        nbytes,
    };
    // SAFETY: The context is still current, the range is registered and live,
    // and device_address is a valid output pointer.
    unsafe {
        registration.driver.check(
            "cuMemHostGetDevicePointer_v2",
            (registration.driver.host_pointer)(
                &mut registration.device_address,
                address as Handle,
                0,
            ),
        )?;
    }
    Ok(registration)
}

/// Pinned host memory in one reserved address range, split into parts that
/// each only one device and the host can access.
///
/// Registering host memory maps it into every device's page tables, which costs
/// each device about 2 MiB per GiB however little of it the device copies. A
/// part created through the virtual memory management API and opened to one
/// device costs only that device.
pub(crate) struct DeviceHostParts {
    driver: Arc<Driver>,
    address: u64,
    nbytes: usize,
    /// Every part created so far, in request order.
    parts: Vec<HostPart>,
}

struct HostPart {
    offset: usize,
    nbytes: usize,
    handle: u64,
    mapped: bool,
}

impl DeviceHostParts {
    pub(crate) fn address(&self) -> usize {
        self.address as usize
    }
    /// Bytes of the reserved range, including each part's rounding.
    pub(crate) fn nbytes(&self) -> usize {
        self.nbytes
    }
    /// Address of the part made for request `index`.
    pub(crate) fn part_address(&self, index: usize) -> usize {
        self.address as usize + self.parts[index].offset
    }
}

impl Drop for DeviceHostParts {
    fn drop(&mut self) {
        // SAFETY: The mappings, handles, and reservation were made by
        // `map_device_host_parts`; each is released once, mappings first.
        unsafe {
            for part in &self.parts {
                if part.mapped {
                    (self.driver.memory_unmap)(self.address + part.offset as u64, part.nbytes);
                }
                (self.driver.memory_release)(part.handle);
            }
            (self.driver.address_free)(self.address, self.nbytes);
        }
    }
}

/// Reserve one range and back it with one pinned host part per `(context,
/// device, bytes)` request, in request order. Each part is created on the host
/// NUMA node closest to its device, rounded up to the allocation granularity,
/// and made accessible to that device and the host only. Each request's
/// context is made current on the calling thread while its part is planned
/// and created.
pub(crate) fn map_device_host_parts(requests: &[(usize, i32, usize)]) -> Result<DeviceHostParts> {
    let invalid = |message: String| Error::execution(ExecutionError::State, message);
    if requests.is_empty() || requests.iter().any(|(_, _, bytes)| *bytes == 0) {
        return Err(invalid(
            "device host memory requires at least one nonempty part".into(),
        ));
    }
    let driver = Driver::load()?;
    let mut planned = Vec::with_capacity(requests.len());
    let (mut total, mut alignment) = (0usize, 1usize);
    for &(context, device, bytes) in requests {
        let mut node = -1;
        let mut granularity = 0;
        let mut properties = AllocationProperties {
            kind: ALLOCATION_PINNED,
            handle_types: 0,
            location: MemoryLocation {
                kind: LOCATION_HOST_NUMA,
                id: 0,
            },
            windows_metadata: std::ptr::null_mut(),
            flags: [0; 8],
        };
        // SAFETY: The context handle was produced by the driver and outlives
        // this call; every output pointer is valid for its call.
        unsafe {
            driver.check("cuCtxSetCurrent", (driver.set_context)(context as Handle))?;
            driver.check(
                "cuDeviceGetAttribute",
                (driver.device_attribute)(&mut node, ATTRIBUTE_HOST_NUMA_ID, device),
            )?;
            // A host without NUMA reports -1 and takes node 0.
            properties.location.id = node.max(0);
            driver.check(
                "cuMemGetAllocationGranularity",
                (driver.granularity)(&mut granularity, &properties, GRANULARITY_MINIMUM),
            )?;
        }
        let granularity = granularity.max(1);
        let offset = total.checked_next_multiple_of(granularity);
        let rounded = bytes.checked_next_multiple_of(granularity);
        let Some((offset, rounded, end)) = offset
            .zip(rounded)
            .and_then(|(offset, rounded)| Some((offset, rounded, offset.checked_add(rounded)?)))
        else {
            return Err(invalid(format!(
                "{bytes} bytes of device host memory exceed the address space"
            )));
        };
        total = end;
        alignment = alignment.max(granularity);
        planned.push((context, device, properties, offset, rounded));
    }
    let mut address = 0;
    // SAFETY: `address` is a valid output pointer; the range is released on drop.
    driver.check("cuMemAddressReserve", unsafe {
        (driver.address_reserve)(&mut address, total, alignment, 0, 0)
    })?;
    let mut memory = DeviceHostParts {
        driver: driver.clone(),
        address,
        nbytes: total,
        parts: Vec::with_capacity(planned.len()),
    };
    for (context, device, properties, offset, nbytes) in planned {
        let mut handle = 0;
        let access = [
            AccessDescriptor {
                location: MemoryLocation {
                    kind: LOCATION_DEVICE,
                    id: device,
                },
                flags: ACCESS_READ_WRITE,
            },
            AccessDescriptor {
                location: MemoryLocation {
                    kind: LOCATION_HOST_NUMA,
                    id: properties.location.id,
                },
                flags: ACCESS_READ_WRITE,
            },
        ];
        let start = address + offset as u64;
        // SAFETY: The context outlives this call, the part lies inside the
        // reservation, and `memory` records each step so its drop undoes it.
        unsafe {
            driver.check("cuCtxSetCurrent", (driver.set_context)(context as Handle))?;
            driver.check(
                "cuMemCreate",
                (driver.memory_create)(&mut handle, nbytes, &properties, 0),
            )?;
            memory.parts.push(HostPart {
                offset,
                nbytes,
                handle,
                mapped: false,
            });
            driver.check("cuMemMap", (driver.memory_map)(start, nbytes, 0, handle, 0))?;
            if let Some(part) = memory.parts.last_mut() {
                part.mapped = true;
            }
            driver.check(
                "cuMemSetAccess",
                (driver.memory_access)(start, nbytes, access.as_ptr(), access.len()),
            )?;
        }
    }
    Ok(memory)
}

pub(crate) struct Stream {
    driver: Arc<Driver>,
    handle: usize,
}

/// Reject a batch whose arrays disagree in length, that has more than
/// [`BATCH_ENTRIES`] entries, that carries more than [`BATCH_BYTES`], or that
/// contains an empty entry.
fn check_batch(destinations: &[u64], sources: &[u64], sizes: &[usize]) -> Result<()> {
    let invalid = |message: String| Error::execution(ExecutionError::State, message);
    if destinations.len() != sources.len() || sources.len() != sizes.len() {
        return Err(invalid(format!(
            "batched copy arrays disagree: {} destinations, {} sources, {} sizes",
            destinations.len(),
            sources.len(),
            sizes.len()
        )));
    }
    if sizes.len() > BATCH_ENTRIES {
        return Err(invalid(format!(
            "batched copy has {} entries; at most {BATCH_ENTRIES} are allowed",
            sizes.len()
        )));
    }
    let total = sizes
        .iter()
        .try_fold(0usize, |total, size| {
            (*size > 0).then_some(())?;
            total.checked_add(*size)
        })
        .ok_or_else(|| invalid("batched copy contains an empty entry or overflows".into()))?;
    if total > BATCH_BYTES {
        return Err(invalid(format!(
            "batched copy carries {total} bytes; at most {BATCH_BYTES} are allowed"
        )));
    }
    Ok(())
}

impl Stream {
    /// Make `context` current on the calling thread and confirm its device.
    fn bind(context: usize, expected_device: i32) -> Result<Arc<Driver>> {
        let driver = Driver::load()?;
        // SAFETY: The context handle was produced by the driver and outlives this call.
        driver.check("cuCtxSetCurrent", unsafe {
            (driver.set_context)(context as Handle)
        })?;
        let (_, device) = driver.current()?;
        if device != expected_device {
            return Err(Error::execution(
                ExecutionError::State,
                format!(
                    "CUDA context belongs to device {device}, expected device {expected_device}"
                ),
            ));
        }
        Ok(driver)
    }

    pub(crate) fn open(context: usize, expected_device: i32) -> Result<Self> {
        let driver = Self::bind(context, expected_device)?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: The bound context is current and `handle` is a valid output pointer.
        driver.check("cuStreamCreate", unsafe {
            (driver.stream_create)(&mut handle, STREAM_NON_BLOCKING)
        })?;
        Ok(Self {
            driver,
            handle: handle as usize,
        })
    }

    /// A stream at the lowest priority the context supports, for copies that
    /// must yield to model work.
    pub(crate) fn open_lowest_priority(context: usize, expected_device: i32) -> Result<Self> {
        let driver = Self::bind(context, expected_device)?;
        let (mut least, mut greatest) = (0, 0);
        // SAFETY: The bound context is current and both outputs are valid pointers.
        driver.check("cuCtxGetStreamPriorityRange", unsafe {
            (driver.priority_range)(&mut least, &mut greatest)
        })?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: As above; `least` is within the range the driver reported.
        driver.check("cuStreamCreateWithPriority", unsafe {
            (driver.stream_create_priority)(&mut handle, STREAM_NON_BLOCKING, least)
        })?;
        Ok(Self {
            driver,
            handle: handle as usize,
        })
    }

    /// Issue one batched copy of `destinations[i] <- sources[i]` for `sizes[i]`
    /// bytes each, in stream order on both sides. Every address is a device
    /// address, including the device alias of registered host memory. CPU
    /// addresses of registrations are not necessarily GPU-accessible. The batch
    /// carries at most [`BATCH_BYTES`] and [`BATCH_ENTRIES`]; larger batches and
    /// empty entries are rejected. An empty batch issues nothing.
    ///
    /// # Safety
    /// Every range stays valid and, for host sources, unchanged until the
    /// stream passes the copy.
    pub(crate) unsafe fn copy_batch(
        &self,
        destinations: &[u64],
        sources: &[u64],
        sizes: &[usize],
    ) -> Result<()> {
        check_batch(destinations, sources, sizes)?;
        if sizes.is_empty() {
            return Ok(());
        }
        let attributes = CopyAttributes {
            source_order: SOURCE_ORDER_STREAM,
            source_location: [0; 2],
            destination_location: [0; 2],
            flags: 0,
        };
        let first_entry = [0usize];
        // SAFETY: The arrays have equal validated lengths, the single attribute
        // applies from entry 0, and the caller guarantees every range's lifetime.
        self.driver.check("cuMemcpyBatchAsync_v2", unsafe {
            (self.driver.memcpy_batch)(
                destinations.as_ptr(),
                sources.as_ptr(),
                sizes.as_ptr(),
                sizes.len(),
                &attributes,
                first_entry.as_ptr(),
                1,
                self.handle as Handle,
            )
        })
    }

    pub(crate) fn launch(&self, graph: usize) -> Result<()> {
        self.driver.check("cuGraphLaunch", unsafe {
            (self.driver.graph_launch)(graph as Handle, self.handle as Handle)
        })
    }

    pub(crate) fn host_buffer(&self, nbytes: usize) -> Result<PinnedHost> {
        self.allocate_host(nbytes, 0)
    }

    pub(crate) fn mapped_host_buffer(&self, nbytes: usize) -> Result<PinnedHost> {
        self.allocate_host(nbytes, 3)
    }

    fn allocate_host(&self, nbytes: usize, flags: u32) -> Result<PinnedHost> {
        let mut handle = std::ptr::null_mut();
        self.driver.check("cuMemHostAlloc", unsafe {
            (self.driver.host_alloc)(&mut handle, nbytes, flags)
        })?;
        unsafe {
            std::ptr::write_bytes(handle.cast::<u8>(), 0, nbytes);
        }
        Ok(PinnedHost {
            driver: self.driver.clone(),
            handle: handle as usize,
            nbytes,
        })
    }

    /// An untimed event whose waits spin, observing completion with the least latency.
    pub(crate) fn event(&self) -> Result<Event> {
        self.event_with_timing(false)
    }
    pub(crate) fn event_with_timing(&self, timing: bool) -> Result<Event> {
        self.create_event(if timing { 0 } else { EVENT_DISABLE_TIMING })
    }

    /// An untimed event whose [`Event::synchronize`] sleeps the calling thread
    /// until the work completes: for offload threads that wait on long copies
    /// and must not occupy a core while doing so.
    pub(crate) fn blocking_event(&self) -> Result<Event> {
        self.create_event(EVENT_BLOCKING_SYNC | EVENT_DISABLE_TIMING)
    }

    fn create_event(&self, flags: c_uint) -> Result<Event> {
        let mut handle = std::ptr::null_mut();
        // SAFETY: The driver owns the ABI and `handle` is a valid output pointer.
        self.driver.check("cuEventCreate", unsafe {
            (self.driver.event_create)(&mut handle, flags)
        })?;
        Ok(Event {
            driver: self.driver.clone(),
            handle: handle as usize,
        })
    }

    pub(crate) fn record(&self, event: &Event) -> Result<()> {
        unsafe { self.record_borrowed(event.handle) }
    }

    /// The event belongs to this context and outlives all queued work.
    pub(crate) unsafe fn record_borrowed(&self, event: usize) -> Result<()> {
        self.driver.check("cuEventRecord", unsafe {
            (self.driver.event_record)(event as Handle, self.handle as Handle)
        })
    }

    /// The mapped flag remains allocated until this stream drains.
    pub(crate) unsafe fn wait_value(&self, flag: usize, generation: u32) -> Result<()> {
        self.driver.check("cuStreamWaitValue32_v2", unsafe {
            (self.driver.wait_value)(self.handle as Handle, flag as u64, generation, 1)
        })
    }

    pub(crate) fn mapped_address(&self, host: &PinnedHost) -> Result<usize> {
        let mut address = 0;
        self.driver.check("cuMemHostGetDevicePointer", unsafe {
            (self.driver.host_pointer)(&mut address, host.handle as Handle, 0)
        })?;
        Ok(address as usize)
    }

    /// Resolve a mapped host allocation's CPU address before a native reader
    /// polls a graph-produced mailbox. Ordinary device storage is not readable
    /// through a host pointer, even when its virtual address is nonzero.
    pub(crate) fn host_address(&self, address: usize, nbytes: usize) -> Result<usize> {
        self.mapped_host_address(address, nbytes)?
            .ok_or_else(|| "embedding request mailbox must use mapped host storage".into())
    }

    /// Resolve CPU-readable mapped storage, or identify an ordinary device range.
    /// Callers must observe the producing stream's completion before reading it.
    pub(crate) fn mapped_host_address(&self, address: usize, nbytes: usize) -> Result<Option<usize>> {
        let last = address
            .checked_add(nbytes.checked_sub(1).ok_or("empty mapped storage range")?)
            .ok_or("mapped storage address overflow")?;
        let mut host_start = 0;
        for pointer in [address, last] {
            let mut kind: c_int = 0;
            self.driver
                .check("cuPointerGetAttribute(MEMORY_TYPE)", unsafe {
                    (self.driver.pointer_attribute)(
                        (&mut kind as *mut c_int).cast(),
                        2,
                        pointer as u64,
                    )
                })?;
            if kind != 1 {
                // CU_MEMORYTYPE_HOST
                if pointer == address && kind == 2 {
                    // CU_MEMORYTYPE_DEVICE
                    return Ok(None);
                }
                return Err("mapped storage range crosses memory types".into());
            }
            let mut host: Handle = std::ptr::null_mut();
            self.driver
                .check("cuPointerGetAttribute(HOST_POINTER)", unsafe {
                    (self.driver.pointer_attribute)(
                        (&mut host as *mut Handle).cast(),
                        4,
                        pointer as u64,
                    )
                })?;
            if pointer == address {
                host_start = host as usize;
            } else if host_start.checked_add(nbytes - 1) != Some(host as usize) {
                return Err("mapped storage is not one contiguous host range".into());
            }
        }
        if host_start == 0 {
            return Err("mapped storage lacks a host address".into());
        }
        Ok(Some(host_start))
    }

    pub(crate) fn wait(&self, event: &Event) -> Result<()> {
        self.driver.check("cuStreamWaitEvent", unsafe {
            (self.driver.stream_wait)(self.handle as Handle, event.handle as Handle, 0)
        })
    }

    /// The pinned source remains unchanged until this stream passes the transfer.
    pub(crate) unsafe fn upload(
        &self,
        device: usize,
        host: &PinnedHost,
        offset: usize,
        nbytes: usize,
    ) -> Result<()> {
        self.driver.check("cuMemcpyHtoDAsync_v2", unsafe {
            (self.driver.upload)(
                device as u64,
                (host.handle + offset) as *const c_void,
                nbytes,
                self.handle as Handle,
            )
        })
    }

    /// The pinned destination is not read or reused until this stream passes the transfer.
    pub(crate) unsafe fn download(
        &self,
        host: &PinnedHost,
        device: usize,
        nbytes: usize,
    ) -> Result<()> {
        self.driver.check("cuMemcpyDtoHAsync_v2", unsafe {
            (self.driver.download)(
                host.handle as Handle,
                device as u64,
                nbytes,
                self.handle as Handle,
            )
        })
    }

    /// The source and pinned destination remain owned until this stream completes.
    pub(crate) unsafe fn download_2d(
        &self,
        host: &PinnedHost,
        device: usize,
        source_pitch: usize,
        width: usize,
        rows: usize,
    ) -> Result<()> {
        let copy = Copy2D {
            source_x: 0,
            source_y: 0,
            source_type: 2,
            source_host: std::ptr::null(),
            source_device: device as u64,
            source_array: std::ptr::null_mut(),
            source_pitch,
            destination_x: 0,
            destination_y: 0,
            destination_type: 1,
            destination_host: host.handle as Handle,
            destination_device: 0,
            destination_array: std::ptr::null_mut(),
            destination_pitch: width,
            width,
            height: rows,
        };
        self.driver.check("cuMemcpy2DAsync_v2", unsafe {
            (self.driver.download_2d)(&copy, self.handle as Handle)
        })
    }

    pub(crate) fn synchronize(&self) -> Result<()> {
        self.driver.check("cuStreamSynchronize", unsafe {
            (self.driver.stream_synchronize)(self.handle as Handle)
        })
    }

    pub(crate) fn destroy(mut self) -> Result<()> {
        let handle = std::mem::take(&mut self.handle);
        self.driver.check("cuStreamDestroy_v2", unsafe {
            (self.driver.stream_destroy)(handle as Handle)
        })
    }
}

pub(crate) struct Event {
    driver: Arc<Driver>,
    handle: usize,
}

impl Event {
    pub(crate) fn elapsed_ns(&self, start: &Event) -> Result<u64> {
        let mut millis = 0.0;
        self.driver.check("cuEventElapsedTime", unsafe {
            (self.driver.event_elapsed)(&mut millis, start.handle as Handle, self.handle as Handle)
        })?;
        if !millis.is_finite() || millis <= 0.0 {
            return Err(Error::execution(
                ExecutionError::Completion,
                "invalid speculative CUDA event duration",
            ));
        }
        Ok((f64::from(millis) * 1_000_000.0).round() as u64)
    }

    /// Block the calling thread until the event's work completes. The thread
    /// spins unless the event came from [`Stream::blocking_event`].
    pub(crate) fn synchronize(&self) -> Result<()> {
        self.driver.check("cuEventSynchronize", unsafe {
            (self.driver.event_synchronize)(self.handle as Handle)
        })
    }

    pub(crate) fn completed(&self) -> Result<bool> {
        let status = unsafe { (self.driver.event_query)(self.handle as Handle) };
        if status == 600 {
            return Ok(false);
        }
        self.driver.check("cuEventQuery", status)?;
        Ok(true)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            (self.driver.event_destroy)(self.handle as Handle);
        }
    }
}

pub(crate) struct PinnedHost {
    driver: Arc<Driver>,
    handle: usize,
    nbytes: usize,
}

impl PinnedHost {
    pub(crate) fn address(&self) -> usize {
        self.handle
    }
    pub(crate) fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.handle as *const u8, self.nbytes) }
    }
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.handle as *mut u8, self.nbytes) }
    }
}

impl Drop for PinnedHost {
    fn drop(&mut self) {
        unsafe {
            (self.driver.host_free)(self.handle as Handle);
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if self.handle != 0 {
            // Explicit synchronization reports errors; Drop only releases the stream handle.
            unsafe {
                (self.driver.stream_destroy)(self.handle as Handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_attributes_match_the_driver_layout() {
        assert_eq!(std::mem::size_of::<CopyAttributes>(), 24);
        assert_eq!(std::mem::align_of::<CopyAttributes>(), 4);
    }

    #[test]
    fn batch_bounds_reject_mismatch_oversize_and_empty_entries() {
        assert!(check_batch(&[], &[], &[]).is_ok());
        assert!(check_batch(&[1, 2], &[3, 4], &[8, 8]).is_ok());
        assert!(check_batch(&[1], &[3, 4], &[8]).is_err());
        assert!(check_batch(&[1, 2], &[3, 4], &[8]).is_err());
        assert!(check_batch(&[1], &[3], &[0]).is_err());
        let full = vec![1u64; BATCH_ENTRIES];
        assert!(check_batch(&full, &full, &vec![1; BATCH_ENTRIES]).is_ok());
        let over = vec![1u64; BATCH_ENTRIES + 1];
        assert!(check_batch(&over, &over, &vec![1; BATCH_ENTRIES + 1]).is_err());
        assert!(check_batch(&[1], &[2], &[BATCH_BYTES]).is_ok());
        assert!(check_batch(&[1], &[2], &[BATCH_BYTES + 1]).is_err());
        assert!(check_batch(&[1, 2], &[3, 4], &[BATCH_BYTES, 1]).is_err());
        assert!(check_batch(&[1, 2], &[3, 4], &[usize::MAX, 2]).is_err());
    }

    /// Scattered entries: device offsets ascend with odd gaps while host
    /// offsets descend, so neither side is contiguous or aligned.
    fn scattered_entries(capacity: usize) -> Vec<(usize, usize, usize)> {
        let sizes: Vec<usize> = (0..220)
            .map(|i| match i % 50 {
                0 => (1 << 20) + 4096 + i * 64,
                n if n % 3 == 0 => 100 + i,
                n if n % 3 == 1 => 4096 + 7 * i,
                _ => 2000 + i * 3,
            })
            .collect();
        let mut device = Vec::with_capacity(sizes.len());
        let mut cursor = 0;
        for size in &sizes {
            device.push(cursor);
            cursor += size + 37;
        }
        assert!(cursor <= capacity);
        let mut host = vec![0; sizes.len()];
        cursor = 0;
        for (index, size) in sizes.iter().enumerate().rev() {
            host[index] = cursor;
            cursor += size + 11;
        }
        assert!(cursor <= capacity);
        sizes
            .into_iter()
            .zip(device)
            .zip(host)
            .map(|((size, device), host)| (device, host, size))
            .collect()
    }

    fn pattern(offset: usize) -> u8 {
        ((offset * 7) ^ (offset >> 9)) as u8
    }

    #[cfg(unix)]
    fn thread_cpu_time() -> std::time::Duration {
        let mut time = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `time` is a valid output pointer for the calling thread's clock.
        assert_eq!(
            unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) },
            0
        );
        std::time::Duration::new(time.tv_sec as u64, time.tv_nsec as u32)
    }

    /// Read a patterned file with `O_DIRECT` into the registered mapping and
    /// compare. Returns `false` when the directory refuses direct I/O.
    #[cfg(unix)]
    fn direct_read_matches(host: *mut u8, capacity: usize) -> bool {
        use std::os::unix::ffi::OsStrExt;
        const FILE_BYTES: usize = 4 << 20;
        assert!(FILE_BYTES <= capacity);
        let directory = std::env::var_os("FLARE_OFFLOAD_TEST_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join(format!("flare-cuda-direct-{}.bin", std::process::id()));
        let content: Vec<u8> = (0..FILE_BYTES).map(|i| pattern(i * 3)).collect();
        std::fs::write(&path, &content).unwrap();
        let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        // SAFETY: The path is NUL terminated and the mapping covers FILE_BYTES.
        let fd = unsafe {
            libc::open(
                c_path.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECT | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            let error = std::io::Error::last_os_error();
            std::fs::remove_file(&path).unwrap();
            if error.raw_os_error() == Some(libc::EINVAL) {
                eprintln!(
                    "skipping direct read: {} refuses O_DIRECT",
                    directory.display()
                );
                return false;
            }
            panic!("open {} with O_DIRECT: {error}", path.display());
        }
        unsafe {
            std::ptr::write_bytes(host, 0xA5, capacity);
            let mut done = 0;
            while done < FILE_BYTES {
                let read = libc::pread(fd, host.add(done).cast(), FILE_BYTES - done, done as i64);
                assert!(read > 0, "pread: {}", std::io::Error::last_os_error());
                done += read as usize;
            }
            libc::close(fd);
        }
        std::fs::remove_file(&path).unwrap();
        // SAFETY: The mapping stays valid for the comparison.
        let read = unsafe { std::slice::from_raw_parts(host, capacity) };
        assert!(read[..FILE_BYTES] == content[..]);
        assert!(read[FILE_BYTES..].iter().all(|byte| *byte == 0xA5));
        true
    }

    #[cfg(feature = "runtime-package")]
    #[test]
    #[ignore = "requires a CUDA device; FLARE_OFFLOAD_TEST_DIR names a directory supporting direct I/O"]
    fn batched_copies_round_trip_through_registered_memory() {
        use crate::runtime_package::driver::Driver as Loader;
        const BYTES: usize = 8 << 20;
        let loader = Loader::load().unwrap();
        let mut context = std::ptr::null_mut();
        let (mut a, mut b) = (0u64, 0u64);
        // SAFETY: Driver signatures match cuda.h and every output pointer is valid.
        unsafe {
            loader.check("cuInit", (loader.init)(0)).unwrap();
            let mut device = 0;
            loader
                .check("cuDeviceGet", (loader.device)(&mut device, 0))
                .unwrap();
            loader
                .check(
                    "cuCtxCreate_v2",
                    (loader.context_create)(&mut context, 0, device),
                )
                .unwrap();
            loader
                .check("cuMemAlloc_v2", (loader.allocate)(&mut a, BYTES))
                .unwrap();
            loader
                .check("cuMemAlloc_v2", (loader.allocate)(&mut b, BYTES))
                .unwrap();
        }
        let stream = Stream::open_lowest_priority(context as usize, 0).unwrap();
        let event = stream.blocking_event().unwrap();
        let host = crate::os::map_host(BYTES).unwrap() as *mut u8;
        assert_eq!(host as usize % 4096, 0);
        let registration =
            unsafe { register_host(context as usize, host as usize, BYTES) }.unwrap();
        assert_eq!(registration.address(), host as usize);
        assert_eq!(registration.nbytes(), BYTES);
        let device_host = registration.device_address();

        let image_a: Vec<u8> = (0..BYTES).map(pattern).collect();
        let entries = scattered_entries(BYTES);
        assert!(entries.len() >= 200);
        assert!(entries.iter().any(|(_, _, size)| *size < 4096));
        assert!(entries.iter().any(|(_, _, size)| *size > 1 << 20));
        unsafe {
            loader
                .check(
                    "cuMemcpyHtoD_v2",
                    (loader.upload)(a, image_a.as_ptr().cast(), BYTES),
                )
                .unwrap();
            loader
                .check("cuMemsetD8_v2", (loader.zero)(b, 0, BYTES))
                .unwrap();
        }

        let sizes: Vec<usize> = entries.iter().map(|entry| entry.2).collect();
        let to_host: Vec<u64> = entries.iter().map(|e| device_host + e.1 as u64).collect();
        let from_a: Vec<u64> = entries.iter().map(|e| a + e.0 as u64).collect();
        unsafe { stream.copy_batch(&to_host, &from_a, &sizes) }.unwrap();
        stream.record(&event).unwrap();
        event.synchronize().unwrap();
        let mut expected_host = vec![0u8; BYTES];
        for (device, host_offset, size) in &entries {
            expected_host[*host_offset..host_offset + size]
                .copy_from_slice(&image_a[*device..device + size]);
        }
        // SAFETY: The mapping stays valid and no copy is in flight.
        let host_bytes = unsafe { std::slice::from_raw_parts(host, BYTES) };
        assert!(
            host_bytes == &expected_host[..],
            "device to host batch differs"
        );

        let poison = vec![0xEEu8; BYTES];
        unsafe {
            loader
                .check(
                    "cuMemcpyHtoD_v2",
                    (loader.upload)(a, poison.as_ptr().cast(), BYTES),
                )
                .unwrap();
        }
        let to_b: Vec<u64> = entries.iter().map(|e| b + e.0 as u64).collect();
        let from_host: Vec<u64> = entries.iter().map(|e| device_host + e.1 as u64).collect();
        unsafe { stream.copy_batch(&to_b, &from_host, &sizes) }.unwrap();
        stream.record(&event).unwrap();
        event.synchronize().unwrap();
        let mut image_b = vec![0u8; BYTES];
        unsafe {
            loader
                .check(
                    "cuMemcpyDtoH_v2",
                    (loader.download)(image_b.as_mut_ptr().cast(), b, BYTES),
                )
                .unwrap();
        }
        let mut expected_b = vec![0u8; BYTES];
        for (device, _, size) in &entries {
            expected_b[*device..device + size].copy_from_slice(&image_a[*device..device + size]);
        }
        assert!(image_b == expected_b, "host to device batch differs");
        assert!(unsafe { stream.copy_batch(&[b], &[a], &[BATCH_BYTES + 1]) }.is_err());
        stream.synchronize().unwrap();

        // Waiting on the blocking event must sleep rather than spin: queue
        // enough host-to-device traffic to outlast its issue and compare the
        // thread's CPU time against the wall time of the wait.
        for _ in 0..600 {
            unsafe { stream.copy_batch(&[b], &[device_host], &[BYTES]) }.unwrap();
        }
        stream.record(&event).unwrap();
        #[cfg(unix)]
        let (wall, cpu) = (std::time::Instant::now(), thread_cpu_time());
        event.synchronize().unwrap();
        #[cfg(unix)]
        {
            let (wall, cpu) = (wall.elapsed(), thread_cpu_time() - cpu);
            eprintln!("blocking event wait: {wall:?} wall, {cpu:?} thread CPU");
            if wall >= std::time::Duration::from_millis(40) {
                assert!(
                    cpu * 4 < wall,
                    "the blocking event wait spun: {cpu:?} CPU over {wall:?}"
                );
            }
        }

        #[cfg(unix)]
        let direct = direct_read_matches(host, BYTES);

        drop(registration);
        unsafe {
            crate::os::unmap_host(host as usize, BYTES);
        }
        drop(event);
        drop(stream);
        unsafe {
            loader.check("cuMemFree_v2", (loader.free)(a)).unwrap();
            loader.check("cuMemFree_v2", (loader.free)(b)).unwrap();
            loader
                .check("cuCtxDestroy_v2", (loader.context_destroy)(context))
                .unwrap();
        }
        #[cfg(unix)]
        assert!(
            direct || std::env::var_os("FLARE_OFFLOAD_TEST_DIR").is_none(),
            "the configured test directory refused direct I/O"
        );
    }
}
