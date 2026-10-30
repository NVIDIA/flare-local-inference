//! Native callbacks used by the preparation process's PyTorch memory pool.

use std::{
    collections::BTreeMap,
    ffi::{CString, c_char, c_void},
    ptr::null_mut,
    sync::Mutex,
};

use crate::runtime_package::driver::{Access, AllocationProperties, Driver, Location};
use crate::{Error, ExecutionError, Result};

fn invalid(message: impl Into<String>) -> Error {
    Error::execution(ExecutionError::Manifest, message)
}

// PyTorch's CPU allocator is mimalloc, which places its 1 GiB arenas at random addresses
// in 2-6 TiB; a peer rank allocates CPU tensors before reserving these ranges.
#[cfg(target_os = "linux")]
const BASE: u64 = 0x80000000000;
#[cfg(target_os = "windows")]
const BASE: u64 = 0x40000000000;
const EXTENT: usize = 1 << 40;
// Keep small Linux exports within the Windows WDDM address range as well.
const HOST_BASE: u64 = 0xc0000000000;

fn host_base(device_base: u64, world: u32) -> u64 {
    if device_base + u64::from(world) * EXTENT as u64 <= HOST_BASE {
        HOST_BASE
    } else {
        0x200000000000
    }
}

#[derive(Default)]
struct RetainedBacking {
    active: bool,
    aborted: bool,
    finished: bool,
    reference_bytes: usize,
    handles: Vec<(u64, usize)>,
}

impl RetainedBacking {
    fn begin(&mut self) -> Result<()> {
        if self.active || self.aborted || !self.handles.is_empty() {
            return Err(invalid(
                "physical retention scope is already active or abandoned",
            ));
        }
        self.active = true;
        self.finished = false;
        Ok(())
    }

    fn defer_free(&self) -> Result<bool> {
        if self.aborted || (!self.active && !self.finished) {
            return Err(invalid(
                "transient free outside a healthy physical retention scope",
            ));
        }
        Ok(self.active)
    }

    fn may_reuse(&self) -> Result<bool> {
        if !self.active || self.aborted {
            return Err(invalid(
                "transient allocation outside an active physical retention scope",
            ));
        }
        Ok(true)
    }

    fn retain(&mut self, size: usize, acquire: impl FnOnce() -> Result<u64>) -> Result<()> {
        if !self.active || self.aborted {
            return Err(invalid(
                "transient retention outside an active physical retention scope",
            ));
        }
        let reference_bytes = self
            .reference_bytes
            .checked_add(size)
            .ok_or_else(|| invalid("retained physical byte count overflow"))?;
        self.handles
            .try_reserve(1)
            .map_err(|e| invalid(format!("retained handle storage: {e}")))?;
        let handle = acquire()?;
        self.handles.push((handle, size));
        self.reference_bytes = reference_bytes;
        Ok(())
    }

    fn check_finishable(&self) -> Result<()> {
        if !self.active || self.aborted {
            return Err(invalid("physical retention scope is not active"));
        }
        Ok(())
    }

    fn finish(&mut self, mut release: impl FnMut(u64) -> Result<()>) -> Result<()> {
        if !self.active || self.aborted {
            return Err(invalid("physical retention scope is not active"));
        }
        while let Some(&(handle, size)) = self.handles.last() {
            release(handle)?;
            self.handles.pop();
            self.reference_bytes -= size;
        }
        self.active = false;
        self.finished = true;
        Ok(())
    }

    fn abort_on_free_error(&mut self, error: Error) -> Option<Error> {
        if self.aborted {
            return None;
        }
        self.abort();
        Some(invalid(format!(
            "{error}; retained reference bytes={}, handles={}",
            self.reference_bytes,
            self.handles.len()
        )))
    }

    fn abort(&mut self) {
        self.active = false;
        self.aborted = true;
    }
}

struct RetainedMapping {
    address: u64,
    size: usize,
}

#[derive(Default)]
struct RetainedMappingPool {
    mappings: BTreeMap<usize, Vec<RetainedMapping>>,
}

impl RetainedMappingPool {
    fn take(&mut self, size: usize) -> Option<RetainedMapping> {
        let mapping_size = self.mappings.range(size..).next().map(|(&size, _)| size)?;
        let mappings = self.mappings.get_mut(&mapping_size)?;
        let mapping = mappings.pop()?;
        if mappings.is_empty() {
            self.mappings.remove(&mapping_size);
        }
        Some(mapping)
    }

    fn insert(&mut self, mapping: RetainedMapping) {
        self.mappings.entry(mapping.size).or_default().push(mapping);
    }

    fn peek(&self) -> Option<&RetainedMapping> {
        self.mappings.values().flatten().next()
    }

    fn pop(&mut self) -> Option<RetainedMapping> {
        let size = self.mappings.keys().next().copied()?;
        let mappings = self.mappings.get_mut(&size)?;
        let mapping = mappings.pop()?;
        if mappings.is_empty() {
            self.mappings.remove(&size);
        }
        Some(mapping)
    }

    fn len(&self) -> usize {
        self.mappings.values().map(Vec::len).sum()
    }

    fn bytes(&self) -> usize {
        self.mappings
            .iter()
            .map(|(&size, handles)| size.saturating_mul(handles.len()))
            .sum()
    }
}

struct Arena {
    driver: Driver,
    device: i32,
    granularity: usize,
    gpudirect: bool,
    host_base: u64,
    free: Vec<(u64, usize)>,
    mappings: BTreeMap<u64, usize>,
    retained: RetainedBacking,
    retained_mappings: RetainedMappingPool,
}
static ARENA: Mutex<Option<Arena>> = Mutex::new(None);
static ERROR: Mutex<Option<CString>> = Mutex::new(None);

pub(crate) struct MemoryInventory {
    pub device: i32,
    pub free: usize,
    pub total: usize,
    pub mappings: Vec<(u64, usize)>,
}

pub(crate) fn memory_inventory() -> Result<MemoryInventory> {
    let arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
    let a = arena
        .as_ref()
        .ok_or_else(|| invalid("native package allocator is uninitialized"))?;
    let (_, device) = crate::current_context()?;
    if device != a.device {
        return Err(invalid(
            "memory inventory requires the package rank's CUDA context",
        ));
    }
    let (mut free, mut total) = (0, 0);
    // SAFETY: Preparation retains the current context and both outputs are valid.
    unsafe {
        a.driver
            .check("cuCtxSynchronize", (a.driver.context_sync)())?;
        a.driver.check(
            "cuMemGetInfo_v2",
            (a.driver.memory_info)(&mut free, &mut total),
        )?;
    }
    Ok(MemoryInventory {
        device,
        free,
        total,
        mappings: a
            .mappings
            .iter()
            .map(|(&address, &size)| (address, size))
            .collect(),
    })
}

fn properties(device: i32, gpudirect: bool) -> AllocationProperties {
    AllocationProperties {
        kind: 1,
        handles: i32::from(cfg!(target_os = "linux")),
        location: Location { kind: 1, device },
        windows: null_mut(),
        // cuFile registration requires gpuDirectRDMACapable on VMM allocations.
        flags: [0, u8::from(gpudirect), 0, 0, 0, 0, 0, 0],
    }
}

fn failure(error: impl ToString) {
    *ERROR.lock().unwrap_or_else(|e| e.into_inner()) =
        Some(CString::new(error.to_string().replace('\0', " ")).unwrap());
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_initialize(rank: u32, world: u32, device: i32) -> i32 {
    let initialize = || -> Result<()> {
        let mut arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
        if arena.is_some() {
            return Err(invalid("native package allocator is already initialized"));
        }
        if world == 0 || rank >= world || device < 0 {
            return Err(invalid("invalid package rank or device"));
        }
        let host_base = if cfg!(target_os = "windows") {
            HOST_BASE
        } else {
            host_base(BASE, world)
        };
        if u64::from(world) * EXTENT as u64 > BASE.abs_diff(host_base) {
            return Err(invalid(
                "package host reservations would overlap device reservations",
            ));
        }
        let driver = Driver::load()?;
        let (mut address, mut granularity, mut gpudirect) = (0, 0, 0);
        unsafe {
            driver.check(
                "cuDeviceGetAttribute(GPU_DIRECT_RDMA_SUPPORTED)",
                (driver.device_attribute)(&mut gpudirect, 116, device),
            )?;
            driver.check(
                "cuMemGetAllocationGranularity",
                (driver.granularity)(&mut granularity, &properties(device, gpudirect != 0), 0),
            )?;
            let mut reserved = Vec::new();
            for peer in 0..world {
                let base = BASE + u64::from(peer) * EXTENT as u64;
                let status = (driver.reserve)(&mut address, EXTENT, 0, base, 0);
                if status != 0 || address != base {
                    if status == 0 {
                        (driver.release_address)(address, EXTENT);
                    }
                    for address in reserved {
                        (driver.release_address)(address, EXTENT);
                    }
                    driver.check("cuMemAddressReserve", status)?;
                    return Err(invalid(format!(
                        "native package device address range {base:#x}+{EXTENT:#x} is unavailable; CUDA returned {address:#x}"
                    )));
                }
                reserved.push(address);
            }
        }
        *arena = Some(Arena {
            driver,
            device,
            granularity,
            gpudirect: gpudirect != 0,
            host_base: host_base + u64::from(rank) * EXTENT as u64,
            free: vec![(BASE + u64::from(rank) * EXTENT as u64, EXTENT / 2)],
            mappings: BTreeMap::new(),
            retained: RetainedBacking::default(),
            retained_mappings: RetainedMappingPool::default(),
        });
        Ok(())
    };
    match initialize() {
        Ok(()) => 0,
        Err(error) => {
            failure(error);
            -1
        }
    }
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_allocate(
    size: usize,
    device: i32,
    _stream: *mut c_void,
) -> *mut c_void {
    package_allocate(size, device, false)
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_transient_allocate(
    size: usize,
    device: i32,
    _stream: *mut c_void,
) -> *mut c_void {
    package_allocate(size, device, true)
}

fn package_allocate(size: usize, device: i32, transient: bool) -> *mut c_void {
    let allocate = || -> Result<u64> {
        let mut guard = ARENA.lock().unwrap_or_else(|e| e.into_inner());
        let a = guard
            .as_mut()
            .ok_or_else(|| invalid("native package allocator is uninitialized"))?;
        if device != a.device {
            return Err(invalid("allocation device differs from package rank"));
        }
        let size = size
            .checked_add(a.granularity - 1)
            .ok_or_else(|| invalid("allocation size overflow"))?
            / a.granularity
            * a.granularity;
        if transient && a.retained.may_reuse()? {
            if let Some(mapping) = a.retained_mappings.take(size) {
                a.mappings.insert(mapping.address, mapping.size);
                return Ok(mapping.address);
            }
        }
        let index = a
            .free
            .iter()
            .position(|(_, n)| *n >= size)
            .ok_or_else(|| invalid("native package address range exhausted"))?;
        let (address, available) = a.free[index];
        let mut physical = 0;
        unsafe {
            a.driver.check(
                "cuMemCreate",
                (a.driver.physical_create)(
                    &mut physical,
                    size,
                    &properties(device, a.gpudirect),
                    0,
                ),
            )?;
            let mapped = (a.driver.map)(address, size, 0, physical, 0);
            (a.driver.physical_release)(physical);
            a.driver.check("cuMemMap", mapped)?;
            let access = Access {
                location: Location { kind: 1, device },
                flags: 3,
            };
            let status = (a.driver.access)(address, size, &access, 1);
            if status != 0 {
                (a.driver.unmap)(address, size);
            }
            a.driver.check("cuMemSetAccess", status)?;
        }
        if size == available {
            a.free.remove(index);
        } else {
            a.free[index] = (address + size as u64, available - size);
        }
        a.mappings.insert(address, size);
        Ok(address)
    };
    match allocate() {
        Ok(address) => address as *mut c_void,
        // PyTorch may release cached segments and retry a physical allocation.
        Err(Error::Execution(ExecutionError::Driver(error)))
            if error.operation == "cuMemCreate" && error.code == 2 =>
        {
            null_mut()
        }
        Err(error) => {
            failure(error);
            null_mut()
        }
    }
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_free(
    pointer: *mut c_void,
    _size: usize,
    _device: i32,
    _stream: *mut c_void,
) {
    package_free(pointer, false);
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_transient_free(
    pointer: *mut c_void,
    _size: usize,
    _device: i32,
    _stream: *mut c_void,
) {
    package_free(pointer, true);
}

fn package_free(pointer: *mut c_void, transient: bool) {
    let release = || -> Result<()> {
        let mut guard = ARENA.lock().unwrap_or_else(|e| e.into_inner());
        let a = guard
            .as_mut()
            .ok_or_else(|| invalid("native package allocator is uninitialized"))?;
        let address = pointer as u64;
        let size = *a
            .mappings
            .get(&address)
            .ok_or_else(|| invalid("unknown native package allocation"))?;
        let retain = transient && a.retained.defer_free()?;
        if retain {
            let driver = &a.driver;
            a.retained.retain(size, || {
                let mut handle = 0;
                unsafe {
                    driver.check(
                        "cuMemRetainAllocationHandle",
                        (driver.physical_retain)(&mut handle, pointer),
                    )?;
                }
                Ok(handle)
            })?;
        }
        a.mappings.remove(&address);
        if retain {
            a.retained_mappings
                .insert(RetainedMapping { address, size });
            return Ok(());
        }
        unsafe {
            a.driver
                .check("cuMemUnmap", (a.driver.unmap)(address, size))?;
        }
        release_free_range(&mut a.free, address, size);
        Ok(())
    };
    if let Err(error) = release() {
        if transient {
            let mut arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(a) = arena.as_mut() {
                if let Some(error) = a.retained.abort_on_free_error(error) {
                    failure(error);
                }
                return;
            }
        }
        failure(error);
    }
}

fn retention_operation(operation: impl FnOnce(&mut Arena) -> Result<()>) -> i32 {
    let mut guard = ARENA.lock().unwrap_or_else(|e| e.into_inner());
    let Some(a) = guard.as_mut() else {
        failure(invalid("native package allocator is uninitialized"));
        return -1;
    };
    match operation(a) {
        Ok(()) => 0,
        Err(error) => {
            failure(invalid(format!(
                "{error}; retained reference bytes={}, handles={}",
                a.retained.reference_bytes,
                a.retained.handles.len()
            )));
            -1
        }
    }
}

fn release_free_range(free: &mut Vec<(u64, usize)>, address: u64, size: usize) {
    free.push((address, size));
    free.sort_unstable();
    let mut merged: Vec<(u64, usize)> = Vec::new();
    for (address, size) in free.drain(..) {
        if let Some(previous) = merged.last_mut().filter(|p| p.0 + p.1 as u64 == address) {
            previous.1 += size;
        } else {
            merged.push((address, size));
        }
    }
    *free = merged;
}

fn finish_retention(
    retained: &mut RetainedBacking,
    retained_mappings: &mut RetainedMappingPool,
    free: &mut Vec<(u64, usize)>,
    mut unmap: impl FnMut(u64, usize) -> Result<()>,
    mut release: impl FnMut(u64) -> Result<()>,
) -> Result<()> {
    retained.check_finishable()?;
    while let Some(mapping) = retained_mappings.peek() {
        let (address, size) = (mapping.address, mapping.size);
        unmap(address, size)?;
        retained_mappings.pop();
        release_free_range(free, address, size);
    }
    retained.finish(|handle| release(handle))
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_retention_begin() -> i32 {
    retention_operation(|a| a.retained.begin())
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_retention_finish() -> i32 {
    retention_operation(|a| {
        let driver = &a.driver;
        finish_retention(
            &mut a.retained,
            &mut a.retained_mappings,
            &mut a.free,
            |address, size| unsafe { driver.check("cuMemUnmap", (driver.unmap)(address, size)) },
            |handle| unsafe { driver.check("cuMemRelease", (driver.physical_release)(handle)) },
        )
    })
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_retention_abort() -> i32 {
    retention_operation(|a| {
        a.retained.abort();
        Ok(())
    })
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_snapshot() -> *mut c_char {
    let arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
    let Some(a) = arena.as_ref() else {
        return null_mut();
    };
    let value = serde_json::json!({
        "granularity": a.granularity, "mappings": a.mappings, "host_base": a.host_base,
        "retained_reference_bytes": a.retained.reference_bytes,
        "retained_handles": a.retained.handles.len(),
        "retained_reusable_handles": a.retained_mappings.len(),
        "retained_reusable_bytes": a.retained_mappings.bytes(),
    });
    CString::new(value.to_string()).unwrap().into_raw()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn flare_package_string_free(pointer: *mut c_char) {
    if !pointer.is_null() {
        unsafe {
            drop(CString::from_raw(pointer));
        }
    }
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_error() -> *const c_char {
    ERROR
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or(std::ptr::null(), |s| s.as_ptr())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secondary_aborted_free_preserves_first_driver_error() {
        let mut retained = RetainedBacking::default();
        retained.begin().unwrap();
        retained.retain(4096, || Ok(7)).unwrap();
        let mut published = retained
            .abort_on_free_error(invalid("cuMemUnmap returned CUDA error 715"))
            .unwrap()
            .to_string();
        let first = published.clone();
        let secondary = retained.defer_free().unwrap_err();
        if let Some(error) = retained.abort_on_free_error(secondary) {
            published = error.to_string();
        }
        assert_eq!(published, first);
        assert!(published.contains("cuMemUnmap returned CUDA error 715"));
        assert!(published.contains("retained reference bytes=4096, handles=1"));
        assert_eq!(retained.handles, vec![(7, 4096)]);
    }

    #[test]
    fn physical_references_preserve_duplicates_and_release_once() {
        let mut retained = RetainedBacking::default();
        assert!(retained.retain(4, || Ok(7)).is_err());
        retained.begin().unwrap();
        assert!(retained.begin().is_err());
        retained.retain(4, || Ok(7)).unwrap();
        retained.retain(8, || Ok(7)).unwrap();
        assert_eq!(retained.reference_bytes, 12);
        let mut released = Vec::new();
        retained
            .finish(|handle| {
                released.push(handle);
                Ok(())
            })
            .unwrap();
        assert_eq!(released, vec![7, 7]);
        assert_eq!(retained.reference_bytes, 0);
        assert!(retained.handles.is_empty());
        assert!(retained.retain(4, || Ok(8)).is_err());
        assert!(!retained.defer_free().unwrap());
    }

    #[test]
    fn physical_reference_errors_preserve_accounting_and_abort_never_releases() {
        let mut retained = RetainedBacking::default();
        retained.begin().unwrap();
        assert!(
            retained
                .retain(4, || Err(invalid("retain failed")))
                .is_err()
        );
        assert_eq!(retained.reference_bytes, 0);
        retained.retain(4, || Ok(7)).unwrap();
        assert!(retained.finish(|_| Err(invalid("release failed"))).is_err());
        assert_eq!(retained.handles, vec![(7, 4)]);
        assert_eq!(retained.reference_bytes, 4);
        retained.abort();
        assert!(retained.defer_free().is_err());
        assert!(
            retained
                .finish(|_| panic!("must not call CUDA after abort"))
                .is_err()
        );
        assert!(retained.begin().is_err());
        assert_eq!(retained.handles, vec![(7, 4)]);
    }

    #[test]
    fn retained_byte_overflow_precedes_driver_call() {
        let mut retained = RetainedBacking::default();
        retained.begin().unwrap();
        retained.reference_bytes = usize::MAX;
        assert!(
            retained
                .retain(1, || panic!("overflow must precede retain"))
                .is_err()
        );
    }

    #[test]
    fn retention_finish_keeps_mapping_when_unmap_fails() {
        let mut retained = RetainedBacking::default();
        retained.begin().unwrap();
        retained.retain(4096, || Ok(7)).unwrap();
        let mut retained_mappings = RetainedMappingPool::default();
        retained_mappings.insert(RetainedMapping {
            address: 0x100,
            size: 4096,
        });
        let mut free = Vec::new();

        let result = finish_retention(
            &mut retained,
            &mut retained_mappings,
            &mut free,
            |_, _| Err(invalid("unmap failed")),
            |_| panic!("release must not run while unmap fails"),
        );

        assert!(result.is_err());
        assert_eq!(retained_mappings.len(), 1);
        assert_eq!(retained.handles, vec![(7, 4096)]);
        assert!(free.is_empty());
    }

    #[test]
    fn retained_mapping_pool_uses_smallest_fit() {
        let mut pool = RetainedMappingPool::default();
        pool.insert(RetainedMapping {
            address: 0x100,
            size: 4096,
        });
        pool.insert(RetainedMapping {
            address: 0x200,
            size: 8192,
        });
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.bytes(), 12288);
        let popped = pool.pop().unwrap();
        assert_eq!((popped.address, popped.size), (0x100, 4096));
        pool.insert(popped);
        let first = pool.take(2048).unwrap();
        assert_eq!((first.address, first.size), (0x100, 4096));
        let second = pool.take(4096).unwrap();
        assert_eq!((second.address, second.size), (0x200, 8192));
        assert_eq!(pool.len(), 0);
        assert_eq!(pool.bytes(), 0);
        assert!(pool.take(4096).is_none());
    }

    unsafe extern "C" fn abort_test_unmap(_address: u64, _size: usize) -> i32 {
        panic!("driver unmap called after abort");
    }

    unsafe extern "C" fn abort_test_release(_handle: u64) -> i32 {
        panic!("driver release called after abort");
    }

    #[test]
    fn retention_finish_after_abort_makes_no_driver_calls() {
        let Ok(mut driver) = Driver::load() else {
            return;
        };
        driver.unmap = abort_test_unmap;
        driver.physical_release = abort_test_release;
        let mut retained = RetainedBacking::default();
        retained.begin().unwrap();
        retained.retain(4096, || Ok(7)).unwrap();
        let mut retained_mappings = RetainedMappingPool::default();
        retained_mappings.insert(RetainedMapping {
            address: 0x100,
            size: 4096,
        });
        retained.abort();
        {
            let mut arena = ARENA.lock().unwrap_or_else(|error| error.into_inner());
            *arena = Some(Arena {
                driver,
                device: 0,
                granularity: 4096,
                gpudirect: false,
                host_base: 0,
                free: Vec::new(),
                mappings: BTreeMap::new(),
                retained,
                retained_mappings,
            });
        }
        assert_eq!(flare_package_retention_finish(), -1);
        let mut arena = ARENA.lock().unwrap_or_else(|error| error.into_inner());
        let allocator = arena.as_mut().unwrap();
        assert_eq!(allocator.retained_mappings.len(), 1);
        assert!(allocator.free.is_empty());
        assert_eq!(allocator.retained.handles, vec![(7, 4096)]);
        *arena = None;
        *ERROR.lock().unwrap_or_else(|error| error.into_inner()) = None;
    }

    #[test]
    fn small_linux_exports_keep_every_host_range_within_windows_addresses() {
        let device_base = 8 << 40;
        for world in 1..=4 {
            let host = host_base(device_base, world);
            assert_eq!(host, 12 << 40);
            assert!(device_base + u64::from(world) * EXTENT as u64 <= host);
            assert!(host + u64::from(world) * EXTENT as u64 <= 16 << 40);
        }
    }

    #[test]
    fn larger_linux_exports_keep_host_ranges_clear_of_all_device_ranks() {
        let device_base = 8 << 40;
        for world in 5..=24 {
            let host = host_base(device_base, world);
            assert_eq!(host, 32 << 40);
            assert!(device_base + u64::from(world) * EXTENT as u64 <= host);
        }
    }
}
