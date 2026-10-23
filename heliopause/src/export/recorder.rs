//! Preparation-scoped CUPTI recording. Callbacks copy owned data without CUDA calls.

use super::{ExportResult, invalid};
use libloading::Library;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, c_char, c_void};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::ThreadId;

#[repr(C)]
struct Resource {
    context: usize,
    handle: usize,
    descriptor: *const Module,
}
#[repr(C)]
struct Module {
    id: u32,
    size: usize,
    image: *const u8,
}
#[repr(C)]
struct Call {
    site: i32,
    name: *const c_char,
    params: *const usize,
    result: *const i32,
    symbol: *const c_char,
    context: usize,
    context_id: u32,
    correlation: *mut u64,
    correlation_id: u32,
}
#[derive(Clone, Debug)]
struct Image {
    context: usize,
    module_id: u32,
    cubin: Arc<[u8]>,
}
#[derive(Clone, Debug)]
struct Binding {
    cubin: Arc<[u8]>,
    context: usize,
    image: Option<Arc<Image>>,
    owner: Arc<()>,
}
impl Binding {
    fn new(cubin: Arc<[u8]>, context: usize) -> Self {
        Self {
            cubin,
            context,
            image: None,
            owner: Arc::new(()),
        }
    }
    fn materialized(image: &Arc<Image>) -> Self {
        Self {
            image: Some(image.clone()),
            ..Self::new(image.cubin.clone(), image.context)
        }
    }
    fn in_context(mut self, context: usize) -> Self {
        self.context = context;
        if self
            .image
            .as_ref()
            .is_some_and(|image| image.context != context)
        {
            self.image = None;
        }
        self
    }
}
#[derive(Clone, Debug)]
pub(crate) struct RecordedAllocation {
    pub address: u64,
    pub nbytes: u64,
    pub memory: String,
    pub context: Option<usize>,
}
#[derive(Default)]
struct State {
    images: BTreeMap<(usize, u32), Arc<Image>>,
    modules: BTreeMap<usize, Binding>,
    libraries: BTreeMap<usize, Binding>,
    functions: BTreeMap<usize, Arc<Image>>,
    kernels: BTreeMap<usize, Arc<Image>>,
    pending: HashMap<ThreadId, Vec<(String, Vec<Arc<Image>>)>>,
    allocations: BTreeMap<u64, RecordedAllocation>,
    error: Option<String>,
}
struct CallbackState {
    state: Mutex<State>,
    inflight: Mutex<usize>,
    drained: Condvar,
}
impl CallbackState {
    fn new() -> Self {
        Self {
            state: Mutex::new(State::default()),
            inflight: Mutex::new(0),
            drained: Condvar::new(),
        }
    }
}
fn query(name: &str) -> Option<(bool, usize)> {
    match name {
        "cuKernelGetAttribute" => Some((true, 2)),
        "cuFuncGetAttribute"
        | "cuOccupancyMaxPotentialBlockSize"
        | "cuOccupancyMaxPotentialBlockSizeWithFlags" => Some((false, 2)),
        "cuOccupancyMaxActiveBlocksPerMultiprocessor"
        | "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"
        | "cuOccupancyAvailableDynamicSMemPerBlock"
        | "cuOccupancyMaxPotentialClusterSize"
        | "cuOccupancyMaxActiveClusters" => Some((false, 1)),
        _ => None,
    }
}
fn materializer(name: &str) -> bool {
    query(name).is_some()
        || matches!(
            name,
            "cuModuleLoad"
                | "cuModuleLoadData"
                | "cuModuleLoadDataEx"
                | "cuModuleLoadFatBinary"
                | "cuModuleGetFunction"
                | "cuLibraryLoadData"
                | "cuLibraryLoadFromFile"
                | "cuLibraryGetModule"
                | "cuLibraryGetKernel"
                | "cuKernelGetFunction"
                | "cuKernelSetAttribute"
                | "cuFuncSetAttribute"
                | "cuFuncLoad"
                | "cuLaunchKernel"
                | "cuLaunchKernel_ptsz"
                | "cuLaunchKernelEx"
                | "cuLaunchKernelEx_ptsz"
                | "cuLaunchCooperativeKernel"
                | "cuLaunchCooperativeKernel_ptsz"
        )
}
impl State {
    fn drop_module(&mut self, context: usize, module_id: u32) {
        let image = self.images.remove(&(context, module_id));
        if let Some(image) = image {
            self.modules.retain(|_, binding| {
                !binding
                    .image
                    .as_ref()
                    .is_some_and(|i| Arc::ptr_eq(i, &image))
            });
            for binding in self.libraries.values_mut() {
                if binding
                    .image
                    .as_ref()
                    .is_some_and(|i| Arc::ptr_eq(i, &image))
                {
                    binding.image = None;
                }
            }
        }
        self.functions
            .retain(|_, i| i.context != context || i.module_id != module_id);
        self.kernels
            .retain(|_, i| i.context != context || i.module_id != module_id);
    }
    fn live(&self, image: &Arc<Image>) -> bool {
        self.images
            .get(&(image.context, image.module_id))
            .is_some_and(|i| Arc::ptr_eq(i, image))
    }
    fn bind_module(&mut self, handle: usize, image: &Arc<Image>) {
        if self.live(image) {
            let mut binding = Binding::materialized(image);
            if let Some(previous) = self.modules.get(&handle) {
                binding.owner = previous.owner.clone();
            }
            self.modules.insert(handle, binding);
        }
    }
    fn bind_library(&mut self, handle: usize, image: &Arc<Image>) {
        if self.live(image) {
            let mut binding = Binding::materialized(image);
            if let Some(previous) = self.libraries.get(&handle) {
                binding.owner = previous.owner.clone();
            }
            self.libraries.insert(handle, binding);
        }
    }
    fn drop_library(&mut self, handle: usize) {
        if let Some(library) = self.libraries.remove(&handle) {
            self.modules
                .retain(|_, binding| !Arc::ptr_eq(&binding.owner, &library.owner));
        }
    }
    fn bind(&mut self, kernel: bool, handle: usize, image: &Arc<Image>) {
        // Module unload can precede the exit of a materializing driver call.
        if self
            .images
            .get(&(image.context, image.module_id))
            .is_some_and(|i| Arc::ptr_eq(i, image))
        {
            if kernel {
                &mut self.kernels
            } else {
                &mut self.functions
            }
            .insert(handle, image.clone());
        }
    }
    unsafe fn resource(&mut self, callback: u32, data: *const c_void) -> ExportResult<()> {
        let resource = unsafe { &*data.cast::<Resource>() };
        if callback == 2 {
            let keys: Vec<_> = self
                .images
                .keys()
                .copied()
                .filter(|key| key.0 == resource.context)
                .collect();
            for (c, m) in keys {
                self.drop_module(c, m);
            }
            self.allocations
                .retain(|_, a| a.context != Some(resource.context));
            self.modules.retain(|_, i| i.context != resource.context);
            // Libraries can retain the same cubin in another CUDA context.
            for binding in self.libraries.values_mut() {
                if binding
                    .image
                    .as_ref()
                    .is_some_and(|i| i.context == resource.context)
                {
                    binding.image = None;
                }
            }
            return Ok(());
        }
        if !matches!(callback, 6 | 7) {
            return Ok(());
        }
        if resource.descriptor.is_null() {
            return Err(invalid("CUPTI returned a module without its descriptor"));
        }
        let module = unsafe { &*resource.descriptor };
        if callback == 7 {
            self.drop_module(resource.context, module.id);
            return Ok(());
        }
        if module.image.is_null() || module.size == 0 {
            return Err(invalid("CUPTI returned a module without its cubin"));
        }
        let image = Arc::new(Image {
            context: resource.context,
            module_id: module.id,
            cubin: Arc::from(unsafe { std::slice::from_raw_parts(module.image, module.size) }),
        });
        self.images
            .insert((image.context, image.module_id), image.clone());
        if let Some(stack) = self.pending.get_mut(&std::thread::current().id()) {
            if let Some((_, images)) = stack.last_mut() {
                images.push(image);
            }
        }
        Ok(())
    }
    unsafe fn driver(&mut self, call: &Call) -> ExportResult<()> {
        if call.name.is_null() {
            return Err(invalid("CUPTI returned a driver callback without a name"));
        }
        let name = unsafe { CStr::from_ptr(call.name) }.to_str()?;
        let thread = std::thread::current().id();
        let mut images = Vec::new();
        if materializer(name) {
            if call.site == 0 {
                self.pending
                    .entry(thread)
                    .or_default()
                    .push((name.into(), vec![]));
                return Ok(());
            }
            let stack = self
                .pending
                .get_mut(&thread)
                .ok_or_else(|| invalid("unbalanced CUDA driver callbacks"))?;
            let (entered, loaded) = stack
                .pop()
                .ok_or_else(|| invalid("unbalanced CUDA driver callbacks"))?;
            if entered != name {
                return Err(invalid("unbalanced CUDA driver callbacks"));
            }
            images = loaded;
            if stack.is_empty() {
                self.pending.remove(&thread);
            }
        }
        if call.site != 1 {
            return Ok(());
        }
        if call.params.is_null() || call.result.is_null() {
            return Err(invalid("incomplete CUDA driver callback"));
        }
        let value = |i| unsafe { *call.params.add(i) };
        let output = |i| -> ExportResult<usize> {
            let p = value(i) as *const usize;
            if p.is_null() {
                Err(invalid("null CUDA driver output"))
            } else {
                Ok(unsafe { *p })
            }
        };
        if images.len() == 1 {
            if let Some((kernel, index)) = query(name) {
                self.bind(kernel, value(index), &images[0]);
            }
        }
        if unsafe { *call.result } != 0 {
            return Ok(());
        }
        if images.len() == 1 {
            let image = &images[0];
            if name.starts_with("cuLaunchKernelEx") {
                self.bind(false, value(1), image);
            } else if name.starts_with("cuLaunchKernel")
                || name.starts_with("cuLaunchCooperativeKernel")
                || matches!(name, "cuFuncLoad" | "cuFuncSetAttribute")
            {
                self.bind(false, value(0), image);
            } else if name == "cuKernelSetAttribute" {
                self.bind(true, value(1), image);
            } else if name == "cuKernelGetFunction" {
                self.bind(false, output(0)?, image);
            } else if name == "cuModuleGetFunction" {
                self.bind_module(value(1), image);
            } else if matches!(name, "cuLibraryGetKernel" | "cuLibraryGetModule") {
                self.bind_library(value(1), image);
            }
        }
        match name {
            "cuModuleLoadData"
            | "cuModuleLoadDataEx"
            | "cuLibraryLoadData"
            | "cuModuleLoad"
            | "cuLibraryLoadFromFile" => {
                let handle = output(0)?;
                let cubin = if matches!(name, "cuModuleLoad" | "cuLibraryLoadFromFile") {
                    let path = unsafe { CStr::from_ptr(value(1) as *const c_char) }.to_str()?;
                    Some(std::fs::read(path)?)
                } else {
                    unsafe { elf_image(value(1) as *const u8) }?
                };
                let cubin = cubin
                    .filter(|i| i.starts_with(b"\x7fELF"))
                    .map(Arc::<[u8]>::from)
                    .or_else(|| {
                        if images.len() == 1 {
                            Some(images[0].cubin.clone())
                        } else {
                            None
                        }
                    });
                if let Some(cubin) = cubin {
                    let target = if name.starts_with("cuLibrary") {
                        &mut self.libraries
                    } else {
                        &mut self.modules
                    };
                    target.insert(handle, Binding::new(cubin, call.context));
                }
            }
            "cuLibraryGetModule" => {
                if images.iter().all(|image| self.live(image)) {
                    if let Some(image) = self.libraries.get(&value(1)).cloned() {
                        self.modules
                            .insert(output(0)?, image.in_context(call.context));
                    }
                }
            }
            "cuModuleUnload" => {
                self.modules.remove(&value(0));
            }
            "cuLibraryUnload" => {
                self.drop_library(value(0));
            }
            "cuMemAlloc_v2"
            | "cuMemAllocAsync"
            | "cuMemAllocAsync_ptsz"
            | "cuMemAllocFromPoolAsync"
            | "cuMemAllocFromPoolAsync_ptsz" => {
                let address = output(0)? as u64;
                self.allocations.insert(
                    address,
                    RecordedAllocation {
                        address,
                        nbytes: value(1) as u64,
                        memory: "device".into(),
                        context: Some(call.context),
                    },
                );
            }
            "cuMemFree_v2"
            | "cuMemFreeAsync"
            | "cuMemFreeAsync_ptsz"
            | "cuMemUnmap"
            | "cuMemFreeHost"
            | "cuMemHostUnregister" => {
                self.allocations.remove(&(value(0) as u64));
            }
            "cuMemMap" => {
                let address = value(0) as u64;
                self.allocations.insert(
                    address,
                    RecordedAllocation {
                        address,
                        nbytes: value(1) as u64,
                        memory: "device".into(),
                        context: Some(call.context),
                    },
                );
            }
            "cuMemHostAlloc" | "cuMemHostAlloc_v2" | "cuMemAllocHost_v2" => {
                let address = output(0)? as u64;
                self.allocations.insert(
                    address,
                    RecordedAllocation {
                        address,
                        nbytes: value(1) as u64,
                        memory: "mapped_host".into(),
                        context: None,
                    },
                );
            }
            "cuMemHostRegister" | "cuMemHostRegister_v2" => {
                let address = value(0) as u64;
                self.allocations.insert(
                    address,
                    RecordedAllocation {
                        address,
                        nbytes: value(1) as u64,
                        memory: "mapped_host".into(),
                        context: None,
                    },
                );
            }
            _ => {}
        }
        Ok(())
    }
}
unsafe fn elf_image(address: *const u8) -> ExportResult<Option<Vec<u8>>> {
    if address.is_null() {
        return Ok(None);
    }
    let header = unsafe { std::slice::from_raw_parts(address, 64) };
    if !header.starts_with(b"\x7fELF\x02\x01") {
        return Ok(None);
    }
    let u16at = |p| u16::from_le_bytes(header[p..p + 2].try_into().unwrap()) as usize;
    let u64at = |p| u64::from_le_bytes(header[p..p + 8].try_into().unwrap()) as usize;
    let offset = u64at(40);
    let count = u16at(60);
    if u16at(58) != 64 || count == 0 {
        return Err(invalid("unsupported cubin ELF section table"));
    }
    let mut end = offset
        .checked_add(count * 64)
        .ok_or_else(|| invalid("invalid cubin extent"))?
        .max(
            u64at(32)
                .checked_add(u16at(54) * u16at(56))
                .ok_or_else(|| invalid("invalid cubin extent"))?,
        );
    let sections = unsafe { std::slice::from_raw_parts(address.add(offset), count * 64) };
    for s in sections.chunks_exact(64) {
        if u32::from_le_bytes(s[4..8].try_into().unwrap()) != 8 {
            let start = u64::from_le_bytes(s[24..32].try_into().unwrap()) as usize;
            let size = u64::from_le_bytes(s[32..40].try_into().unwrap()) as usize;
            end = end.max(
                start
                    .checked_add(size)
                    .ok_or_else(|| invalid("invalid cubin extent"))?,
            );
        }
    }
    Ok(Some(
        unsafe { std::slice::from_raw_parts(address, end) }.to_vec(),
    ))
}
unsafe extern "C" fn callback(user: *mut c_void, domain: i32, id: u32, data: *const c_void) {
    let owner = unsafe { &*user.cast::<CallbackState>() };
    *owner.inflight.lock().unwrap_or_else(|e| e.into_inner()) += 1;
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut state = owner.state.lock().unwrap_or_else(|e| e.into_inner());
        let result = if data.is_null() {
            Err(invalid("CUPTI returned null callback data"))
        } else if domain == 1 {
            unsafe { state.driver(&*data.cast::<Call>()) }
        } else if domain == 3 {
            unsafe { state.resource(id, data) }
        } else {
            Ok(())
        };
        if let Err(error) = result {
            if state.error.is_none() {
                state.error = Some(error.to_string());
            }
        }
    }));
    if outcome.is_err() {
        owner.state.lock().unwrap_or_else(|e| e.into_inner()).error =
            Some("panic in CUPTI module callback".into());
    }
    let mut inflight = owner.inflight.lock().unwrap_or_else(|e| e.into_inner());
    *inflight -= 1;
    if *inflight == 0 {
        owner.drained.notify_all();
    }
}
type Subscribe = unsafe extern "C" fn(
    *mut usize,
    unsafe extern "C" fn(*mut c_void, i32, u32, *const c_void),
    *mut c_void,
) -> i32;
type Enable = unsafe extern "C" fn(u32, usize, i32, u32) -> i32;
type Unsubscribe = unsafe extern "C" fn(usize) -> i32;
pub(crate) struct Recorder {
    library: Option<Library>,
    state: Option<Box<CallbackState>>,
    subscriber: usize,
}
impl Recorder {
    pub(crate) fn new(path: &str) -> ExportResult<Self> {
        Ok(Self {
            library: Some(unsafe { Library::new(path) }?),
            state: Some(Box::new(CallbackState::new())),
            subscriber: 0,
        })
    }
    fn state(&self) -> &CallbackState {
        self.state.as_deref().unwrap()
    }
    fn check(&self, name: &str, status: i32) -> ExportResult<()> {
        if status == 0 {
            return Ok(());
        }
        let mut message = std::ptr::null();
        type GetString = unsafe extern "C" fn(i32, *mut *const c_char) -> i32;
        let description = unsafe {
            self.library
                .as_ref()
                .unwrap()
                .get::<GetString>(b"cuptiGetResultString\0")
                .ok()
                .and_then(|f| {
                    if f(status, &mut message) == 0 && !message.is_null() {
                        Some(CStr::from_ptr(message).to_string_lossy().into_owned())
                    } else {
                        None
                    }
                })
        };
        Err(invalid(format!(
            "{name}: CUPTI {status}: {}",
            description.unwrap_or_default()
        )))
    }
    pub(crate) fn start(&mut self) -> ExportResult<()> {
        if self.subscriber != 0 {
            return Err(invalid("module image recorder is already active"));
        }
        let library = self.library.as_ref().unwrap();
        let subscribe = unsafe { *library.get::<Subscribe>(b"cuptiSubscribe\0")? };
        let enable = unsafe { *library.get::<Enable>(b"cuptiEnableCallback\0")? };
        let user = self.state.as_deref_mut().unwrap() as *mut CallbackState;
        let status = unsafe { subscribe(&mut self.subscriber, callback, user.cast()) };
        self.check("cuptiSubscribe", status)?;
        let result = (|| {
            for id in [2, 6, 7] {
                self.check("cuptiEnableCallback", unsafe {
                    enable(1, self.subscriber, 3, id)
                })?;
            }
            for id in [
                18, 19, 20, 22, 23, 678, 679, 680, 681, 682, 683, 687, 742, 85, 686, 374, 384, 451,
                452, 543, 675, 676, 307, 442, 477, 478, 481, 652, 653, 243, 245, 38, 39, 271, 294,
                301, 302, 379, 551, 552, 598, 599, 600, 601, 611, 612,
            ] {
                self.check("cuptiEnableCallback", unsafe {
                    enable(1, self.subscriber, 1, id)
                })?;
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = self.stop();
        }
        result
    }
    pub(crate) fn stop(&mut self) -> ExportResult<()> {
        if self.subscriber != 0 {
            let unsubscribe = unsafe {
                *self
                    .library
                    .as_ref()
                    .unwrap()
                    .get::<Unsubscribe>(b"cuptiUnsubscribe\0")?
            };
            self.check("cuptiUnsubscribe", unsafe { unsubscribe(self.subscriber) })?;
            self.subscriber = 0;
            let state = self.state();
            let mut count = state.inflight.lock().unwrap_or_else(|e| e.into_inner());
            while *count > 0 {
                count = state.drained.wait(count).unwrap_or_else(|e| e.into_inner());
            }
        }
        self.check_error()
    }
    pub(crate) fn check_error(&self) -> ExportResult<()> {
        if let Some(error) = &self
            .state()
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .error
        {
            Err(invalid(format!("module image callback failed: {error}")))
        } else {
            Ok(())
        }
    }
    pub(crate) fn recorded_allocations(&self) -> ExportResult<Vec<RecordedAllocation>> {
        self.check_error()?;
        Ok(self
            .state()
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .allocations
            .values()
            .cloned()
            .collect())
    }
    pub(crate) fn graph_node_id(&self, node: usize) -> ExportResult<u64> {
        self.check_error()?;
        type GetId = unsafe extern "C" fn(usize, *mut u64) -> i32;
        let get = unsafe {
            *self
                .library
                .as_ref()
                .unwrap()
                .get::<GetId>(b"cuptiGetGraphNodeId\0")?
        };
        let mut id = 0;
        self.check("cuptiGetGraphNodeId", unsafe { get(node, &mut id) })?;
        Ok(id)
    }

    pub(crate) fn image_for(&self, function: usize) -> ExportResult<Arc<[u8]>> {
        self.check_error()?;
        #[cfg(windows)]
        let library: Library =
            unsafe { libloading::os::windows::Library::load_with_flags("nvcuda.dll", 0x800)? }
                .into();
        #[cfg(not(windows))]
        let library = unsafe { Library::new("libcuda.so.1") }?;
        type Get = unsafe extern "C" fn(*mut usize, usize) -> i32;
        type Name = unsafe extern "C" fn(*mut *const c_char, usize) -> i32;
        type Current = unsafe extern "C" fn(*mut usize) -> i32;
        let get = unsafe { *library.get::<Get>(b"cuFuncGetModule\0")? };
        let kernel_get = unsafe { *library.get::<Get>(b"cuKernelGetFunction\0")? };
        let library_get = unsafe { *library.get::<Get>(b"cuLibraryGetModule\0")? };
        let check = |name: &str, status: i32| -> ExportResult<()> {
            if status == 0 {
                Ok(())
            } else {
                Err(invalid(format!("{name} failed with CUDA status {status}")))
            }
        };
        let mut context = 0;
        let current = unsafe { *library.get::<Current>(b"cuCtxGetCurrent\0")? };
        check("cuCtxGetCurrent", unsafe { current(&mut context) })?;
        let mut module = 0;
        check("cuFuncGetModule", unsafe { get(&mut module, function) })?;
        {
            let state = self.state().state.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(image) = state.modules.get(&module) {
                return Ok(image.cubin.clone());
            }
        }
        let (kernels, mut functions, libraries) = {
            let state = self.state().state.lock().unwrap_or_else(|e| e.into_inner());
            (
                state.kernels.clone(),
                state.functions.clone(),
                state.libraries.clone(),
            )
        };
        for (kernel, image) in kernels {
            let mut handle = 0;
            check("cuKernelGetFunction", unsafe {
                kernel_get(&mut handle, kernel)
            })?;
            functions.insert(handle, image);
        }
        let mut resolved = Vec::new();
        for (handle, image) in functions {
            let mut parent = 0;
            let status = unsafe { get(&mut parent, handle) };
            if status == 400 {
                let mut function = 0;
                check("cuKernelGetFunction", unsafe {
                    kernel_get(&mut function, handle)
                })?;
                check("cuFuncGetModule", unsafe { get(&mut parent, function) })?;
            } else {
                check("cuFuncGetModule", status)?;
            }
            resolved.push((parent, Binding::materialized(&image)));
        }
        for (handle, image) in libraries {
            let mut parent = 0;
            check("cuLibraryGetModule", unsafe {
                library_get(&mut parent, handle)
            })?;
            resolved.push((parent, image.in_context(context)));
        }
        {
            let mut state = self.state().state.lock().unwrap_or_else(|e| e.into_inner());
            for (handle, binding) in resolved {
                let live = binding.image.as_ref().map_or_else(
                    || {
                        state
                            .libraries
                            .values()
                            .any(|library| Arc::ptr_eq(&library.owner, &binding.owner))
                    },
                    |image| state.live(image),
                );
                if live {
                    state.modules.insert(handle, binding);
                }
            }
            if let Some(image) = state.modules.get(&module) {
                return Ok(image.cubin.clone());
            }
        }
        let mut name = std::ptr::null();
        let get_name = unsafe { *library.get::<Name>(b"cuFuncGetName\0")? };
        check("cuFuncGetName", unsafe { get_name(&mut name, function) })?;
        if name.is_null() {
            return Err(invalid("CUDA returned a kernel without its name"));
        }
        let needle = unsafe { CStr::from_ptr(name) }.to_bytes_with_nul();
        let mut state = self.state().state.lock().unwrap_or_else(|e| e.into_inner());
        let mut matches: Vec<Arc<Image>> = vec![];
        for image in state.images.values() {
            if image.cubin.windows(needle.len()).any(|w| w == needle)
                && !matches.iter().any(|other| other.cubin == image.cubin)
            {
                matches.push(image.clone());
            }
        }
        if matches.len() != 1 {
            return Err(invalid(format!(
                "cannot uniquely associate captured kernel {:?} with its loaded cubin",
                unsafe { CStr::from_ptr(name) }
            )));
        }
        let image = matches.pop().unwrap();
        if !state.live(&image) {
            return Err(invalid(
                "captured function's module was unloaded during export",
            ));
        }
        state.bind_module(module, &image);
        Ok(image.cubin.clone())
    }
}
impl Drop for Recorder {
    fn drop(&mut self) {
        if self.stop().is_err() && self.subscriber != 0 {
            // Failed unsubscription leaves CUDA free to invoke the retained callback state.
            if let Some(state) = self.state.take() {
                Box::leak(state);
            }
            if let Some(library) = self.library.take() {
                std::mem::forget(library);
            }
        }
    }
}

#[pyclass]
pub(crate) struct NativeModuleRecorder {
    pub(crate) recorder: Mutex<Recorder>,
}
#[pymethods]
impl NativeModuleRecorder {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        Ok(Self {
            recorder: Mutex::new(Recorder::new(path).map_err(py_error)?),
        })
    }
    fn start(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.recorder.lock().unwrap().start())
            .map_err(py_error)
    }
    fn stop(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.recorder.lock().unwrap().stop())
            .map_err(py_error)
    }
    fn check_error(&self) -> PyResult<()> {
        self.recorder
            .lock()
            .unwrap()
            .check_error()
            .map_err(py_error)
    }
    fn allocations(&self) -> PyResult<BTreeMap<u64, (u64, String)>> {
        Ok(self
            .recorder
            .lock()
            .unwrap()
            .recorded_allocations()
            .map_err(py_error)?
            .into_iter()
            .map(|a| (a.address, (a.nbytes, a.memory)))
            .collect())
    }
    fn allocation_contexts(&self) -> PyResult<BTreeMap<u64, usize>> {
        Ok(self
            .recorder
            .lock()
            .unwrap()
            .recorded_allocations()
            .map_err(py_error)?
            .into_iter()
            .filter_map(|a| a.context.map(|context| (a.address, context)))
            .collect())
    }
}
fn py_error(error: Box<dyn std::error::Error + Send + Sync>) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}
#[pyfunction]
fn cubin_strip_debug<'py>(py: Python<'py>, image: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    Ok(PyBytes::new(
        py,
        &super::cubin::strip_debug(image).map_err(|e| PyValueError::new_err(e.to_string()))?,
    ))
}
#[pyfunction]
fn cubin_strip_global_source<'py>(
    py: Python<'py>,
    symbol: &str,
    data: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    Ok(PyBytes::new(
        py,
        &super::cubin::strip_global_source(symbol, data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    ))
}
#[pyfunction]
fn cubin_global_symbols(image: &[u8]) -> PyResult<Vec<String>> {
    super::cubin::global_symbols(image).map_err(|e| PyValueError::new_err(e.to_string()))
}
#[pyfunction]
fn cubin_redact_symbols<'py>(
    py: Python<'py>,
    image: &[u8],
) -> PyResult<(Bound<'py, PyBytes>, BTreeMap<String, String>)> {
    let (image, symbols) =
        super::cubin::redact_symbols(image).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((PyBytes::new(py, &image), symbols))
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeModuleRecorder>()?;
    module.add_function(wrap_pyfunction!(cubin_strip_debug, module)?)?;
    module.add_function(wrap_pyfunction!(cubin_strip_global_source, module)?)?;
    module.add_function(wrap_pyfunction!(cubin_global_symbols, module)?)?;
    module.add_function(wrap_pyfunction!(cubin_redact_symbols, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    fn call(
        owner: &CallbackState,
        name: &str,
        site: i32,
        params: &[usize],
        status: i32,
        context: usize,
    ) {
        let name = CString::new(name).unwrap();
        let call = Call {
            site,
            name: name.as_ptr(),
            params: params.as_ptr(),
            result: &status,
            symbol: std::ptr::null(),
            context,
            context_id: 0,
            correlation: std::ptr::null_mut(),
            correlation_id: 0,
        };
        unsafe {
            callback(
                owner as *const _ as *mut c_void,
                1,
                0,
                &call as *const _ as *const c_void,
            );
        }
    }
    fn resource(owner: &CallbackState, context: usize, id: u32, kind: u32, image: &[u8]) {
        let module = Module {
            id,
            size: image.len(),
            image: image.as_ptr(),
        };
        let resource = Resource {
            context,
            handle: 0,
            descriptor: &module,
        };
        unsafe {
            callback(
                owner as *const _ as *mut c_void,
                3,
                kind,
                &resource as *const _ as *const c_void,
            );
        }
    }
    fn launch(owner: &CallbackState, name: &str, handle: usize, context: usize, id: u32) {
        call(owner, name, 0, &[], 0, context);
        resource(owner, context, id, 6, b"native cubin");
        let params = if name == "cuLaunchKernel" {
            vec![handle, 0]
        } else {
            vec![0, handle]
        };
        call(owner, name, 1, &params, 0, context);
    }
    #[test]
    fn occupancy_queries_bind_even_when_configuration_fails() {
        for name in [
            "cuFuncGetAttribute",
            "cuKernelGetAttribute",
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
            "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
            "cuOccupancyMaxPotentialBlockSize",
            "cuOccupancyMaxPotentialBlockSizeWithFlags",
            "cuOccupancyAvailableDynamicSMemPerBlock",
            "cuOccupancyMaxPotentialClusterSize",
            "cuOccupancyMaxActiveClusters",
        ] {
            for status in [0, 1] {
                let owner = CallbackState::new();
                let (kernel, index) = query(name).unwrap();
                let mut values = vec![0; index + 1];
                values[index] = 0x1234;
                call(&owner, name, 0, &[], 0, 0x1000);
                resource(&owner, 0x1000, 7, 6, b"queried cubin");
                call(&owner, name, 1, &values, status, 0x1000);
                {
                    let state = owner.state.lock().unwrap();
                    assert!(state.error.is_none(), "{:?}", state.error);
                    let bindings = if kernel {
                        &state.kernels
                    } else {
                        &state.functions
                    };
                    assert_eq!(&*bindings[&0x1234].cubin, b"queried cubin");
                }
                resource(&owner, 0x1000, 7, 7, &[]);
                let state = owner.state.lock().unwrap();
                assert!(state.functions.is_empty());
                assert!(state.kernels.is_empty());
            }
        }
    }
    #[test]
    fn rebinding_preserves_other_modules_and_contexts() {
        let owner = CallbackState::new();
        launch(&owner, "cuKernelSetAttribute", 0x11, 0x1000, 7);
        launch(&owner, "cuLaunchKernelEx", 0x11, 0x1000, 7);
        launch(&owner, "cuLaunchKernelEx", 0x11, 0x1000, 9);
        launch(&owner, "cuLaunchKernel", 0x22, 0x2000, 7);
        resource(&owner, 0x1000, 7, 7, &[]);
        let state = owner.state.lock().unwrap();
        assert!(state.kernels.is_empty());
        assert_eq!(state.functions[&0x11].module_id, 9);
        assert_eq!(state.functions[&0x22].context, 0x2000);
        assert_eq!(state.images.len(), 2);
    }
    #[test]
    fn unloaded_cubin_is_released() {
        let owner = CallbackState::new();
        launch(&owner, "cuLaunchKernel", 0x11, 0x1000, 7);
        let discarded = {
            let state = owner.state.lock().unwrap();
            Arc::downgrade(&state.images[&(0x1000, 7)])
        };
        resource(&owner, 0x1000, 7, 7, &[]);
        assert!(discarded.upgrade().is_none());
    }
    #[test]
    fn nested_and_concurrent_materializers_have_disjoint_stacks() {
        let owner = Arc::new(CallbackState::new());
        std::thread::scope(|scope| {
            for i in 1..=8 {
                let owner = &owner;
                scope.spawn(move || {
                    call(owner, "cuFuncLoad", 0, &[], 0, i);
                    launch(owner, "cuLaunchKernel", i, i, i as u32);
                    resource(owner, i, 100 + i as u32, 6, b"outer cubin");
                    call(owner, "cuFuncLoad", 1, &[100 + i], 0, i);
                });
            }
        });
        let state = owner.state.lock().unwrap();
        assert!(state.error.is_none(), "{:?}", state.error);
        assert!(state.pending.is_empty());
        assert_eq!(state.functions.len(), 16);
        for i in 1..=8 {
            assert_eq!(&*state.functions[&i].cubin, b"native cubin");
            assert_eq!(&*state.functions[&(100 + i)].cubin, b"outer cubin");
        }
    }
    #[test]
    fn context_destruction_removes_only_its_device_allocations_and_modules() {
        let owner = CallbackState::new();
        launch(&owner, "cuLaunchKernel", 0x11, 0x1000, 7);
        launch(&owner, "cuLaunchKernel", 0x22, 0x2000, 7);
        let device = 0x8000usize;
        call(
            &owner,
            "cuMemAlloc_v2",
            1,
            &[&device as *const _ as usize, 4096],
            0,
            0x1000,
        );
        let host = 0x9000usize;
        call(
            &owner,
            "cuMemHostAlloc",
            1,
            &[&host as *const _ as usize, 4096],
            0,
            0x1000,
        );
        resource(&owner, 0x1000, 0, 2, &[]);
        let state = owner.state.lock().unwrap();
        assert!(!state.allocations.contains_key(&0x8000));
        assert_eq!(state.allocations[&0x9000].context, None);
        assert_eq!(state.images.len(), 1);
        assert_eq!(state.functions.len(), 1);
        assert!(state.functions.contains_key(&0x22));
    }
    #[test]
    fn allocation_and_mapping_lifetimes_follow_successful_calls() {
        let owner = CallbackState::new();
        let address = 0x8000usize;
        call(
            &owner,
            "cuMemAllocAsync",
            1,
            &[&address as *const _ as usize, 4096, 0],
            1,
            0x1000,
        );
        assert!(owner.state.lock().unwrap().allocations.is_empty());
        call(&owner, "cuMemMap", 1, &[address, 8192, 0, 0, 0], 0, 0x1000);
        assert_eq!(
            owner.state.lock().unwrap().allocations[&(address as u64)].nbytes,
            8192
        );
        call(&owner, "cuMemUnmap", 1, &[address, 8192], 0, 0x1000);
        assert!(owner.state.lock().unwrap().allocations.is_empty());
    }
    #[test]
    fn callback_failure_is_retained_without_crossing_ffi_boundary() {
        let owner = CallbackState::new();
        call(&owner, "cuFuncLoad", 1, &[123], 0, 0);
        assert!(
            owner
                .state
                .lock()
                .unwrap()
                .error
                .as_ref()
                .unwrap()
                .contains("unbalanced")
        );
        resource(&owner, 1, 1, 6, &[]);
        assert!(
            owner
                .state
                .lock()
                .unwrap()
                .error
                .as_ref()
                .unwrap()
                .contains("unbalanced")
        );
        assert_eq!(*owner.inflight.lock().unwrap(), 0);
    }
    #[test]
    fn unload_and_bind_are_serialized() {
        for _ in 0..30 {
            let owner = Arc::new(CallbackState::new());
            launch(&owner, "cuLaunchKernel", 0x11, 0x1000, 7);
            let image = owner.state.lock().unwrap().images[&(0x1000, 7)].clone();
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    let mut state = owner.state.lock().unwrap();
                    state.bind(false, 0x22, &image);
                    state.bind_module(0x33, &image);
                });
                scope.spawn(|| resource(&owner, 0x1000, 7, 7, &[]));
            });
            let state = owner.state.lock().unwrap();
            assert!(state.images.is_empty());
            assert!(state.functions.is_empty());
            assert!(state.modules.is_empty());
        }
    }
    fn library_image(tag: u8) -> Vec<u8> {
        let mut image = vec![0; 128];
        image[..6].copy_from_slice(b"\x7fELF\x02\x01");
        image[9] = tag;
        image[40..48].copy_from_slice(&64u64.to_le_bytes());
        image[58..60].copy_from_slice(&64u16.to_le_bytes());
        image[60..62].copy_from_slice(&1u16.to_le_bytes());
        image[68..72].copy_from_slice(&8u32.to_le_bytes());
        image
    }
    fn load_library(owner: &CallbackState, library: usize, image: &[u8], context: usize) {
        call(owner, "cuLibraryLoadData", 0, &[], 0, context);
        call(
            owner,
            "cuLibraryLoadData",
            1,
            &[&library as *const _ as usize, image.as_ptr() as usize],
            0,
            context,
        );
    }
    fn library_module(
        owner: &CallbackState,
        library: usize,
        module: usize,
        context: usize,
        image: Option<(u32, &[u8])>,
    ) {
        call(owner, "cuLibraryGetModule", 0, &[], 0, context);
        if let Some((id, image)) = image {
            resource(owner, context, id, 6, image);
        }
        call(
            owner,
            "cuLibraryGetModule",
            1,
            &[&module as *const _ as usize, library],
            0,
            context,
        );
    }
    #[test]
    fn library_unload_removes_lazy_aliases_and_reused_handles_get_their_cubin() {
        let owner = CallbackState::new();
        let first = library_image(1);
        let replacement = library_image(2);
        load_library(&owner, 0x10, &first, 0x1000);
        load_library(&owner, 0x20, &first, 0x1000);
        library_module(&owner, 0x10, 0x100, 0x1000, None);
        library_module(&owner, 0x20, 0x200, 0x1000, None);
        {
            let state = owner.state.lock().unwrap();
            assert!(state.modules[&0x100].image.is_none());
            assert!(!Arc::ptr_eq(
                &state.modules[&0x100].owner,
                &state.modules[&0x200].owner
            ));
        }
        call(&owner, "cuLibraryUnload", 1, &[0x10], 0, 0x1000);
        {
            let state = owner.state.lock().unwrap();
            assert!(!state.modules.contains_key(&0x100));
            assert_eq!(&*state.modules[&0x200].cubin, first);
            assert!(state.libraries.contains_key(&0x20));
        }
        load_library(&owner, 0x10, &replacement, 0x1000);
        library_module(&owner, 0x10, 0x100, 0x1000, None);
        let state = owner.state.lock().unwrap();
        assert!(state.error.is_none(), "{:?}", state.error);
        assert_eq!(&*state.modules[&0x100].cubin, replacement);
        assert_eq!(&*state.modules[&0x200].cubin, first);
    }
    #[test]
    fn module_unload_invalidates_only_its_materialized_aliases() {
        let owner = CallbackState::new();
        let image = library_image(1);
        load_library(&owner, 0x10, &image, 0x1000);
        load_library(&owner, 0x20, &image, 0x1000);
        library_module(&owner, 0x10, 0x100, 0x1000, Some((7, &image)));
        library_module(&owner, 0x20, 0x200, 0x1000, Some((9, &image)));
        resource(&owner, 0x1000, 7, 7, &[]);
        {
            let state = owner.state.lock().unwrap();
            assert!(!state.modules.contains_key(&0x100));
            assert!(state.modules.contains_key(&0x200));
            assert!(state.libraries[&0x10].image.is_none());
            assert!(state.libraries[&0x20].image.is_some());
        }
        library_module(&owner, 0x10, 0x100, 0x1000, Some((11, &image)));
        resource(&owner, 0x1000, 7, 7, &[]);
        let state = owner.state.lock().unwrap();
        assert!(state.error.is_none(), "{:?}", state.error);
        assert_eq!(state.modules[&0x100].image.as_ref().unwrap().module_id, 11);
        assert_eq!(state.modules[&0x200].image.as_ref().unwrap().module_id, 9);
    }
    #[test]
    fn context_unload_preserves_library_ownership_and_other_context_aliases() {
        let owner = CallbackState::new();
        let image = library_image(1);
        load_library(&owner, 0x10, &image, 0);
        library_module(&owner, 0x10, 0x100, 0x1000, Some((7, &image)));
        library_module(&owner, 0x10, 0x200, 0x2000, Some((9, &image)));
        resource(&owner, 0x1000, 0, 2, &[]);
        {
            let state = owner.state.lock().unwrap();
            assert!(!state.modules.contains_key(&0x100));
            assert_eq!(state.modules[&0x200].context, 0x2000);
            assert!(state.libraries.contains_key(&0x10));
        }
        call(&owner, "cuLibraryUnload", 1, &[0x10], 0, 0);
        let state = owner.state.lock().unwrap();
        assert!(state.error.is_none(), "{:?}", state.error);
        assert!(state.modules.is_empty());
        assert!(state.libraries.is_empty());
    }
    #[test]
    fn unload_before_materializer_exit_cannot_resurrect_module_alias() {
        let owner = CallbackState::new();
        let image = library_image(1);
        load_library(&owner, 0x10, &image, 0x1000);
        call(&owner, "cuLibraryGetModule", 0, &[], 0, 0x1000);
        resource(&owner, 0x1000, 7, 6, &image);
        resource(&owner, 0x1000, 7, 7, &[]);
        let module = 0x100usize;
        call(
            &owner,
            "cuLibraryGetModule",
            1,
            &[&module as *const _ as usize, 0x10],
            0,
            0x1000,
        );
        let state = owner.state.lock().unwrap();
        assert!(state.error.is_none(), "{:?}", state.error);
        assert!(state.modules.is_empty());
        assert!(state.libraries[&0x10].image.is_none());
    }
}
