//! Hardware revision selection through NVML without initializing CUDA.

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};

use libloading::Library;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn platform(pci_id: u32) -> Option<&'static str> {
    // Single mapping table shared with package export naming.
    flare_helios::package::naming::platform_from_pci(pci_id)
}

/// Keep NVML's background threads out of the process that sets CUDA visibility.
pub fn detect_isolated(count: usize) -> Result<String> {
    let output = std::process::Command::new(std::env::current_exe()?)
        .arg("hardware-revision")
        .arg(count.to_string())
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "hardware detection failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

pub fn detect(count: usize) -> Result<&'static str> {
    let inherited = std::env::var("CUDA_VISIBLE_DEVICES").ok();
    let indices: Vec<_> = (0..count as i32).collect();
    let visible = crate::environment::visibility(&indices, inherited.as_deref())?;
    // SAFETY: Only the system driver is loaded, and the library remains live through all calls.
    #[cfg(unix)]
    let library = unsafe { Library::new("libnvidia-ml.so.1") };
    #[cfg(windows)]
    let library = unsafe {
        libloading::os::windows::Library::load_with_flags(
            "nvml.dll",
            windows_sys::Win32::System::LibraryLoader::LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
        .map(Library::from)
    };
    let library = library.map_err(|error| {
        format!("cannot detect hardware without NVIDIA NVML: {error}; specify --revision")
    })?;
    // SAFETY: These signatures and PciInfo match nvml.h; output pointers are valid.
    unsafe {
        let init = library.get::<unsafe extern "C" fn() -> c_int>(b"nvmlInit_v2\0")?;
        let shutdown = library.get::<unsafe extern "C" fn() -> c_int>(b"nvmlShutdown\0")?;
        check("nvmlInit_v2", init())?;
        let result = (|| {
            let get = library.get::<unsafe extern "C" fn(c_uint, *mut Device) -> c_int>(
                b"nvmlDeviceGetHandleByIndex_v2\0",
            )?;
            let get_by_uuid = library
                .get::<unsafe extern "C" fn(*const c_char, *mut Device) -> c_int>(
                    b"nvmlDeviceGetHandleByUUID\0",
                )?;
            let get_count = library
                .get::<unsafe extern "C" fn(*mut c_uint) -> c_int>(b"nvmlDeviceGetCount_v2\0")?;
            let get_uuid = library
                .get::<unsafe extern "C" fn(Device, *mut c_char, c_uint) -> c_int>(
                    b"nvmlDeviceGetUUID\0",
                )?;
            let get_pci = library.get::<unsafe extern "C" fn(Device, *mut PciInfo) -> c_int>(
                b"nvmlDeviceGetPciInfo_v3\0",
            )?;
            let mut selected = None;
            for entry in visible.split(',') {
                let mut device = std::ptr::null_mut();
                if let Ok(index) = entry.parse::<u32>() {
                    check("nvmlDeviceGetHandleByIndex_v2", get(index, &mut device))?;
                } else if entry.len() == 40 {
                    let uuid = CString::new(entry)?;
                    check(
                        "nvmlDeviceGetHandleByUUID",
                        get_by_uuid(uuid.as_ptr(), &mut device),
                    )?;
                } else {
                    let mut total = 0;
                    check("nvmlDeviceGetCount_v2", get_count(&mut total))?;
                    let mut candidates = Vec::new();
                    for index in 0..total {
                        let mut candidate = std::ptr::null_mut();
                        check("nvmlDeviceGetHandleByIndex_v2", get(index, &mut candidate))?;
                        let mut uuid = [0; 96];
                        check(
                            "nvmlDeviceGetUUID",
                            get_uuid(candidate, uuid.as_mut_ptr(), 96),
                        )?;
                        candidates.push((
                            CStr::from_ptr(uuid.as_ptr()).to_str()?.to_owned(),
                            candidate,
                        ));
                    }
                    device = candidates[unique_uuid(entry, &candidates)?].1;
                }
                let mut pci: PciInfo = std::mem::zeroed();
                check("nvmlDeviceGetPciInfo_v3", get_pci(device, &mut pci))?;
                let detected = platform(pci.device_id).ok_or_else(|| {
                    format!(
                        "GPU {entry} has unsupported PCI ID {:#010x}; specify --revision",
                        pci.device_id
                    )
                })?;
                if selected.is_some_and(|value| value != detected) {
                    return Err(
                        "selected GPUs span multiple platforms; select one platform or specify --revision"
                            .into(),
                    );
                }
                selected = Some(detected);
            }
            selected.ok_or_else(|| "select at least one GPU or specify --revision".into())
        })();
        let closed = check("nvmlShutdown", shutdown());
        result.and_then(|revision| closed.map(|()| revision))
    }
}

type Device = *mut c_void;

/// nvmlPciInfo_t as filled by nvmlDeviceGetPciInfo_v3.
#[repr(C)]
struct PciInfo {
    _bus_id_legacy: [c_char; 16],
    _domain: c_uint,
    _bus: c_uint,
    _device: c_uint,
    device_id: c_uint,
    _subsystem_id: c_uint,
    _bus_id: [c_char; 32],
}

fn unique_uuid(prefix: &str, candidates: &[(String, Device)]) -> Result<usize> {
    let mut matches = candidates
        .iter()
        .enumerate()
        .filter(|(_, (uuid, _))| uuid.starts_with(prefix));
    match (matches.next(), matches.next()) {
        (Some((index, _)), None) => Ok(index),
        _ => {
            Err(format!("CUDA_VISIBLE_DEVICES entry {prefix} must identify exactly one GPU").into())
        }
    }
}

fn check(operation: &str, code: c_int) -> Result<()> {
    if code != 0 {
        return Err(format!(
            "hardware detection: {operation} failed with NVML error {code}; specify --revision"
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_selection_requires_an_unambiguous_prefix() {
        let candidates = vec![
            ("GPU-abcdef01".into(), std::ptr::null_mut()),
            ("GPU-abcdef02".into(), std::ptr::null_mut()),
            ("GPU-12345678".into(), std::ptr::null_mut()),
        ];
        assert_eq!(unique_uuid("GPU-abcdef01", &candidates).unwrap(), 0);
        assert_eq!(unique_uuid("GPU-123", &candidates).unwrap(), 2);
        assert!(unique_uuid("GPU-abcdef0", &candidates).is_err());
        assert!(unique_uuid("GPU-999", &candidates).is_err());
    }

    #[test]
    fn pci_ids_select_the_chip_without_matching_marketing_names() {
        assert_eq!(platform(0x2e1210de), Some("gb10"));
        assert_eq!(platform(0x2bb410de), Some("gb202"));
        assert_eq!(platform(0x2bb110de), Some("gb202"));
        assert_eq!(platform(0x2bb510de), Some("gb202"));
        assert_eq!(platform(0x2b8510de), Some("gb202"));
        assert_eq!(platform(0x31c310de), Some("gb300"));
        assert_eq!(platform(0x26b110de), None);
        assert_eq!(platform(0x2bb41234), None);
    }
}
