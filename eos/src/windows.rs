/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use std::{
    fs::{File, OpenOptions},
    io,
    os::windows::{
        ffi::OsStrExt,
        fs::OpenOptionsExt,
        io::{AsRawHandle, FromRawHandle},
    },
    path::Path,
};
use windows_sys::Win32::{
    Foundation::*,
    Storage::FileSystem::*,
    System::{IO::*, Memory::*, SystemInformation::*, Threading::*},
};

pub fn wide(path: &Path) -> io::Result<Vec<u16>> {
    let mut text: Vec<_> = path.as_os_str().encode_wide().collect();
    if text.contains(&0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "path contains NUL",
        ));
    }
    text.push(0);
    Ok(text)
}

/// # Safety
///
/// `address` must be valid for writes of `bytes` bytes for the call.
pub unsafe fn read_at_raw(
    file: &File,
    address: *mut u8,
    bytes: usize,
    offset: u64,
) -> io::Result<usize> {
    unsafe {
        let event = CreateEventW(std::ptr::null(), 1, 0, std::ptr::null());
        if event.is_null() {
            return Err(io::Error::last_os_error());
        }
        let mut op: OVERLAPPED = std::mem::zeroed();
        op.Anonymous.Anonymous.Offset = offset as u32;
        op.Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
        // Suppress IOCP delivery for this synchronous positional operation.
        op.hEvent = (event as usize | 1) as HANDLE;
        let mut count = 0;
        let ok = ReadFile(
            file.as_raw_handle(),
            address,
            bytes.min(u32::MAX as usize) as u32,
            &mut count,
            &mut op,
        );
        let mut result = if ok != 0 {
            Ok(count as usize)
        } else {
            let error = GetLastError();
            if error == ERROR_IO_PENDING {
                if GetOverlappedResult(file.as_raw_handle(), &op, &mut count, 1) != 0 {
                    Ok(count as usize)
                } else {
                    Err(io::Error::last_os_error())
                }
            } else if error == ERROR_HANDLE_EOF {
                Ok(0)
            } else {
                Err(io::Error::from_raw_os_error(error as i32))
            }
        };
        if result.as_ref().err().and_then(io::Error::raw_os_error) == Some(ERROR_HANDLE_EOF as i32)
        {
            result = Ok(0);
        }
        CloseHandle(event);
        result
    }
}

fn check_alignment(file: File) -> io::Result<File> {
    unsafe {
        let mut info: FILE_STORAGE_INFO = std::mem::zeroed();
        if GetFileInformationByHandleEx(
            file.as_raw_handle(),
            FileStorageInfo,
            (&mut info as *mut FILE_STORAGE_INFO).cast(),
            std::mem::size_of_val(&info) as u32,
        ) == 0
        {
            return Err(io::Error::last_os_error());
        }
        // Unbuffered I/O requires the sector size for atomicity to divide the reader's 4096-byte
        // granularity. `PhysicalBytesPerSectorForPerformance` is a performance hint, not a
        // constraint: this NVMe reports 32768 there, and measured 4096-byte unbuffered reads
        // complete, so rejecting the volume on it makes every direct read unusable.
        for sector in [
            info.LogicalBytesPerSector,
            info.PhysicalBytesPerSectorForAtomicity,
        ] {
            if sector == 0 || 4096 % sector != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("direct I/O sector size {sector} does not divide 4096"),
                ));
            }
        }
    }
    Ok(file)
}

pub fn direct_file(path: &Path, write: bool) -> io::Result<File> {
    let file = OpenOptions::new()
        .read(!write)
        .write(write)
        .create(write)
        .truncate(write)
        .custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)
        .open(path)?;
    check_alignment(file)
}

pub fn direct_reopen(file: &File) -> io::Result<File> {
    reopen_direct(file, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
}

/// Keep a package immutable without retaining a buffered handle alongside its direct readers.
pub fn direct_read_guard(file: &File) -> io::Result<File> {
    reopen_direct(file, FILE_SHARE_READ)
}

fn reopen_direct(file: &File, sharing: u32) -> io::Result<File> {
    let handle = unsafe {
        ReOpenFile(
            file.as_raw_handle(),
            GENERIC_READ,
            sharing,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    check_alignment(unsafe { File::from_raw_handle(handle) })
}

pub fn release_page_cache(_: &File) {}

pub fn page_size() -> usize {
    let mut info: SYSTEM_INFO = unsafe { std::mem::zeroed() };
    unsafe {
        GetSystemInfo(&mut info);
    }
    info.dwPageSize as usize
}

pub fn lower_priority(_: i32) {
    unsafe {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
}

pub fn map_host(bytes: usize) -> io::Result<usize> {
    let ptr = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            bytes,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE,
        )
    };
    if ptr.is_null() {
        return Err(io::Error::last_os_error());
    }
    // Populate the pages before the slab enters the transfer pool.
    for offset in (0..bytes).step_by(page_size()) {
        unsafe {
            ptr.cast::<u8>().add(offset).write_volatile(0);
        }
    }
    Ok(ptr as usize)
}

/// # Safety
///
/// `address` and `bytes` must describe a mapping made by `map_host` that nothing else
/// references any longer, such as one the driver has already unregistered.
pub unsafe fn unmap_host(address: usize, _: usize) {
    unsafe {
        VirtualFree(address as *mut _, 0, MEM_RELEASE);
    }
}

pub fn free_space(path: &Path) -> io::Result<u64> {
    let text = wide(path)?;
    let mut available = 0;
    if unsafe {
        GetDiskFreeSpaceExW(
            text.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(available)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FileReadAt, completion::Queue};

    #[test]
    fn direct_package_guard_preserves_immutability_after_buffered_reader_closes() {
        let path =
            std::env::temp_dir().join(format!("flare-package-guard-{}.bin", std::process::id()));
        std::fs::write(&path, [0x39; 4096]).unwrap();
        let source = OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ)
            .open(&path)
            .unwrap();
        let guard = direct_read_guard(&source).unwrap();
        drop(source);
        assert_eq!(
            OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap_err()
                .raw_os_error(),
            Some(ERROR_SHARING_VIOLATION as i32)
        );
        assert_eq!(
            std::fs::remove_file(&path).unwrap_err().raw_os_error(),
            Some(ERROR_SHARING_VIOLATION as i32)
        );
        let reader = direct_reopen(&guard).unwrap();
        let address = map_host(4096).unwrap();
        let count = unsafe { read_at_raw(&reader, address as *mut u8, 4096, 0) }.unwrap();
        assert_eq!(count, 4096);
        assert!(
            unsafe { std::slice::from_raw_parts(address as *const u8, 4096) }
                .iter()
                .all(|value| *value == 0x39)
        );
        unsafe { unmap_host(address, 4096) };
        drop(reader);
        drop(guard);
        drop(OpenOptions::new().write(true).open(&path).unwrap());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn direct_io_completes_and_cancellation_drains_before_reuse() {
        let path = std::env::temp_dir().join(format!("flare-直接-{}.bin", std::process::id()));
        let address = map_host(8192).unwrap();
        unsafe {
            std::ptr::write_bytes(address as *mut u8, 0x71, 8192);
        }
        let file = direct_file(&path, true).unwrap();
        let mut queue = Queue::new(2).unwrap();
        queue.attach(&file).unwrap();
        for slot in 0..2 {
            unsafe {
                queue
                    .submit(
                        slot,
                        &file,
                        address + slot * 4096,
                        4096,
                        slot as u64 * 4096,
                        true,
                    )
                    .unwrap();
            }
        }
        while queue.pending() > 0 {
            assert_eq!(queue.wait().unwrap().1.unwrap(), 4096);
        }
        drop(queue);
        file.sync_all().unwrap();
        drop(file);
        let source = File::open(&path).unwrap();
        let file = direct_reopen(&source).unwrap();
        let mut queue = Queue::new(2).unwrap();
        queue.attach(&file).unwrap();
        for slot in 0..2 {
            unsafe {
                queue
                    .submit(
                        slot,
                        &file,
                        address + slot * 4096,
                        4096,
                        slot as u64 * 4096,
                        false,
                    )
                    .unwrap();
            }
        }
        queue.cancel();
        while queue.pending() > 0 {
            match queue.wait().unwrap().1 {
                Ok(n) => assert_eq!(n, 4096),
                Err(error) => {
                    assert_eq!(error.raw_os_error(), Some(ERROR_OPERATION_ABORTED as i32))
                }
            }
        }
        unsafe {
            queue.submit(0, &file, address, 4096, 4096, false).unwrap();
        }
        assert_eq!(queue.wait().unwrap().1.unwrap(), 4096);
        assert!(
            unsafe { std::slice::from_raw_parts(address as *const u8, 4096) }
                .iter()
                .all(|byte| *byte == 0x71)
        );
        let mut metadata = [0; 17];
        source.read_exact_at(&mut metadata, 4093).unwrap();
        assert_eq!(metadata, [0x71; 17]);
        drop(queue);
        drop(file);
        drop(source);
        unsafe {
            unmap_host(address, 8192);
        }
        std::fs::remove_file(path).unwrap();
    }
}
