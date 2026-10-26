//! Bounded Windows overlapped I/O with stable request storage.

use std::{fs::File, io, os::windows::io::AsRawHandle};
use windows_sys::Win32::{
    Foundation::*,
    Storage::FileSystem::*,
    System::{IO::*, Threading::INFINITE},
};

struct Slot {
    operation: OVERLAPPED,
    file: HANDLE,
    pending: bool,
}

pub struct Queue {
    port: HANDLE,
    slots: Box<[Slot]>,
}

// SAFETY: The owner serializes submissions and completions; addresses remain stable when moved.
unsafe impl Send for Queue {}

impl Queue {
    pub fn new(depth: usize) -> io::Result<Self> {
        let port =
            unsafe { CreateIoCompletionPort(INVALID_HANDLE_VALUE, std::ptr::null_mut(), 0, 0) };
        if port.is_null() {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            port,
            slots: (0..depth)
                .map(|_| Slot {
                    operation: unsafe { std::mem::zeroed() },
                    file: std::ptr::null_mut(),
                    pending: false,
                })
                .collect(),
        })
    }

    pub fn attach(&self, file: &File) -> io::Result<()> {
        if unsafe { CreateIoCompletionPort(file.as_raw_handle(), self.port, 0, 0) }.is_null() {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// The file must be attached to this queue. Retain it and the buffer until completion.
    pub unsafe fn submit(
        &mut self,
        slot: usize,
        file: &File,
        address: usize,
        bytes: u32,
        offset: u64,
        write: bool,
    ) -> io::Result<()> {
        let entry = self
            .slots
            .get_mut(slot)
            .ok_or(io::ErrorKind::InvalidInput)?;
        if entry.pending {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "I/O slot is in use",
            ));
        }
        let handle = file.as_raw_handle();
        entry.operation = unsafe { std::mem::zeroed() };
        entry.operation.Anonymous.Anonymous.Offset = offset as u32;
        entry.operation.Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
        entry.file = handle;
        let ok = unsafe {
            if write {
                WriteFile(
                    handle,
                    address as *const u8,
                    bytes,
                    std::ptr::null_mut(),
                    &mut entry.operation,
                )
            } else {
                ReadFile(
                    handle,
                    address as *mut u8,
                    bytes,
                    std::ptr::null_mut(),
                    &mut entry.operation,
                )
            }
        };
        if ok == 0 {
            let error = unsafe { GetLastError() };
            if error != ERROR_IO_PENDING {
                return Err(io::Error::from_raw_os_error(error as i32));
            }
        }
        // Successful synchronous operations also enqueue completion packets.
        entry.pending = true;
        Ok(())
    }

    pub fn pending(&self) -> usize {
        self.slots.iter().filter(|slot| slot.pending).count()
    }

    pub fn wait(&mut self) -> io::Result<(usize, io::Result<usize>)> {
        let mut bytes = 0;
        let mut key = 0;
        let mut operation = std::ptr::null_mut();
        let ok = unsafe {
            GetQueuedCompletionStatus(self.port, &mut bytes, &mut key, &mut operation, INFINITE)
        };
        let error = if ok == 0 {
            Some(io::Error::last_os_error())
        } else {
            None
        };
        if operation.is_null() {
            return Err(error.unwrap_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "I/O completion has no request")
            }));
        }
        let slot = self
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, slot)| std::ptr::eq(&slot.operation, operation))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "unknown I/O completion"))?;
        if !slot.1.pending {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "duplicate I/O completion",
            ));
        }
        slot.1.pending = false;
        Ok((
            slot.0,
            match error {
                Some(error) => Err(error),
                None => Ok(bytes as usize),
            },
        ))
    }

    pub fn cancel(&mut self) {
        for slot in &self.slots {
            if slot.pending {
                unsafe {
                    CancelIoEx(slot.file, &slot.operation);
                }
            }
        }
    }

    pub fn cancel_and_drain(&mut self) {
        self.cancel();
        // Cancellation is only a request. Keep buffers and OVERLAPPED records live until completion.
        while self.pending() != 0 {
            if self.wait().is_err() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        self.cancel_and_drain();
        unsafe {
            CloseHandle(self.port);
        }
    }
}
