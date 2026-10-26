//! Host operating-system operations used by native serving.

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::*;
#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::*;
#[cfg(windows)]
pub mod completion;
#[cfg(windows)]
pub mod ioring;

use std::{fs::File, io};

pub trait FileReadAt {
    fn read_at(&self, bytes: &mut [u8], offset: u64) -> io::Result<usize>;
    fn read_exact_at(&self, mut bytes: &mut [u8], mut offset: u64) -> io::Result<()> {
        while !bytes.is_empty() {
            match self.read_at(bytes, offset) {
                Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => {
                    offset += n as u64;
                    bytes = &mut bytes[n..];
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

impl FileReadAt for File {
    fn read_at(&self, bytes: &mut [u8], offset: u64) -> io::Result<usize> {
        unsafe { read_at_raw(self, bytes.as_mut_ptr(), bytes.len(), offset) }
    }
}

pub fn allocate_aligned(bytes: usize) -> io::Result<usize> {
    let layout = std::alloc::Layout::from_size_align(bytes.max(1), 4096)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let ptr = unsafe { std::alloc::alloc(layout) };
    if ptr.is_null() {
        return Err(io::ErrorKind::OutOfMemory.into());
    }
    Ok(ptr as usize)
}

/// # Safety
///
/// `address` and `bytes` must be those of an unfreed [`allocate_aligned`] allocation.
pub unsafe fn free_aligned(address: usize, bytes: usize) {
    let layout = std::alloc::Layout::from_size_align(bytes.max(1), 4096).unwrap();
    unsafe { std::alloc::dealloc(address as *mut u8, layout) };
}
