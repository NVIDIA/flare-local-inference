//! Shared-memory regions and window validation.
//!
//! Both files the manager returns are mapped read-write with `MAP_SHARED`.
//! The protocol hands each window to exactly one owner at a time, so the
//! client only performs CPU `copy_from_slice` into or out of a checked range
//! and never gives these mappings to CUDA.

use std::{fs::File, os::fd::AsRawFd, path::Path};

use crate::Result;

use super::super::geometry::invalid;
use super::proto;

/// Every window offset and length is a multiple of the record block size.
pub(super) const BLOCK: u64 = 4096;

/// A mapped manager region. `Send` is required because the mapping lives on
/// the `selene-io` thread while it is created by a blocking task.
pub(super) struct SharedRegion {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: CPU-only access; the protocol hands each window to one owner.
unsafe impl Send for SharedRegion {}

impl SharedRegion {
    /// Open `path`, require exactly `expected` bytes, and map it shared.
    pub(super) fn map(path: &Path, expected: u64) -> Result<Self> {
        let file = File::options()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|error| {
                format!(
                    "cannot open cache manager region {}: {error}",
                    path.display()
                )
            })?;
        let size = file
            .metadata()
            .map_err(|error| {
                format!(
                    "cannot inspect cache manager region {}: {error}",
                    path.display()
                )
            })?
            .len();
        if size != expected {
            return Err(format!(
                "cache manager region {} is {size} bytes; the manager announced {expected}",
                path.display()
            )
            .into());
        }
        if expected == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
            });
        }
        let len = usize::try_from(expected)
            .map_err(|_| invalid("cache manager region exceeds the address space"))?;
        // SAFETY: A fresh file descriptor is mapped for its exact announced
        // length; the mapping is private to this client and unmapped on drop.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(format!(
                "cannot map cache manager region {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            )
            .into());
        }
        Ok(Self {
            ptr: ptr.cast(),
            len,
        })
    }

    fn checked(&self, offset: u64, length: u64) -> Result<(usize, usize)> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| invalid("cache manager window overflows its region"))?;
        if length == 0 || !offset.is_multiple_of(BLOCK) || !length.is_multiple_of(BLOCK) {
            return Err(invalid(
                "cache manager window is not a nonzero whole block multiple",
            ));
        }
        let (offset, end) = (usize::try_from(offset), usize::try_from(end));
        let (Ok(offset), Ok(end)) = (offset, end) else {
            return Err(invalid("cache manager window exceeds the address space"));
        };
        if end > self.len {
            return Err(invalid(format!(
                "cache manager window {offset}..{end} exceeds its {} byte region",
                self.len
            )));
        }
        Ok((offset, end))
    }

    /// Require `offset..offset + length` to be a whole, in-range block range.
    pub(super) fn check(&self, offset: u64, length: u64) -> Result<()> {
        self.checked(offset, length).map(|_| ())
    }

    /// Copy a window range into `destination`, which must be the same length.
    pub(super) fn read(&self, offset: u64, length: u64, destination: &mut [u8]) -> Result<()> {
        let (start, end) = self.checked(offset, length)?;
        // SAFETY: The range was bounds-checked against this mapping, and one
        // side owns the window, so no concurrent writer aliases it.
        let source = unsafe { std::slice::from_raw_parts(self.ptr.add(start), end - start) };
        destination.copy_from_slice(source);
        Ok(())
    }

    /// Copy `source` into a window range, which must be the same length.
    pub(super) fn write(&self, offset: u64, length: u64, source: &[u8]) -> Result<()> {
        let (start, end) = self.checked(offset, length)?;
        // SAFETY: The range was bounds-checked and this side owns the window,
        // so no concurrent reader aliases it.
        let destination =
            unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), end - start) };
        destination.copy_from_slice(source);
        Ok(())
    }
}

impl Drop for SharedRegion {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len != 0 {
            // SAFETY: This is the mapping returned by `mmap` above, still
            // owned solely by this value.
            unsafe {
                libc::munmap(self.ptr.cast(), self.len);
            }
        }
    }
}

/// A validated `WindowGrant` for the local service.
#[derive(Clone, Debug)]
pub(super) struct Grant {
    pub transfer: u64,
    pub index: u32,
    pub record: u32,
    pub record_offset: u64,
    pub length: u64,
    pub arena: bool,
    pub offset: u64,
    /// Absolute page index within `record`'s chain record. A Put record starts
    /// at the selected parent, so its indices include the inherited pages; a
    /// Get record is one chain node. Row-only grants carry zero.
    pub page_first: usize,
    pub page_count: usize,
    pub row: bool,
}

impl Grant {
    pub(super) fn decode(grant: proto::WindowGrant) -> Result<Self> {
        let memory = grant
            .memory
            .as_ref()
            .ok_or("cache manager window grant has no memory range")?;
        let region = proto::Region::try_from(memory.region)
            .map_err(|_| "cache manager window grant has an unknown region")?;
        let arena = match region {
            proto::Region::Arena => true,
            proto::Region::Staging => false,
            proto::Region::Unspecified => {
                return Err("cache manager window grant has no region".into());
            }
        };
        if memory.length != grant.length {
            return Err("cache manager window grant memory length disagrees".into());
        }
        let page_first = usize::try_from(grant.page_first)
            .map_err(|_| "cache manager window grant page index overflows")?;
        let page_count = usize::try_from(grant.page_count)
            .map_err(|_| "cache manager window grant page count overflows")?;
        if page_count == 0 && !grant.row {
            return Err("cache manager window grant carries neither pages nor a row".into());
        }
        Ok(Self {
            transfer: grant.transfer,
            index: grant.index,
            record: grant.record,
            record_offset: grant.record_offset,
            length: grant.length,
            arena,
            offset: memory.offset,
            page_first,
            page_count,
            row: grant.row,
        })
    }
}
