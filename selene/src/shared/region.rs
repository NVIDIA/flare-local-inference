//! Shared arena and staging files; owners and clients keep their own mapping lifetimes.

/// The two shared region files.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Region {
    Arena,
    Staging,
}

impl Region {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Arena),
            2 => Some(Self::Staging),
            _ => None,
        }
    }
}

pub const ARENA_FILE: &str = "arena";
pub const STAGING_FILE: &str = "staging";

#[cfg(unix)]
pub fn map_file(file: &std::fs::File, length: usize) -> std::io::Result<*mut libc::c_void> {
    use std::os::fd::AsRawFd;
    let pointer = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            length,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    if pointer == libc::MAP_FAILED {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(pointer)
    }
}
