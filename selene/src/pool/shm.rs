//! The manager's shared-memory regions, mapped and registered by the client.
//!
//! Windows are slices of these regions. The protocol, not the type system,
//! decides which side may touch a window at any moment, so slices are handed
//! out only for the span of one check, hash or copy submission.

use flare_eos as os;

use crate::{Error, Registration, Result};

use super::wire;

enum Owner {
    #[cfg(unix)]
    Mapped,
    /// Private anonymous memory this process owns: the staging of a session
    /// without shared memory.
    Anonymous,
    #[cfg(test)]
    Heap,
}

/// One mapped region. Dropping it unregisters the range, then unmaps it.
pub(crate) struct Region {
    // Declared first so it drops first: unregister before unmapping.
    registration: Option<Box<dyn Registration>>,
    base: usize,
    len: usize,
    owner: Owner,
}

impl Region {
    /// Map `bytes` of the file at `path` shared and writable. The file must
    /// exist and be at least that long; the client never creates, truncates or
    /// unlinks it.
    #[cfg(unix)]
    pub(crate) fn map(path: &str, bytes: u64) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| Error::caused_by(format!("cannot open shared memory {path}"), e))?;
        let len = usize::try_from(bytes).map_err(|_| Error::new("shared region too large"))?;
        let size = file
            .metadata()
            .map_err(|e| Error::caused_by(format!("cannot stat {path}"), e))?
            .len();
        if len == 0 || size < bytes {
            return Err(Error::new(format!(
                "shared memory {path} has {size} bytes, expected {bytes}"
            )));
        }
        let address = crate::shared::region::map_file(&file, len)
            .map_err(|e| Error::caused_by(format!("cannot map shared memory {path}"), e))?;
        Ok(Self {
            registration: None,
            base: address as usize,
            len,
            owner: Owner::Mapped,
        })
    }

    #[cfg(not(unix))]
    pub(crate) fn map(_path: &str, _bytes: u64) -> Result<Self> {
        Err(Error::new("shared memory needs a Unix host"))
    }

    /// `bytes` of zeroed private memory.
    pub(crate) fn anonymous(bytes: usize) -> Result<Self> {
        if bytes == 0 {
            return Err(Error::new("a staging region needs some bytes"));
        }
        let base = os::map_host(bytes)
            .map_err(|e| Error::caused_by("cannot map the data channel's staging memory", e))?;
        Ok(Self {
            registration: None,
            base,
            len: bytes,
            owner: Owner::Anonymous,
        })
    }

    /// A zeroed private region, for tests.
    #[cfg(test)]
    pub(crate) fn heap(bytes: usize) -> Self {
        let base = os::allocate_aligned(bytes).unwrap();
        unsafe { std::ptr::write_bytes(base as *mut u8, 0, bytes) };
        Self {
            registration: None,
            base,
            len: bytes,
            owner: Owner::Heap,
        }
    }

    pub(crate) fn register(
        &mut self,
        registrar: &dyn crate::HostRegistrar,
        context: usize,
    ) -> Result<()> {
        self.registration = Some(registrar.register(context, self.base, self.len)?);
        Ok(())
    }

    pub(crate) fn base(&self) -> usize {
        self.base
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The address of `[offset, offset + len)` if it lies inside the region.
    pub(crate) fn address(&self, offset: u64, len: u64) -> Option<usize> {
        let end = offset.checked_add(len)?;
        (end <= self.len as u64).then(|| self.base + offset as usize)
    }

    /// The bytes of a window the caller owns under the protocol.
    pub(crate) fn bytes(&self, offset: u64, len: u64) -> Option<&[u8]> {
        let address = self.address(offset, len)?;
        Some(unsafe { std::slice::from_raw_parts(address as *const u8, len as usize) })
    }

    /// Mutable access, to fill a window in tests.
    #[cfg(test)]
    #[allow(clippy::mut_from_ref)]
    pub(crate) fn bytes_mut(&self, offset: u64, len: u64) -> Option<&mut [u8]> {
        let address = self.address(offset, len)?;
        Some(unsafe { std::slice::from_raw_parts_mut(address as *mut u8, len as usize) })
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        self.registration.take();
        match self.owner {
            #[cfg(unix)]
            Owner::Mapped => unsafe {
                libc::munmap(self.base as *mut _, self.len);
            },
            // SAFETY: made by `map_host`, unregistered just above, referenced by nothing else.
            Owner::Anonymous => unsafe { os::unmap_host(self.base, self.len) },
            #[cfg(test)]
            Owner::Heap => unsafe { os::free_aligned(self.base, self.len) },
        }
    }
}

unsafe impl Send for Region {}
unsafe impl Sync for Region {}

/// The two regions of a session.
pub(crate) struct Regions {
    pub(crate) arena: Region,
    pub(crate) staging: Region,
}

impl Regions {
    pub(crate) fn map(shared: &wire::SharedMemory) -> Result<Self> {
        Ok(Self {
            arena: Region::map(&shared.arena_path, shared.arena_bytes)?,
            staging: Region::map(&shared.staging_path, shared.staging_bytes)?,
        })
    }

    pub(crate) fn get(&self, region: wire::Region) -> Option<&Region> {
        match crate::shared::region::Region::from_u8(region as u8)? {
            crate::shared::region::Region::Arena => Some(&self.arena),
            crate::shared::region::Region::Staging => Some(&self.staging),
        }
    }
}

/// What the first connect attached this process to. It is decided once, while registration is
/// open, and every later connect must find the same.
pub(crate) enum Attachment {
    /// The manager's shared files, mapped and registered.
    Shm {
        shared: wire::SharedMemory,
        regions: std::sync::Arc<Regions>,
    },
    /// This process's own registered staging, for a manager on another host.
    Net,
    /// Nothing was registered; copies are off for the life of the process.
    Nothing,
}
