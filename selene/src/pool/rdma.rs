//! The instance's end of an RDMA data channel (experimental).
//!
//! The manager does every transfer: it writes a claim's windows into slots of this
//! process's private staging, with one-sided RDMA on
//! a reliable-connected queue pair made for the transfer. This side only registers
//! its staging once, makes a queue pair per transfer, brings it up against the
//! manager's description and tears it down; it posts no work. Tearing it down is
//! the fence: moving the pair to the error state and destroying it stops every
//! packet the manager might still send for it, so nothing lands in a slot after it
//! is gone.
//!
pub use crate::shared::rdma::Handshake;
use std::{io, sync::Arc};

#[cfg(not(feature = "rdma"))]
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

#[cfg(feature = "rdma")]
mod verbs {
    use super::*;
    use crate::shared::rdma::{self, Qp, ffi};

    pub struct Device {
        device: Arc<rdma::Device>,
        mr: *mut ffi::ibv_mr,
        base: u64,
    }

    // SAFETY: the registration is immutable and libibverbs allows concurrent use.
    unsafe impl Send for Device {}
    // SAFETY: as above.
    unsafe impl Sync for Device {}

    impl Device {
        /// Register private staging on the first active device.
        ///
        /// # Safety
        /// The range must remain mapped at this address until the device is dropped.
        pub unsafe fn open(base: *mut u8, len: usize) -> io::Result<Arc<Self>> {
            let device = rdma::Device::open()?;
            let access = ffi::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0
                | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0;
            // SAFETY: the caller keeps the range mapped for this registration's lifetime.
            let mr = unsafe { ffi::ibv_reg_mr(device.pd, base.cast(), len, access as i32) };
            if mr.is_null() {
                return Err(io::Error::last_os_error());
            }
            Ok(Arc::new(Self {
                device,
                mr,
                base: base as u64,
            }))
        }
    }

    impl Drop for Device {
        fn drop(&mut self) {
            // SAFETY: every pair has released or quarantined its registration owner.
            unsafe { ffi::ibv_dereg_mr(self.mr) };
        }
    }

    pub struct Pair {
        qp: Qp,
        device: Arc<Device>,
    }

    impl Pair {
        pub fn new(device: &Arc<Device>) -> io::Result<Self> {
            let access = ffi::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0
                | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0;
            Ok(Self {
                qp: Qp::new(&device.device, 1, 16, access)?,
                device: device.clone(),
            })
        }

        pub fn handshake(&self) -> Handshake {
            // SAFETY: this pair retains the registration.
            self.qp
                .handshake(unsafe { (*self.device.mr).rkey }, self.device.base)
        }

        pub fn connect(&self, peer: &Handshake) -> io::Result<()> {
            self.qp.connect(peer)
        }

        pub fn close(&mut self) -> io::Result<()> {
            self.qp.close()
        }
    }

    impl Drop for Pair {
        fn drop(&mut self) {
            if self.close().is_err() {
                // Retain the registration with the unproven QP.
                std::mem::forget(self.device.clone());
            }
        }
    }
}

#[cfg(feature = "rdma")]
pub use verbs::{Device, Pair};

/// Without the `rdma` feature there is no device: opening one says so.
#[cfg(not(feature = "rdma"))]
mod absent {
    use super::*;

    pub struct Device;

    impl Device {
        /// # Safety
        ///
        /// Never registers anything.
        pub unsafe fn open(_: *mut u8, _: usize) -> io::Result<Arc<Self>> {
            Err(invalid("this build has no RDMA support"))
        }
    }

    pub struct Pair;

    impl Pair {
        pub fn new(_: &Arc<Device>) -> io::Result<Self> {
            Err(invalid("this build has no RDMA support"))
        }

        pub fn handshake(&self) -> Handshake {
            unreachable!("no pair exists without RDMA support")
        }

        pub fn connect(&self, _: &Handshake) -> io::Result<()> {
            unreachable!("no pair exists without RDMA support")
        }

        pub fn close(&mut self) -> io::Result<()> {
            unreachable!("no pair exists without RDMA support")
        }
    }
}

#[cfg(not(feature = "rdma"))]
pub use absent::{Device, Pair};
