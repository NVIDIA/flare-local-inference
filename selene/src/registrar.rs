//! The seam through which the GPU tier makes host memory addressable by its devices.

use crate::Result;

/// Registers host memory with the devices that copy into and out of it.
pub trait HostRegistrar {
    /// Register `bytes` at `address`, a page-aligned range that stays mapped
    /// and unchanged in size until the handle is dropped, so that every rank's
    /// context can address it. `context` is made current while registering.
    fn register(
        &self,
        context: usize,
        address: usize,
        bytes: usize,
    ) -> Result<Box<dyn Registration>>;

    /// Reserve one range of pinned host memory made of one part per
    /// `(context, device, bytes)` request, in request order, each accessible
    /// to its own device and the host only. The host cannot address a record
    /// across parts, so slab addresses in such a pool are identities.
    fn map_split(&self, requests: &[(usize, i32, usize)]) -> Result<Box<dyn SplitMapping>>;
}

/// An owned registration. Dropping it unregisters the range; the caller
/// unmaps the range only afterwards. A failed drain keeps the handle, and so
/// the registered range, alive.
pub trait Registration: Send {
    /// The address devices use for the start of the registered range.
    fn device_address(&self) -> u64;
}

/// An owned split mapping. Dropping it unmaps the parts and releases the range.
pub trait SplitMapping: Send {
    /// First address of the reserved range.
    fn address(&self) -> usize;
    /// Bytes of the reserved range, including each part's rounding.
    fn nbytes(&self) -> usize;
    /// Address of the part made for request `index`.
    fn part_address(&self, index: usize) -> usize;
}
