//! Registration of host memory with the CUDA driver for the selene host tiers.

use flare_selene::{HostRegistrar, Registration, Result, SplitMapping};

use crate::cuda::{self, DeviceHostParts, HostRegistration};

pub(super) struct CudaRegistrar;

impl HostRegistrar for CudaRegistrar {
    fn register(
        &self,
        context: usize,
        address: usize,
        bytes: usize,
    ) -> Result<Box<dyn Registration>> {
        // SAFETY: The slab pool that asks keeps the range mapped at this size
        // until it drops the returned handle, which unregisters it first.
        let registration = unsafe { cuda::register_host(context, address, bytes) }?;
        Ok(Box::new(registration))
    }

    fn map_split(&self, requests: &[(usize, i32, usize)]) -> Result<Box<dyn SplitMapping>> {
        Ok(Box::new(cuda::map_device_host_parts(requests)?))
    }
}

impl Registration for HostRegistration {
    fn device_address(&self) -> u64 {
        HostRegistration::device_address(self)
    }
}

impl SplitMapping for DeviceHostParts {
    fn address(&self) -> usize {
        DeviceHostParts::address(self)
    }

    fn nbytes(&self) -> usize {
        DeviceHostParts::nbytes(self)
    }

    fn part_address(&self, index: usize) -> usize {
        DeviceHostParts::part_address(self, index)
    }
}
