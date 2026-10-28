//! Host retention of published prefix checkpoints in registered memory and on disk.
//!
//! A GPU prefix checkpoint owns its tail pages, the complete cache pages since its
//! parent, and one continuation-state row. The host store copies those bytes out
//! at publication and copies them back into free GPU
//! ids when a later request matches a checkpoint the GPU no longer holds. The
//! restored bytes are published as an ordinary GPU checkpoint, so the model
//! worker, the scheduler, and the recipes see no difference between a checkpoint
//! produced by prefill and one restored from the host.
//!
#[cfg(all(test, feature = "runtime-package"))]
mod gpu_tests;
#[cfg(feature = "runtime-package")]
pub(crate) mod pool;
mod registrar;
pub mod shard;
pub mod transfer;

pub use flare_selene::pool::{PoolOptions, parse_pool_address};
use flare_selene::{CacheGeometry, HostStore};

use crate::Result;

pub use flare_selene::offload::{
    Binding, PrefixCacheOffload, Preparation, Prepared, Resolved, Shared, input_memory_bytes,
    parse_drain_on_shutdown, parse_size,
};

pub fn prepare(
    options: &PrefixCacheOffload,
    package: &std::path::Path,
    identity: [u8; 32],
    geometry: Vec<CacheGeometry>,
    contexts: Vec<(usize, i32)>,
) -> Result<Preparation> {
    prepare_with_inputs(options, package, identity, geometry, contexts, 0, None)
}

pub fn prepare_with_inputs(
    options: &PrefixCacheOffload,
    package: &std::path::Path,
    identity: [u8; 32],
    geometry: Vec<CacheGeometry>,
    contexts: Vec<(usize, i32)>,
    input_staging: u64,
    shard: Option<(usize, usize)>,
) -> Result<Preparation> {
    Ok(flare_selene::offload::prepare_with_inputs(
        options,
        package,
        identity,
        geometry,
        contexts,
        input_staging,
        shard,
        &registrar::CudaRegistrar,
    )?)
}

pub fn open(prepared: Prepared) -> Result<Box<dyn HostStore>> {
    Ok(flare_selene::offload::open(prepared, &transfer::Copiers)?)
}

/// One bounded ordinary-memory window for serialized prepared inputs and disk-read leases.
pub fn input_staging_bytes(
    contract: &crate::model::PreparedContract,
    buffer_bytes: impl Fn(&str) -> Option<u64>,
) -> Result<u64> {
    use crate::model::contract::ValueSource;
    let mut maximum = 0;
    for graph in &contract.graphs {
        let mut bytes = 0u64;
        for binding in graph
            .inputs
            .iter()
            .filter(|binding| binding.source == ValueSource::ImageTensor)
        {
            let input =
                buffer_bytes(&binding.buffer).ok_or("prepared image input buffer is absent")?;
            bytes = bytes
                .checked_add(input)
                .ok_or("prepared image input extent overflows")?;
        }
        maximum = maximum.max(bytes);
    }
    if maximum == 0 {
        Ok(0)
    } else {
        maximum
            .checked_add(4096)
            .ok_or_else(|| "prepared input staging extent overflows".into())
    }
}

#[derive(Clone, Copy)]
pub enum LookupOutcome {
    NoMatch,
    CostRejected,
    CapacityRejected,
    EnqueueRejected,
    Admitted,
}
