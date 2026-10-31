/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA registration and progress display for Selene startup.
use super::registrar;
use crate::{Result, startup::StartupProgress};
pub(crate) use flare_selene::pool::Budget;
use flare_selene::pool::PoolBuilder;
use std::sync::Arc;

pub(crate) fn start(
    startup: &mut flare_selene::offload::Startup,
    builder: PoolBuilder<'_>,
    progress: &StartupProgress,
    approve: impl FnOnce(bool) -> flare_selene::Result<bool>,
) -> Result<Vec<u64>> {
    let phase = progress.phase("Wait for prefix cache pool", None);
    let (summary, charges) =
        startup.start_pool(builder, Arc::new(registrar::CudaRegistrar), approve)?;
    phase.finish(summary);
    Ok(charges)
}
#[cfg(all(test, unix))]
mod tests;
