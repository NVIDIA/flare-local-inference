//! Choose physical cache capacities within reserved virtual address ranges.

use flare_selene::{CacheGeometry, Plane};
use super::{
    KvCacheLimit, LoadOptions, MemoryLimit, NativeModel, NativeResources,
    format::{Memory, Package, invalid},
};
use crate::{
    Result,
    startup::StartupProgress,
};

// Graphs are uploaded before sizing caches; leave room for driver bookkeeping
// and the execution streams and events created when the server starts.
const HEADROOM_BYTES: u64 = 64 << 20;
// A GPU that allocates from system memory (GB10) shares it with the operating system and with
// the server's host allocations made after sizing: frontend workers, tokenizers and request
// buffers took about 1.4 GiB on a DGX Spark. Caches leave room for both beyond the reserve.
use flare_selene::offload::{
    HOST_SERVICE_HEADROOM_BYTES as INTEGRATED_HOST_BYTES,
    HOST_SYSTEM_SHARE as INTEGRATED_SYSTEM_SHARE,
};

#[cfg(any(unix, test))]
pub(super) fn shared_available(meminfo: &str, total: usize) -> Result<usize> {
    Ok(flare_selene::offload::available_memory(meminfo, total)?)
}

impl NativeResources {
    /// Memory the driver reports in total for the rank's device: on an integrated GPU, the
    /// system's memory.
    pub(super) unsafe fn total_memory(&self) -> Result<usize> {
        let (mut free, mut total) = (0, 0);
        unsafe {
            self.driver
                .check("cuCtxSetCurrent", (self.driver.context_set)(self.context))?;
            self.driver.check(
                "cuMemGetInfo_v2",
                (self.driver.memory_info)(&mut free, &mut total),
            )?;
        }
        Ok(total)
    }

    pub(super) unsafe fn available_memory(&self) -> Result<usize> {
        let (mut free, mut total, mut integrated) = (0, 0, 0);
        unsafe {
            self.driver
                .check("cuCtxSetCurrent", (self.driver.context_set)(self.context))?;
            self.driver.check(
                "cuMemGetInfo_v2",
                (self.driver.memory_info)(&mut free, &mut total),
            )?;
            self.driver.check(
                "cuDeviceGetAttribute(INTEGRATED)",
                (self.driver.device_attribute)(&mut integrated, 18, self.device),
            )?;
        }
        if integrated == 0 {
            Ok(free)
        } else {
            #[cfg(windows)]
            {
                // CUDA includes both the dedicated carveout and shared system memory.
                Ok(free.min(total))
            }
            #[cfg(unix)]
            {
                let meminfo =
                    std::fs::read_to_string("/proc/meminfo").map_err(|e| e.to_string())?;
                shared_available(&meminfo, total)
            }
        }
    }
}

fn headroom(config: &serde_json::Value) -> Result<u64> {
    let reserve = config
        .get("reserve_bytes")
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| invalid("memory reserve must be a nonnegative integer"))
        })
        .transpose()?
        .unwrap_or(0);
    reserve
        .checked_add(HEADROOM_BYTES)
        .ok_or_else(|| invalid("memory reserve overflows the device budget"))
}

/// Memory caches may use on one rank: free memory less the headroom, and on an integrated GPU
/// also the server's later host allocations and a twentieth of memory for the system.
fn budget(free: u64, total: u64, headroom: u64, integrated: bool) -> u64 {
    let host = if integrated {
        INTEGRATED_HOST_BYTES + total / INTEGRATED_SYSTEM_SHARE
    } else {
        0
    };
    free.saturating_sub(headroom.saturating_add(host))
}

/// Account for loaded storage and observed module/graph growth. The device observation is
/// taken before loading model resources; on integrated GPUs it includes host allocations too.
#[derive(Clone, Copy, Debug)]
pub(super) struct MemorySnapshot {
    pub free: u64,
    pub total: u64,
}

impl NativeResources {
    pub(super) unsafe fn memory_snapshot(&self) -> Result<MemorySnapshot> {
        // Keep raw system availability for consumption deltas. Clamping both observations
        // to CUDA's smaller total can hide driver/graph allocations while RAM is plentiful.
        #[cfg(unix)]
        let free = if self.integrated {
            let info = std::fs::read_to_string("/proc/meminfo").map_err(|e| e.to_string())?;
            shared_available(&info, usize::MAX)? as u64
        } else {
            unsafe { self.available_memory()? as u64 }
        };
        #[cfg(windows)]
        let free = unsafe { self.available_memory()? as u64 };
        Ok(MemorySnapshot {
            free,
            total: unsafe { self.total_memory()? as u64 },
        })
    }
}

fn fixed_bytes(package: &Package, integrated: bool) -> Result<u64> {
    package
        .allocations
        .iter()
        .filter(|a| {
            a.sizing.is_none()
                && (matches!(a.memory, Memory::Device)
                    || integrated && matches!(a.memory, Memory::MappedHost))
        })
        .try_fold(0u64, |total, a| {
            total
                .checked_add(a.nbytes as u64)
                .ok_or_else(|| invalid("fixed model memory overflows the device budget"))
        })
}

fn limited_budget(
    memory: MemorySnapshot,
    loaded: u64,
    limit: Option<MemoryLimit>,
    default_headroom: u64,
    integrated: bool,
    future: u64,
) -> Result<u64> {
    let headroom = match limit {
        None => default_headroom,
        Some(MemoryLimit::Reserve(reserve)) => reserve
            .checked_add(HEADROOM_BYTES)
            .ok_or_else(|| invalid("memory reserve overflows the device budget"))?,
        Some(_) => HEADROOM_BYTES,
    };
    let available = budget(
        memory.free.min(memory.total),
        memory.total,
        headroom,
        integrated,
    )
    .saturating_sub(future);
    let Some(cap) = limit.and_then(|limit| limit.occupied_bytes(memory.total)) else {
        return Ok(available);
    };
    // System headroom is unoccupied memory, whereas the server's later host allocations
    // and driver allowance are part of its occupied budget.
    let required = loaded
        .checked_add(future)
        .and_then(|n| n.checked_add(HEADROOM_BYTES))
        .and_then(|n| n.checked_add(if integrated { INTEGRATED_HOST_BYTES } else { 0 }))
        .ok_or_else(|| invalid("loaded memory footprint overflows the device budget"))?;
    let remaining = cap.checked_sub(required).ok_or_else(|| invalid(format!(
        "memory limit is {cap} bytes, but loaded model resources and runtime headroom require {required} bytes"
    )))?;
    Ok(available.min(remaining))
}

/// Token limits describe useful history. Draft-page offsets are storage required in addition
/// to that history, and depend on the selected graphs' actual page-table bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PageLimit {
    minimum: u64,
    maximum: u64,
}

fn page_ceiling(
    mode: KvCacheLimit,
    context: u64,
    active: u64,
    contract: &crate::model::PreparedContract,
) -> Result<Option<PageLimit>> {
    if mode == KvCacheLimit::Memory {
        return Ok(None);
    }
    if context == 0 || contract.cache.page_tokens == 0 {
        return Err(invalid(
            "KV cache limit requires a positive context and page size",
        ));
    }
    let histories = if mode == KvCacheLimit::Concurrency {
        active
    } else {
        1
    };
    let draft = contract
        .cache
        .page_tables
        .iter()
        .filter(|view| {
            contract
                .graphs
                .iter()
                .any(|g| g.inputs.iter().any(|b| b.buffer == view.buffer))
        })
        .map(|view| view.page_offset as u64)
        .max()
        .unwrap_or(0);
    let maximum = context
        .div_ceil(contract.cache.page_tokens as u64)
        .checked_mul(histories)
        .and_then(|pages| draft.checked_mul(active).and_then(|n| pages.checked_add(n)))
        .ok_or_else(|| invalid("KV cache limit overflows page capacity"))?;
    let minimum = draft
        .checked_add(1)
        .and_then(|n| n.checked_mul(active))
        .ok_or_else(|| invalid("minimum KV storage overflows page capacity"))?;
    if minimum > maximum {
        return Err(invalid(
            "KV cache limit cannot hold one history page and required draft storage per active request",
        ));
    }
    Ok(Some(PageLimit { minimum, maximum }))
}

fn bytes(package: &Package, capacity: i32, units: u64) -> Result<u64> {
    package
        .allocations
        .iter()
        .filter_map(|a| {
            a.sizing
                .as_ref()
                .filter(|s| s.capacity == capacity)
                .map(|s| (a, s))
        })
        .try_fold(0u64, |total, (a, s)| {
            if units > s.maximum {
                return Err(invalid("cache capacity exceeds its virtual extent"));
            }
            let size = units
                .checked_mul(s.unit_bytes)
                .and_then(|n| n.checked_add(a.alignment as u64 - 1))
                .ok_or_else(|| invalid("cache allocation size overflow"))?
                / a.alignment as u64
                * a.alignment as u64;
            total
                .checked_add(size)
                .ok_or_else(|| invalid("cache allocation size overflow"))
        })
}

fn capacities(
    packages: &[Package],
    available: &[u64],
    active: u64,
    width: u64,
    page_tokens: u64,
    page_limit: Option<PageLimit>,
) -> Result<(u64, u64, u64)> {
    if packages.len() != available.len()
        || packages.is_empty()
        || !(1..=64).contains(&active)
        || !(1..=8).contains(&width)
        || page_tokens == 0
    {
        return Err(invalid("invalid cache planning geometry"));
    }
    let execution = 1 + active * width;
    let minimum_pages = page_limit.map_or(active * 8192u64.div_ceil(page_tokens), |limit| {
        limit.minimum
    });
    if minimum_pages == 0 {
        return Err(invalid("KV cache limit leaves no physical pages"));
    }
    let mut cached = active * 8;
    let mut maximum_pages = page_limit.map_or(u64::MAX, |limit| limit.maximum);
    for (rank, (package, &budget)) in packages.iter().zip(available).enumerate() {
        let rows: Vec<_> = package
            .allocations
            .iter()
            .filter_map(|a| a.sizing.as_ref().filter(|s| s.capacity == 0))
            .collect();
        let pages: Vec<_> = package
            .allocations
            .iter()
            .filter_map(|a| a.sizing.as_ref().filter(|s| s.capacity == 1))
            .collect();
        if rows.len() != 1 || pages.is_empty() {
            return Err(invalid(
                "model requires continuation-row and KV-page allocations",
            ));
        }
        let fixed = bytes(package, 0, execution)?;
        if fixed > budget {
            return Err(invalid(format!(
                "selected concurrency and verification width exceed available memory: \
                 rank {rank} needs {fixed} bytes for {execution} continuation rows, \
                 but only {budget} bytes are available"
            )));
        }
        cached = cached.min((budget - fixed) / rows[0].unit_bytes);
        if rows[0].maximum < execution {
            return Err(invalid(
                "continuation virtual extent cannot hold execution state and prefix cache",
            ));
        }
        cached = cached.min(rows[0].maximum - execution);
        maximum_pages = maximum_pages.min(pages.iter().map(|s| s.maximum).min().unwrap());
    }
    if maximum_pages < minimum_pages {
        return Err(invalid(
            "prepared virtual extent cannot hold minimum KV capacity",
        ));
    }
    let fits = |rows, pages| {
        packages.iter().zip(available).all(|(p, &budget)| {
            bytes(p, 0, rows)
                .and_then(|n| {
                    n.checked_add(bytes(p, 1, pages)?)
                        .ok_or_else(|| invalid("cache footprint overflow"))
                })
                .is_ok_and(|n| n <= budget)
        })
    };
    for (rank, (package, &budget)) in packages.iter().zip(available).enumerate() {
        let required = bytes(package, 0, execution)?
            .checked_add(bytes(package, 1, minimum_pages)?)
            .ok_or_else(|| invalid("cache footprint overflow"))?;
        if required > budget {
            return Err(invalid(format!(
                "available memory cannot hold execution state and minimum KV capacity: \
                 rank {rank} needs {required} bytes for {execution} continuation rows and \
                 {minimum_pages} KV pages, but only {budget} bytes are available"
            )));
        }
    }
    let (mut low, mut high) = (minimum_pages, maximum_pages);
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if fits(execution, mid) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    // Snapshot virtual rows alias KV backing; they do not consume another physical budget.
    for package in packages {
        let row = package
            .allocations
            .iter()
            .find(|a| a.sizing.as_ref().is_some_and(|s| s.capacity == 0))
            .unwrap();
        let unit = row.sizing.as_ref().unwrap().unit_bytes;
        let planes: Vec<_> = package
            .allocations
            .iter()
            .filter_map(|a| {
                a.sizing
                    .as_ref()
                    .filter(|s| s.capacity == 1)
                    .map(|s| (a, s))
            })
            .collect();
        let largest = planes.iter().map(|(_, s)| s.unit_bytes).max().unwrap();
        let page: u64 = planes.iter().map(|(_, s)| s.unit_bytes).sum();
        let borrowed: u64 = planes
            .iter()
            .filter(|(a, s)| {
                super::cache_memory::alias_plane(
                    s.unit_bytes,
                    largest,
                    unit,
                    page,
                    a.alignment as u64,
                )
            })
            .map(|(a, s)| (low * s.unit_bytes).div_ceil(a.alignment as u64) * a.alignment as u64)
            .sum();
        let mapped = bytes(package, 0, execution)? + borrowed;
        cached = cached.min(mapped / unit - execution);
        while cached > 0 && bytes(package, 0, execution + cached)? > mapped {
            cached -= 1;
        }
    }
    Ok((execution + cached, low, cached))
}

impl NativeModel {
    /// Refuse an occupied limit smaller than the known fixed model storage before loading it.
    pub(super) unsafe fn check_memory_limit(
        &self,
        packages: &[Package],
        options: &LoadOptions,
        initial: &[MemorySnapshot],
        embedding_host_bytes: u64,
    ) -> Result<()> {
        if !matches!(
            options.memory_limit,
            Some(MemoryLimit::Bytes(_) | MemoryLimit::Fraction(_))
        ) {
            return Ok(());
        }
        for ((rank, package), &memory) in self.ranks.iter().zip(packages).zip(initial) {
            let fixed = if rank.integrated {
                packages
                    .iter()
                    .zip(&self.ranks)
                    .filter(|(_, r)| r.integrated)
                    .try_fold(0u64, |total, (p, _)| {
                        fixed_bytes(p, true).and_then(|n| {
                            total
                                .checked_add(n)
                                .ok_or_else(|| invalid("shared fixed memory overflow"))
                        })
                    })?
            } else {
                fixed_bytes(package, false)?
            };
            let current = unsafe { rank.memory_snapshot()? };
            let loaded = fixed
                .checked_add(if rank.integrated {
                    embedding_host_bytes
                } else {
                    0
                })
                .and_then(|fixed| fixed.checked_add(memory.free.saturating_sub(current.free)))
                .ok_or_else(|| invalid("fixed memory and context footprint overflow"))?;
            limited_budget(
                current,
                loaded,
                options.memory_limit,
                headroom(&self.server_config)?,
                rank.integrated,
                0,
            )?;
        }
        Ok(())
    }

    pub(super) unsafe fn allocate_caches(
        &mut self,
        packages: &mut [Package],
        progress: &StartupProgress,
        active: usize,
        agree: &mut dyn super::model::CacheAgreement,
        options: &LoadOptions,
        initial: &[MemorySnapshot],
        embedding_host_bytes: u64,
    ) -> Result<()> {
        // Local host tiers register after cache backing; pool regions register before sizing.
        let reserve = headroom(&self.server_config)?;
        let page_tokens = self.server_config["page_tokens"]
            .as_u64()
            .ok_or_else(|| invalid("model requires physical page geometry"))?;
        let mut contract: crate::model::PreparedContract =
            serde_json::from_value(self.server_config["model_contract"].clone())
                .map_err(|error| invalid(format!("invalid prepared storage contract: {error}")))?;
        // The package also declares graphs excluded by runtime reductions. Their storage
        // requirements must not enlarge the selected cache budget.
        contract.graphs.retain(|g| {
            packages[0]
                .execution
                .as_ref()
                .unwrap()
                .graphs
                .iter()
                .any(|graph| graph.name == g.graph)
        });
        let width = packages[0]
            .execution
            .as_ref()
            .unwrap()
            .graphs
            .iter()
            .map(|graph| {
                contract
                    .graph(&graph.name)
                    .map(|g| g.state_rows_per_request as u64)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(1);
        let observations = self
            .ranks
            .iter()
            .map(|rank| unsafe { rank.memory_snapshot() })
            .collect::<Result<Vec<_>>>()?;
        let integrated = self.ranks.iter().filter(|rank| rank.integrated).count() as u64;
        let input_window = crate::offload::input_staging_bytes(&contract, |name| {
            packages[0]
                .execution
                .as_ref()?
                .buffers
                .iter()
                .find(|buffer| buffer.name == name)
                .map(|buffer| buffer.nbytes)
        })?;
        let (offload_bytes, offload_host) = if options.prefix_cache.enabled() {
            let geometry = packages
                .iter()
                .map(|package| {
                    cache_geometry(
                        package,
                        contract.cache.page_tokens as u64,
                        1 + active as u64 * width,
                        1,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let staging = input_window;
            let (device, host) = options.prefix_cache.headroom_with_inputs(
                self.ranks.len(),
                &flare_selene::Layout::new(&geometry)?,
                staging != 0,
            )?;
            let index = if staging == 0 {
                0
            } else {
                flare_selene::input::INDEX_HEADROOM_BYTES
            };
            let host = host
                .checked_add(staging)
                .and_then(|n| n.checked_add(index))
                .ok_or_else(|| invalid("prepared input staging overflows host memory"))?;
            (device, host)
        } else {
            (0, 0)
        };
        #[cfg(feature = "images")]
        let source_window = if input_window == 0 {
            0
        } else {
            crate::frontend::images::ImageLimits::default().source_retention_bytes()?
        };
        #[cfg(not(feature = "images"))]
        let source_window = 0;
        let host_future = offload_host
            .checked_add(input_window)
            .and_then(|bytes| bytes.checked_add(source_window))
            .ok_or_else(|| invalid("CPU image input windows overflow host memory"))?;
        let mut available = Vec::new();
        let mut loaded_bytes = Vec::new();
        for (index, (rank, package)) in self.ranks.iter().zip(packages.iter()).enumerate() {
            let (memory, loaded, future, share) = if rank.integrated {
                let memory = MemorySnapshot {
                    free: observations
                        .iter()
                        .zip(&self.ranks)
                        .filter(|(_, r)| r.integrated)
                        .map(|(m, _)| m.free)
                        .min()
                        .unwrap(),
                    total: observations[index].total,
                };
                let fixed = packages
                    .iter()
                    .zip(&self.ranks)
                    .filter(|(_, r)| r.integrated)
                    .try_fold(0u64, |n, (p, _)| {
                        fixed_bytes(p, true).and_then(|bytes| {
                            n.checked_add(bytes)
                                .ok_or_else(|| invalid("shared fixed memory overflow"))
                        })
                    })?;
                let fixed = fixed
                    .checked_add(embedding_host_bytes)
                    .ok_or_else(|| invalid("resident host memory overflows the shared budget"))?;
                (
                    memory,
                    fixed.max(initial[index].free.saturating_sub(memory.free)),
                    offload_bytes
                        .saturating_mul(integrated)
                        .saturating_add(host_future),
                    integrated,
                )
            } else {
                (
                    observations[index],
                    fixed_bytes(package, false)?
                        .max(initial[index].free.saturating_sub(observations[index].free)),
                    offload_bytes,
                    1,
                )
            };
            available.push(
                limited_budget(
                    memory,
                    loaded,
                    options.memory_limit,
                    reserve,
                    rank.integrated,
                    future,
                )? / share,
            );
            loaded_bytes.push(loaded / share);
        }
        let context = self.server_config["max_sequence_tokens"]
            .as_u64()
            .ok_or_else(|| invalid("model requires a prepared context ceiling"))?;
        let ceiling = page_ceiling(options.kv_cache_limit, context, active as u64, &contract)?;
        let geometry = if options.prefix_cache.pool.is_some() {
            packages
                .iter()
                .map(|package| cache_geometry(package, contract.cache.page_tokens as u64, 1, 1))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let peers = if let Some(cluster) = self.cluster() {
            agree.pool(crate::offload::shard::PoolOffer {
                enabled: options.prefix_cache.pool.is_some(),
                identity: if options.prefix_cache.pool.is_some() {
                    super::model::offload_identity(&self.server_config)?
                } else {
                    [0; 32]
                },
                layout: if geometry.is_empty() {
                    [0; 16]
                } else {
                    flare_selene::geometry::fingerprint(&geometry)
                },
                rank: cluster.rank,
                world: cluster.world,
                gpu_uuid: if options.prefix_cache.pool.is_some() {
                    crate::cuda::device_uuid(self.ranks[0].device)?
                } else {
                    [0; 16]
                },
            })?
        } else {
            None
        };
        if let Some(pool) = &options.prefix_cache.pool {
            let minimum_pages = ceiling
                .map_or(active as u64 * 8192u64.div_ceil(page_tokens), |limit| {
                    limit.minimum
                });
            let minimum = packages
                .iter()
                .map(|package| {
                    bytes(package, 0, 1 + active as u64 * width)?
                        .checked_add(bytes(package, 1, minimum_pages)?)
                        .ok_or_else(|| invalid("minimum cache footprint overflow"))
                })
                .collect::<Result<Vec<_>>>()?;
            // Establish that local serving fits even if the pool is refused.
            capacities(
                packages,
                &available,
                active as u64,
                width,
                page_tokens,
                ceiling,
            )?;
            let host = flare_selene::offload::host_pool_budget(host_future)
                .map_err(|error| error.to_string());
            let budget = crate::offload::pool::Budget {
                host,
                devices: available
                    .iter()
                    .zip(&minimum)
                    .zip(&self.ranks)
                    .map(|((&available, &minimum), rank)| {
                        (available.saturating_sub(minimum), rank.integrated)
                    })
                    .collect(),
            };
            let uuids = self
                .ranks
                .iter()
                .map(|rank| crate::cuda::device_uuid(rank.device))
                .collect::<Result<Vec<_>>>()?;
            let cluster = self.cluster().map(|cluster| cluster.world);
            let charges = crate::offload::pool::start(
                &mut self.offload,
                flare_selene::pool::PoolBuilder {
                    options: pool,
                    identity: super::model::offload_identity(&self.server_config)?,
                    geometry: &geometry,
                    gpu_uuids: uuids,
                    registration_context: self.ranks[0].context as usize,
                    budget,
                    peers,
                    disk: options.prefix_cache.disk.is_some()
                        && options.prefix_cache.disk_limit != Some(0),
                },
                progress,
                |approved| match cluster {
                    Some(world) => Ok(agree.pool_approved(approved, world)?),
                    None => Ok(true),
                },
            )?;
            for (available, charge) in available.iter_mut().zip(charges) {
                *available = available.saturating_sub(charge);
            }
        }
        let (local_rows, local_pages, _) = capacities(
            packages,
            &available,
            active as u64,
            width,
            page_tokens,
            ceiling,
        )?;
        // Fewer pages only shrink KV, and fewer rows only the snapshot slots aliased into it, so
        // the smallest capacity any rank can back fits every rank. Snapshot rows need no
        // dedicated backing; the alias layout below rejects rows the agreed pages cannot back.
        let (rows, pages) = agree.caches((local_rows, local_pages))?;
        let virtual_pages = packages
            .iter()
            .flat_map(|p| &p.allocations)
            .filter_map(|a| a.sizing.as_ref().filter(|s| s.capacity == 1))
            .map(|s| s.maximum)
            .min()
            .unwrap();
        let limiting = if ceiling.is_some_and(|limit| pages == limit.maximum) {
            options.kv_cache_limit.to_string()
        } else if pages == virtual_pages {
            "prepared extent".into()
        } else {
            "memory".into()
        };
        let gib = |values: &[u64]| {
            let format = |n: u64| format!("{:.2}", n as f64 / (1u64 << 30) as f64);
            let low = format(*values.iter().min().unwrap());
            let high = format(*values.iter().max().unwrap());
            if low == high {
                low
            } else {
                format!("{low}–{high}")
            }
        };
        let mut summary = format!(
            "per GPU: {} GiB loaded · {} GiB cache budget",
            gib(&loaded_bytes),
            gib(&available),
        );
        match options
            .memory_limit
            .unwrap_or(MemoryLimit::Reserve(reserve - HEADROOM_BYTES))
        {
            MemoryLimit::Reserve(0) => {}
            MemoryLimit::Reserve(bytes) => {
                summary.push_str(&format!(" · reserve {}", MemoryLimit::Bytes(bytes)));
            }
            MemoryLimit::Bytes(bytes) => {
                summary.push_str(&format!(" · limit {}", MemoryLimit::Bytes(bytes)));
            }
            MemoryLimit::Fraction(fraction) => {
                summary.push_str(&format!(" · limit {}% of GPU memory", fraction * 100.0));
            }
        }
        match options.kv_cache_limit {
            KvCacheLimit::Memory => {}
            KvCacheLimit::Context => summary.push_str(" · KV capped at 1 context"),
            KvCacheLimit::Concurrency => {
                summary.push_str(&format!(" · KV capped at {active} contexts"));
            }
        }
        progress.phase("Plan cache", None).finish(summary);
        let execution = 1 + active as u64 * width;
        if rows > local_rows || pages > local_pages || rows < execution {
            return Err(invalid(
                "ranks agreed on a cache capacity this rank cannot back",
            ));
        }
        let snapshots = rows - execution;
        // Scheduler page ids count the model contract's tokens per page, which
        // exceeds the physical page when pages are sharded across ranks.
        let logical_page_tokens = contract.cache.page_tokens as u64;
        let mut geometry = packages
            .iter()
            .map(|package| cache_geometry(package, logical_page_tokens, rows, pages))
            .collect::<Result<Vec<_>>>()?;
        let layouts = packages
            .iter()
            .zip(&geometry)
            .map(|(package, geometry)| {
                let alignment = |name: &str| {
                    package
                        .allocations
                        .iter()
                        .find(|a| a.name == name)
                        .unwrap()
                        .alignment as u64
                };
                super::cache_memory::Layout::new(
                    geometry,
                    rows - snapshots,
                    &geometry
                        .planes
                        .iter()
                        .map(|p| alignment(&p.name))
                        .collect::<Vec<_>>(),
                    alignment(&geometry.row.name),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut aliases = vec![u32::MAX..0; snapshots as usize];
        for layout in &layouts {
            for (union, range) in aliases.iter_mut().zip(&layout.aliases) {
                if !range.is_empty() {
                    union.start = union.start.min(range.start);
                    union.end = union.end.max(range.end);
                }
            }
        }
        for range in &mut aliases {
            if range.start >= range.end {
                *range = 0..0;
            }
        }
        for geometry in &mut geometry {
            geometry.snapshot_pages = aliases.clone();
        }
        let mut gpu_bytes = 0;
        for package in packages.iter_mut() {
            gpu_bytes += bytes(package, 0, rows - snapshots)? + bytes(package, 1, pages)?;
            for allocation in &mut package.allocations {
                if let Some(sizing) = allocation.sizing.take() {
                    let units = if sizing.capacity == 0 { rows } else { pages };
                    allocation.nbytes =
                        (units * sizing.unit_bytes).div_ceil(allocation.alignment as u64) as usize
                            * allocation.alignment;
                }
            }
            for buffer in &mut package.execution.as_mut().unwrap().buffers {
                let Some(reference) = &buffer.reference else {
                    continue;
                };
                let allocation = package
                    .allocations
                    .iter()
                    .find(|a| a.name == reference.allocation)
                    .unwrap();
                if buffer.name == "state_pool" {
                    buffer.shape[0] = rows;
                    buffer.nbytes = rows * buffer.shape[1];
                } else if contract.cache.regions.iter().any(|r| {
                    r.storage.buffer == buffer.name
                        && matches!(
                            r.kind,
                            crate::model::contract::HistoryKind::PagedKeyValue
                                | crate::model::contract::HistoryKind::PagedLatent
                        )
                }) {
                    let unit = buffer.nbytes / buffer.shape[0];
                    if reference.offset != 0
                        || pages
                            .checked_mul(unit)
                            .is_none_or(|end| end > allocation.nbytes as u64)
                    {
                        return Err(invalid("paged history view exceeds physical cache backing"));
                    }
                    buffer.shape[0] = pages;
                    buffer.nbytes = pages * unit;
                }
            }
        }
        let phase = progress.phase("Allocate cache", Some(gpu_bytes));
        for (((rank, package), geometry), layout) in self
            .ranks
            .iter_mut()
            .zip(packages)
            .zip(geometry)
            .zip(layouts)
        {
            unsafe {
                rank.allocate_shared_cache(package, &geometry, &layout)?;
                rank.cache = Some(geometry);
                let binding = package
                    .execution
                    .as_ref()
                    .unwrap()
                    .buffers
                    .iter()
                    .find(|b| b.name == "memory_capacity")
                    .filter(|b| b.dtype == "int64" && b.shape == [2] && b.nbytes == 16)
                    .ok_or_else(|| invalid("model lacks runtime memory capacity binding"))?;
                let reference = binding
                    .reference
                    .as_ref()
                    .ok_or_else(|| invalid("memory capacity lacks a storage reference"))?;
                let address = rank.allocations[&reference.allocation].address + reference.offset;
                let values = [rows, pages];
                rank.driver.check(
                    "cuMemcpyHtoD_v2",
                    (rank.driver.upload)(address, values.as_ptr().cast(), 16),
                )?;
                rank.execution = package.execution.clone();
            }
        }
        self.server_config["state_rows"] = (rows - 1).into();
        self.server_config["snapshots"] = snapshots.into();
        self.server_config["pages"] = pages.into();
        phase.finish(format!(
            "{:.1} GiB · {} KV tokens · {snapshots} shared snapshot slots · limited by {limiting}",
            gpu_bytes as f64 / (1u64 << 30) as f64,
            pages * logical_page_tokens,
        ));
        Ok(())
    }
}

/// The row plane and page planes of one rank in name order, read before
/// the sizing records are consumed. Peer allocations never carry sizing.
fn cache_geometry(
    package: &Package,
    page_tokens: u64,
    rows: u64,
    pages: u64,
) -> Result<CacheGeometry> {
    let mut row = None;
    let mut planes = Vec::new();
    for allocation in &package.allocations {
        let Some(sizing) = allocation.sizing.as_ref() else {
            continue;
        };
        let plane = Plane {
            name: allocation.name.clone(),
            address: allocation
                .address
                .ok_or_else(|| invalid("runtime-sized allocation lacks an address"))?,
            unit_bytes: sizing.unit_bytes,
        };
        if sizing.capacity == 0 {
            if row.replace(plane).is_some() {
                return Err(invalid(
                    "rank declares more than one continuation-row allocation",
                ));
            }
        } else {
            planes.push(plane);
        }
    }
    planes.sort_unstable_by(|a, b| a.name.cmp(&b.name));
    let geometry = CacheGeometry {
        snapshot_pages: Vec::new(),
        page_tokens,
        page_capacity: pages,
        row_capacity: rows,
        row: row.ok_or_else(|| invalid("rank lacks a continuation-row allocation"))?,
        planes,
    };
    geometry.validate()?;
    Ok(geometry)
}

#[cfg(test)]
mod tests {
    use super::super::{
        format::{Allocation, Context, Memory},
        protobuf::wire::Sizing,
    };
    use super::*;

    #[test]
    fn integrated_memory_includes_reclaimable_ram_but_not_swap() {
        let info = "MemFree: 1024 kB\nMemAvailable: 8388608 kB\nSwapFree: 16777216 kB\n";
        assert_eq!(shared_available(info, 16 << 30).unwrap(), 8 << 30);
        assert_eq!(shared_available(info, 4 << 30).unwrap(), 4 << 30);
        for invalid in [
            "MemFree: 1024 kB\nSwapFree: 16777216 kB\n",
            "MemAvailable: -1 kB",
            "MemAvailable: 1024 MB",
            "MemAvailable: 18446744073709551615 kB",
        ] {
            assert!(shared_available(invalid, usize::MAX).is_err());
        }
    }

    #[test]
    fn serving_headroom_preserves_the_user_reserve() {
        assert_eq!(headroom(&serde_json::json!({})).unwrap(), 64 << 20);
        assert_eq!(
            headroom(&serde_json::json!({"reserve_bytes": 2u64 << 30})).unwrap(),
            (2 << 30) + (64 << 20)
        );
        for reserve in [
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!("0"),
            serde_json::json!(null),
            serde_json::json!(u64::MAX),
        ] {
            assert!(headroom(&serde_json::json!({"reserve_bytes": reserve})).is_err());
        }
    }

    #[test]
    fn integrated_gpus_keep_host_memory_for_the_server_beyond_the_reserve() {
        let gib = 1u64 << 30;
        assert_eq!(budget(40 * gib, 120 * gib, 5 * gib, false), 35 * gib);
        // Two GiB for the server and a twentieth of memory for the system.
        assert_eq!(budget(40 * gib, 120 * gib, 5 * gib, true), 27 * gib);
        assert_eq!(budget(40 * gib, 120 * gib, 0, true), 32 * gib);
        assert_eq!(budget(6 * gib, 120 * gib, 0, true), 0);
    }

    #[test]
    fn occupied_limits_include_loaded_resources_and_use_total_capacity() {
        let gib = 1u64 << 30;
        let memory = MemorySnapshot {
            free: 40 * gib,
            total: 100 * gib,
        };
        let legacy = 5 * gib + HEADROOM_BYTES;
        assert_eq!(
            limited_budget(memory, 20 * gib, None, legacy, false, 0).unwrap(),
            35 * gib - HEADROOM_BYTES
        );
        // An explicit flag replaces the package reserve, including for occupied limits.
        assert_eq!(
            limited_budget(
                memory,
                20 * gib,
                Some(MemoryLimit::Bytes(30 * gib)),
                legacy,
                false,
                gib
            )
            .unwrap(),
            9 * gib - HEADROOM_BYTES
        );
        assert_eq!(
            limited_budget(
                memory,
                20 * gib,
                Some(MemoryLimit::Fraction(0.5)),
                legacy,
                false,
                0
            )
            .unwrap(),
            30 * gib - HEADROOM_BYTES
        );
        assert_eq!(
            limited_budget(
                memory,
                20 * gib,
                Some(MemoryLimit::Reserve(10 * gib)),
                legacy,
                false,
                0
            )
            .unwrap(),
            30 * gib - HEADROOM_BYTES
        );
        // Other processes still bound the physical supply, even when our cap is larger.
        let busy = MemorySnapshot {
            free: gib,
            ..memory
        };
        assert_eq!(
            limited_budget(
                busy,
                20 * gib,
                Some(MemoryLimit::Bytes(80 * gib)),
                legacy,
                false,
                0
            )
            .unwrap(),
            gib - HEADROOM_BYTES
        );
        assert!(
            limited_budget(
                memory,
                30 * gib,
                Some(MemoryLimit::Bytes(30 * gib)),
                legacy,
                false,
                0
            )
            .is_err()
        );
        assert!(
            limited_budget(
                memory,
                0,
                Some(MemoryLimit::Reserve(u64::MAX)),
                legacy,
                false,
                0
            )
            .is_err()
        );
    }

    #[test]
    fn integrated_budgets_charge_later_host_allocations_once() {
        let gib = 1u64 << 30;
        let memory = MemorySnapshot {
            free: 60 * gib,
            total: 100 * gib,
        };
        // 20 GiB loaded + 2 GiB server host allowance + 8 GiB offload + driver headroom.
        assert_eq!(
            limited_budget(
                memory,
                20 * gib,
                Some(MemoryLimit::Bytes(40 * gib)),
                HEADROOM_BYTES,
                true,
                8 * gib
            )
            .unwrap(),
            10 * gib - HEADROOM_BYTES
        );
        // Without an occupied cap, retain the five-percent system margin too.
        assert_eq!(
            limited_budget(memory, 20 * gib, None, HEADROOM_BYTES, true, 8 * gib).unwrap(),
            45 * gib - HEADROOM_BYTES
        );
        let plentiful = MemorySnapshot {
            free: 120 * gib,
            ..memory
        };
        assert_eq!(
            limited_budget(plentiful, 0, None, HEADROOM_BYTES, true, 0).unwrap(),
            93 * gib - HEADROOM_BYTES
        );
    }

    #[test]
    fn history_limits_round_each_context_and_use_bound_page_table_offsets() {
        use crate::model::contract::{HistoryKind, PageTableView};
        let mut contract = crate::model::testing::metadata();
        contract.cache.page_tokens = 128;
        contract.cache.page_tables = vec![
            PageTableView {
                buffer: contract.graphs[0].inputs[0].buffer.clone(),
                page_offset: 2,
            },
            PageTableView {
                buffer: "unused-draft-table".into(),
                page_offset: 999,
            },
        ];
        // Cache storage semantics do not select a different sizing policy.
        for kind in [
            HistoryKind::PagedKeyValue,
            HistoryKind::PagedLatent,
            HistoryKind::SlidingWindow,
            HistoryKind::RecurrentState,
        ] {
            for region in &mut contract.cache.regions {
                region.kind = kind;
            }
            assert_eq!(
                page_ceiling(KvCacheLimit::Memory, 1025, 3, &contract).unwrap(),
                None
            );
            assert_eq!(
                page_ceiling(KvCacheLimit::Context, 1025, 3, &contract).unwrap(),
                Some(PageLimit {
                    minimum: 9,
                    maximum: 15
                })
            );
            assert_eq!(
                page_ceiling(KvCacheLimit::Concurrency, 1025, 3, &contract).unwrap(),
                Some(PageLimit {
                    minimum: 9,
                    maximum: 33
                })
            );
        }
        let mut overflow = contract.clone();
        overflow.cache.page_tokens = 1;
        assert!(page_ceiling(KvCacheLimit::Concurrency, u64::MAX, 64, &overflow).is_err());
        assert!(page_ceiling(KvCacheLimit::Context, 0, 3, &contract).is_err());
        assert!(page_ceiling(KvCacheLimit::Context, 1, 64, &contract).is_err());
    }

    fn package(row_bytes: u64, page_bytes: u64, maximum_pages: u64) -> Package {
        Package {
            resources: std::collections::HashMap::new(),
            events: vec![],
            architecture: [12, 0],
            modules: vec![],
            functions: vec![],
            graphs: vec![],
            reservations: vec![],
            execution: None,
            context: Context::default(),
            allocations: [
                ("state", 0, row_bytes, 4096),
                ("kv", 1, page_bytes, maximum_pages),
            ]
            .into_iter()
            .map(|(name, capacity, unit_bytes, maximum)| Allocation {
                name: name.into(),
                nbytes: 0,
                alignment: 65536,
                file: None,
                data_spans: vec![],
                sources: vec![],
                relocations: vec![],
                address: Some(0x200000000000),
                memory: Memory::Device,
                sizing: Some(Sizing {
                    capacity,
                    unit_bytes,
                    maximum,
                }),
            })
            .collect(),
        }
    }

    #[test]
    fn cache_geometry_orders_named_planes_independently_of_allocation_traversal() {
        let mut p = package(100 << 20, 2 << 20, 100000);
        p.allocations.push(Allocation {
            name: "kv-draft".into(),
            nbytes: 0,
            alignment: 65536,
            file: None,
            data_spans: vec![],
            sources: vec![],
            relocations: vec![],
            address: Some(0x300000000000),
            memory: Memory::Device,
            sizing: Some(Sizing {
                capacity: 1,
                unit_bytes: 4096,
                maximum: 100000,
            }),
        });
        p.allocations.push(Allocation {
            name: "peer-0".into(),
            nbytes: 1 << 20,
            alignment: 65536,
            file: None,
            data_spans: vec![],
            sources: vec![],
            relocations: vec![],
            address: Some(0x400000000000),
            memory: Memory::Peer,
            sizing: None,
        });
        let geometry = cache_geometry(&p, 256, 40, 5000).unwrap();
        assert_eq!(geometry.page_tokens, 256);
        assert_eq!((geometry.row_capacity, geometry.page_capacity), (40, 5000));
        assert_eq!(geometry.row.name, "state");
        assert_eq!(geometry.row.unit_bytes, 100 << 20);
        assert_eq!(
            geometry
                .planes
                .iter()
                .map(|plane| (plane.name.as_str(), plane.unit_bytes))
                .collect::<Vec<_>>(),
            [("kv", 2 << 20), ("kv-draft", 4096)]
        );
        assert_eq!(geometry.planes[1].address, 0x300000000000);
        p.allocations.reverse();
        let reordered = cache_geometry(&p, 256, 40, 5000).unwrap();
        assert_eq!(geometry, reordered);
        assert!(geometry.same_layout(&reordered));
        let original_ranks = vec![geometry.clone(); 4];
        let mut reordered_ranks = original_ranks.clone();
        reordered_ranks[3] = reordered;
        assert_eq!(
            flare_selene::geometry::fingerprint(&original_ranks),
            flare_selene::geometry::fingerprint(&reordered_ranks),
        );
        let mut options = package(4096, 4096, 16);
        assert_eq!(
            flare_selene::geometry::fingerprint(&[cache_geometry(&options, 64, 8, 16).unwrap()]),
            flare_selene::geometry::fingerprint(
                &[cache_geometry(&options, 64, 80, 160).unwrap()]
            )
        );
        options.allocations.push(Allocation {
            name: "state-2".into(),
            nbytes: 0,
            alignment: 65536,
            file: None,
            data_spans: vec![],
            sources: vec![],
            relocations: vec![],
            address: Some(0x500000000000),
            memory: Memory::Device,
            sizing: Some(Sizing {
                capacity: 0,
                unit_bytes: 4096,
                maximum: 16,
            }),
        });
        assert!(cache_geometry(&options, 64, 8, 16).is_err());
    }

    #[test]
    fn pool_registration_reduces_kv_supply_within_occupied_and_reserve_limits() {
        let gib = 1u64 << 30;
        let packages = [package(4096, 1 << 20, 65536)];
        let memory = MemorySnapshot {
            free: 20 * gib,
            total: 32 * gib,
        };
        for limit in [
            None,
            Some(MemoryLimit::Bytes(16 * gib)),
            Some(MemoryLimit::Reserve(4 * gib)),
            Some(MemoryLimit::Fraction(0.5)),
        ] {
            let available =
                limited_budget(memory, 4 * gib, limit, HEADROOM_BYTES, false, 0).unwrap();
            let before = capacities(&packages, &[available], 1, 1, 64, None).unwrap();
            let budget = crate::offload::pool::Budget {
                host: Ok(32 * gib),
                devices: vec![(available, false)],
            };
            let charges = budget.charges(32 * gib, (32 * gib) / 512);
            let after = capacities(&packages, &[available - charges[0]], 1, 1, 64, None).unwrap();
            assert!(after.1 < before.1);
            assert!(after.1 >= 8192 / 64);
        }
    }

    #[test]
    fn lower_concurrency_and_verification_move_physical_budget_to_kv() {
        let p = [package(100 << 20, 2 << 20, 100000)];
        let budget = [80 << 30];
        let full = capacities(&p, &budget, 32, 6, 64, None).unwrap();
        let reduced = capacities(&p, &budget, 4, 6, 64, None).unwrap();
        let target_only = capacities(&p, &budget, 4, 1, 64, None).unwrap();
        assert!(full.0 > reduced.0 && reduced.0 > target_only.0);
        assert!(full.1 < reduced.1 && reduced.1 < target_only.1);
        assert_eq!(reduced.0, 1 + 4 * 6 + reduced.2);
        for (rows, pages, snapshots) in [full, reduced, target_only] {
            let used = bytes(&p[0], 0, rows - snapshots).unwrap() + bytes(&p[0], 1, pages).unwrap();
            assert!(used <= budget[0] && budget[0] - used < 2 << 20);
        }
        // Capture concurrency is deliberately absent from the allocation policy.
        assert_eq!(reduced, capacities(&p, &budget, 4, 6, 64, None).unwrap());
    }

    #[test]
    fn ranks_share_capacities_and_respect_alignment_and_virtual_extents() {
        let p = [package(123456, 45678, 1000), package(246912, 91356, 1000)];
        let (rows, pages, snapshots) =
            capacities(&p, &[256 << 20, 80 << 20], 4, 3, 256, None).unwrap();
        assert!(pages <= 1000);
        for (p, budget) in p.iter().zip([256 << 20, 80 << 20]) {
            assert!(bytes(p, 0, rows - snapshots).unwrap() + bytes(p, 1, pages).unwrap() <= budget);
        }
        assert!(capacities(&p, &[256 << 20, 1 << 20], 4, 3, 256, None).is_err());
        let (_, pages, _) =
            capacities(&[package(1024, 1024, 1024)], &[1 << 30], 1, 1, 64, None).unwrap();
        assert_eq!(pages, 1024);
    }

    #[test]
    fn capped_pools_can_shrink_below_the_default_floor_without_exceeding_the_ceiling() {
        let packages = [
            package(65536, 1 << 20, 10000),
            package(65536, 2 << 20, 10000),
        ];
        let limit = Some(PageLimit {
            minimum: 4,
            maximum: 16,
        });
        for width in [1, 6] {
            let (rows, pages, snapshots) =
                capacities(&packages, &[1 << 30, 1 << 30], 4, width, 64, limit).unwrap();
            assert_eq!(pages, 16);
            assert_eq!(rows - snapshots, 1 + 4 * width);
            // A busy TP rank reduces the common pool. Context mode still starts with fewer
            // pages than a full context when the usable memory budget requires it.
            let (_, pages, _) =
                capacities(&packages, &[1 << 30, 20 << 20], 4, width, 64, limit).unwrap();
            assert!((4..16).contains(&pages));
        }
        assert!(capacities(&packages, &[1 << 30, 1 << 20], 4, 6, 64, limit).is_err());
        assert!(capacities(&packages, &[1 << 30, 20 << 20], 4, 6, 64, None).is_err());
    }

    #[test]
    #[ignore = "requires an assigned CUDA_VISIBLE_DEVICES GPU"]
    fn cuda_memory_limits_account_for_context_and_allocations() {
        use super::super::driver::Driver;
        let driver = Driver::load().unwrap();
        unsafe {
            let initial = NativeResources::memory_before_context(0).unwrap();
            let mut device = 0;
            driver
                .check("cuDeviceGet", (driver.device)(&mut device, 0))
                .unwrap();
            let mut p = package(65536, 65536, 1024);
            for (value, attribute) in p.architecture.iter_mut().zip([75, 76]) {
                driver
                    .check(
                        "cuDeviceGetAttribute",
                        (driver.device_attribute)(value, attribute, device),
                    )
                    .unwrap();
            }
            p.allocations.truncate(1);
            p.allocations[0].sizing = None;
            p.allocations[0].address = None;
            p.allocations[0].nbytes = 8 << 20;
            let mut owner = NativeResources::initialize(&p, 0).unwrap();
            let phase = StartupProgress::silent().phase("Test memory", None);
            owner.allocate(&p, &phase).unwrap();
            let memory = owner.memory_snapshot().unwrap();
            assert_eq!(memory.total, initial.total);
            let loaded = fixed_bytes(&p, owner.integrated)
                .unwrap()
                .max(initial.free.saturating_sub(memory.free));
            assert!(
                limited_budget(
                    memory,
                    loaded,
                    Some(MemoryLimit::Bytes(loaded)),
                    0,
                    owner.integrated,
                    0
                )
                .is_err()
            );
            let cap = loaded
                + HEADROOM_BYTES
                + (16 << 20)
                + if owner.integrated {
                    INTEGRATED_HOST_BYTES
                } else {
                    0
                };
            assert_eq!(
                limited_budget(
                    memory,
                    loaded,
                    Some(MemoryLimit::Bytes(cap)),
                    0,
                    owner.integrated,
                    0
                )
                .unwrap(),
                16 << 20
            );
        }
    }
}
