//! RoCEnante's RDMA proxy for packages whose ranks run on separate hosts.
//!
//! A rank's graphs launch b12x's RoCE kernels against a pinned region restored at its recorded
//! address. This side opens the devices of the fabric this host discovered (`roce_fabric`),
//! registers that region, exchanges queue-pair records with every peer, connects each peer's
//! rails, and runs the posting thread (`native/roce_proxy.c`, the proxy the kernels' ABI was
//! compiled against).

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::atomic::{AtomicU32, Ordering};

use super::roce_fabric::Topology;
use super::roce_fabric::Shape;
use crate::runtime_package::RoceSpec;
use crate::{ExecutionError, Failure, Result};

/// The proxy ABI compiled into this runtime; packages record the one their kernels use.
pub const PROXY_ABI: u32 = 10;
/// Control word 7 after a variable-length all-gather met a byte count it cannot send.
const INVALID_VARLEN_COUNT: u32 = 1;

#[repr(C)]
struct Context {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn roce_abi_version() -> c_int;
    fn roce_layout(
        world: c_int,
        slot_bytes: u64,
        pipe_bytes: u64,
        varlen_bytes: u64,
        out: *mut u64,
    ) -> c_int;
    fn roce_blob_bytes() -> u64;
    fn roce_create(
        world: c_int,
        rank: c_int,
        hca_names: *const *const c_char,
        n_hca: c_int,
        gid_index: c_int,
        traffic_class: c_int,
        region: *mut c_void,
        region_bytes: u64,
        slot_bytes: u64,
        pipe_bytes: u64,
        varlen_bytes: u64,
        err: *mut c_char,
        err_len: u64,
    ) -> *mut Context;
    fn roce_enable_ring(c: *mut Context) -> c_int;
    fn roce_enable_ring_open(c: *mut Context) -> c_int;
    fn roce_local_blob(c: *mut Context, out: *mut c_void, out_len: u64) -> c_int;
    fn roce_connect(
        c: *mut Context,
        blobs: *const c_void,
        blobs_len: u64,
        routes: *const i32,
        n_rail: c_int,
    ) -> c_int;
    fn roce_start(c: *mut Context) -> c_int;
    fn roce_failed(c: *mut Context) -> c_int;
    fn roce_error(c: *mut Context) -> *const c_char;
    fn roce_destroy(c: *mut Context);
}

fn failure(message: impl Into<String>) -> crate::Error {
    ExecutionError::Poisoned(Failure::new(message.into())).into()
}

fn peer_blobs(records: &[Vec<u8>], world: usize) -> Result<Vec<u8>> {
    // ABI 10 changes startup-owned ring mailboxes, not the captured kernel region.
    // Older packages can use the new proxy, but all live peers must use its wire layout.
    let size = unsafe { roce_blob_bytes() } as usize;
    if records.len() != world
        || records
            .iter()
            .any(|record| record.len() != size + 4 || record[..4] != PROXY_ABI.to_le_bytes())
    {
        return Err(failure(
            "RoCE queue-pair records must cover every rank with the same native proxy ABI",
        ));
    }
    Ok(records
        .iter()
        .flat_map(|record| record[4..].iter().copied())
        .collect())
}

/// One rank's connected transport. Dropping it stops the proxy and releases the HCAs; the region
/// itself belongs to the package.
pub struct RoceTransport {
    context: *mut Context,
    spec: RoceSpec,
    topology: Topology,
    // The control record: words 2, 3 and 6 name a failed wait's sequence, peer and rail; word 7
    // why a kernel stopped without a timeout.
    control: *const AtomicU32,
}

// SAFETY: The proxy context is driven only through its thread-safe C entry points, and the error
// words are read atomically.
unsafe impl Send for RoceTransport {}
unsafe impl Sync for RoceTransport {}

impl RoceTransport {
    /// Open the topology's devices and register the restored region.
    ///
    /// # Safety
    /// `spec.region_address` must name `spec.region_bytes` of mapped, locked host memory that the
    /// package restored for this rank and that outlives the transport; no graph may run on it
    /// before `connect` returns.
    pub unsafe fn open(spec: &RoceSpec, topology: &Topology) -> Result<Self> {
        let abi = unsafe { roce_abi_version() } as u32;
        // ABI 9 and 10 have identical kernel-visible layouts and control operations.
        // ABI 10's wider CPU ring window is allocated by this runtime at startup.
        if !matches!(spec.proxy_abi, 9 | 10) || abi != PROXY_ABI {
            return Err(failure(format!(
                "package RoCE kernels use proxy ABI {}, this runtime provides {abi}",
                spec.proxy_abi
            )));
        }
        let mut layout = [0u64; 17];
        let status = unsafe {
            roce_layout(
                spec.world as c_int,
                spec.slot_bytes,
                spec.pipe_bytes,
                spec.varlen_bytes,
                layout.as_mut_ptr(),
            )
        };
        if status != 0 || layout[3] != spec.ctrl_offset || layout[4] != spec.region_bytes {
            return Err(failure(
                "package RoCE region differs from the proxy's layout for its geometry",
            ));
        }
        if (topology.shape.relays() && !(2..=16).contains(&spec.world))
            || topology.routes.len() != spec.world
            || !(1..=super::roce_fabric::MAX_RAILS).contains(&topology.rails)
        {
            return Err(failure(format!(
                "RoCE topology does not match the package's ranks: {} routes and {} rails for {} ranks",
                topology.routes.len(),
                topology.rails,
                spec.world
            )));
        }
        for (peer, links) in topology.routes.iter().enumerate() {
            let absent = !topology.shape.linked(spec.rank, peer, spec.world);
            if absent != links.is_empty()
                || (!absent && links.len() != topology.rails)
                || links
                    .iter()
                    .any(|(local, _)| *local as usize >= topology.hcas.len())
            {
                return Err(failure(format!(
                    "RoCE topology does not match the package's ranks: rank {} has {} links to rank {peer},                      expected {} ({} rails, {} HCAs)",
                    spec.rank,
                    links.len(),
                    if absent { 0 } else { topology.rails },
                    topology.rails,
                    topology.hcas.len()
                )));
            }
        }
        let names = topology
            .hcas
            .iter()
            .map(|name| CString::new(name.as_str()))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| failure("RoCE HCA name contains a NUL byte"))?;
        let pointers: Vec<*const c_char> = names.iter().map(|name| name.as_ptr()).collect();
        let region = spec.region_address as *mut c_void;
        // Flags and the control record must start at zero: every rank begins again at sequence 1.
        unsafe { std::ptr::write_bytes(region.cast::<u8>(), 0, spec.region_bytes as usize) };
        let mut message = [0 as c_char; 512];
        let context = unsafe {
            roce_create(
                spec.world as c_int,
                spec.rank as c_int,
                pointers.as_ptr(),
                pointers.len() as c_int,
                topology.gid_index as c_int,
                topology.traffic_class as c_int,
                region,
                spec.region_bytes,
                spec.slot_bytes,
                spec.pipe_bytes,
                spec.varlen_bytes,
                message.as_mut_ptr(),
                message.len() as u64,
            )
        };
        if context.is_null() {
            let text = unsafe { CStr::from_ptr(message.as_ptr()) }.to_string_lossy();
            return Err(failure(format!("RoCE transport setup failed: {text}")));
        }
        let enabled = match topology.shape {
            Shape::Mesh => 0,
            Shape::Ring => unsafe { roce_enable_ring(context) },
            Shape::Chain => unsafe { roce_enable_ring_open(context) },
        };
        if enabled != 0 {
            let message = unsafe { CStr::from_ptr(roce_error(context)) }
                .to_string_lossy()
                .into_owned();
            unsafe { roce_destroy(context) };
            return Err(failure(format!("RoCE ring setup failed: {message}")));
        }
        Ok(Self {
            context,
            spec: spec.clone(),
            topology: topology.clone(),
            control: (spec.region_address + spec.ctrl_offset) as *const AtomicU32,
        })
    }

    pub fn spec(&self) -> &RoceSpec {
        &self.spec
    }

    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    /// This rank's queue-pair record, sent to every peer.
    pub fn local_record(&self) -> Result<Vec<u8>> {
        let mut record = vec![0u8; unsafe { roce_blob_bytes() } as usize];
        let status = unsafe {
            roce_local_blob(
                self.context,
                record.as_mut_ptr().cast(),
                record.len() as u64,
            )
        };
        if status != 0 {
            return Err(failure("RoCE queue-pair record does not fit its buffer"));
        }
        let mut versioned = Vec::with_capacity(record.len() + 4);
        versioned.extend_from_slice(&PROXY_ABI.to_le_bytes());
        versioned.extend_from_slice(&record);
        Ok(versioned)
    }

    /// Connect every peer's rails from all ranks' records, in rank order, and start posting.
    pub fn connect(&mut self, records: &[Vec<u8>]) -> Result<()> {
        let joined = peer_blobs(records, self.spec.world)?;
        let rails = self.topology.rails;
        let mut routes = Vec::with_capacity(self.spec.world * rails * 2);
        for (peer, links) in self.topology.routes.iter().enumerate() {
            for rail in 0..rails {
                let (local, remote) =
                    if peer == self.spec.rank || self.topology.shape.relays() && links.is_empty() {
                        (0, 0)
                    } else {
                        links[rail]
                    };
                routes.push(local as i32);
                routes.push(remote as i32);
            }
        }
        let status = unsafe {
            roce_connect(
                self.context,
                joined.as_ptr().cast(),
                joined.len() as u64,
                routes.as_ptr(),
                rails as c_int,
            )
        };
        if status != 0 {
            return Err(failure(format!(
                "RoCE queue-pair connect failed: {}",
                self.proxy_error()
            )));
        }
        if unsafe { roce_start(self.context) } != 0 {
            return Err(failure(format!(
                "RoCE proxy thread failed to start: {}",
                self.proxy_error()
            )));
        }
        Ok(())
    }

    fn proxy_error(&self) -> String {
        let text = unsafe { roce_error(self.context) };
        if text.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(text) }
                .to_string_lossy()
                .into_owned()
        }
    }

    /// Fail when the proxy stopped or a kernel's peer wait timed out; a frame that completed after
    /// either carries no trustworthy collective result.
    pub fn health(&self) -> Result<()> {
        if unsafe { roce_failed(self.context) } != 0 {
            return Err(failure(format!(
                "RoCE proxy failed: {}",
                self.proxy_error()
            )));
        }
        // SAFETY: The control record lies inside the region the transport was opened over.
        let words = unsafe { std::slice::from_raw_parts(self.control, 8) };
        check_control(self.spec.rank, words)
    }
}

fn check_control(rank: usize, words: &[AtomicU32]) -> Result<()> {
    let sequence = words[2].load(Ordering::Acquire);
    if sequence != 0 && words[7].load(Ordering::Relaxed) == INVALID_VARLEN_COUNT {
        return Err(failure(format!(
            "RoCE variable-length all-gather at sequence {sequence} received an unusable \
             byte count for rank {}; every rank stopped before sending, and the transport \
             is poisoned",
            words[3].load(Ordering::Relaxed),
        )));
    }
    if sequence != 0 {
        return Err(failure(format!(
            "RoCE collective on rank {} timed out waiting for rank {}, HCA {}, at sequence \
             {sequence}; the transport is poisoned",
            rank,
            words[3].load(Ordering::Relaxed),
            words[6].load(Ordering::Relaxed),
        )));
    }
    Ok(())
}

impl Drop for RoceTransport {
    fn drop(&mut self) {
        // Stops the posting thread before releasing queue pairs and memory registrations.
        unsafe { roce_destroy(self.context) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn specification() -> (Vec<u8>, RoceSpec, Topology) {
        let mut layout = [0; 17];
        // SAFETY: The C layout function writes exactly seventeen words and opens no devices.
        assert_eq!(
            unsafe { roce_layout(4, 4096, 4096, 4096, layout.as_mut_ptr()) },
            0
        );
        let mut region = vec![0x5a; layout[4] as usize];
        let spec = RoceSpec {
            proxy_abi: PROXY_ABI,
            world: 4,
            rank: 0,
            region_address: region.as_mut_ptr() as u64,
            region_bytes: layout[4],
            counters_address: 0,
            counters_bytes: 0,
            slot_bytes: 4096,
            pipe_bytes: 4096,
            varlen_bytes: 4096,
            ctrl_offset: layout[3],
        };
        let topology = Topology {
            hcas: vec!["rail0".into(), "rail1".into()],
            gid_index: 3,
            traffic_class: 0,
            rails: 2,
            routes: vec![vec![], vec![(0, 0), (1, 1)], vec![], vec![(0, 0), (1, 1)]],
            shape: Shape::Ring,
        };
        (region, spec, topology)
    }

    #[test]
    fn native_proxy_abi_matches_package_contract() {
        // SAFETY: This query has no arguments, device access, or allocation.
        assert_eq!(unsafe { roce_abi_version() } as u32, PROXY_ABI);
    }

    #[test]
    fn compatible_package_abis_still_validate_the_captured_region() {
        let (region, spec, topology) = specification();
        for abi in [8, 9, 10, 11] {
            let mut spec = spec.clone();
            spec.proxy_abi = abi;
            spec.ctrl_offset += 128;
            // SAFETY: ABI or layout validation rejects this before touching the live region.
            let error = unsafe { RoceTransport::open(&spec, &topology) }
                .err()
                .expect("invalid layout must fail")
                .to_string();
            assert!(
                error.contains(if matches!(abi, 9 | 10) {
                    "region differs"
                } else {
                    "proxy ABI"
                }),
                "ABI {abi}: {error}"
            );
            assert!(region.iter().all(|byte| *byte == 0x5a));
        }
    }

    #[test]
    fn peer_records_require_one_native_ring_protocol() {
        // SAFETY: This query has no arguments, device access, or allocation.
        let size = unsafe { roce_blob_bytes() } as usize;
        let record = |abi: u32, value: u8| {
            let mut bytes = abi.to_le_bytes().to_vec();
            bytes.extend(vec![value; size]);
            bytes
        };
        let records = vec![record(PROXY_ABI, 0x12), record(PROXY_ABI, 0x34)];
        let expected = [vec![0x12; size], vec![0x34; size]].concat();
        assert_eq!(peer_blobs(&records, 2).unwrap(), expected);
        assert!(peer_blobs(&records, 3).is_err());
        let mut foreign = records.clone();
        foreign[1] = record(9, 0x34);
        assert!(peer_blobs(&foreign, 2).is_err());
        let mut unversioned = records.clone();
        unversioned[1] = vec![0; size];
        assert!(peer_blobs(&unversioned, 2).is_err());
        let mut truncated = records;
        truncated[1].truncate(3);
        assert!(peer_blobs(&truncated, 2).is_err());
    }

    #[test]
    fn rejects_incompatible_package_and_routes_before_modifying_region() {
        let (region, spec, topology) = specification();
        for case in 0..10 {
            let mut spec = spec.clone();
            let mut topology = topology.clone();
            match case {
                0 => spec.proxy_abi += 1,
                1 => spec.ctrl_offset += 128,
                2 => spec.region_bytes -= 128,
                3 => {
                    topology.routes.pop();
                }
                4 => topology.rails = 0,
                5 => topology.rails = 3,
                6 => topology.routes[0] = vec![(0, 0), (1, 1)],
                7 => topology.routes[1].clear(),
                8 => topology.routes[2] = vec![(0, 0), (1, 1)],
                9 => topology.routes[3][1].0 = 2,
                _ => unreachable!(),
            }
            // SAFETY: Every case fails preflight before accessing or registering the region.
            // The backing allocation nevertheless remains live throughout the call.
            let result = unsafe { RoceTransport::open(&spec, &topology) };
            assert!(
                result.is_err(),
                "invalid transport case {case} passed preflight"
            );
            assert!(region.iter().all(|byte| *byte == 0x5a));
        }
    }

    #[test]
    fn control_errors_stay_poisoned_across_health_checks() {
        let words: [AtomicU32; 8] = std::array::from_fn(|_| AtomicU32::new(0));
        check_control(2, &words).unwrap();
        // Diagnostic words without an error sequence do not poison the transport.
        words[3].store(1, Ordering::Relaxed);
        words[6].store(1, Ordering::Relaxed);
        words[7].store(INVALID_VARLEN_COUNT, Ordering::Relaxed);
        check_control(2, &words).unwrap();
        words[2].store(37, Ordering::Release);
        let error = check_control(2, &words).unwrap_err().to_string();
        assert!(error.contains("variable-length all-gather at sequence 37"));
        assert!(error.contains("rank 1"));
        assert!(error.contains("transport is poisoned"));
        words[7].store(0, Ordering::Relaxed);
        let error = check_control(2, &words).unwrap_err().to_string();
        assert!(error.contains("rank 2 timed out waiting for rank 1, HCA 1"));
        assert!(error.contains("sequence 37"));
        assert!(error.contains("transport is poisoned"));
        // A subsequent health poll cannot turn the failed sequence into success.
        assert!(check_control(2, &words).is_err());
    }
}
