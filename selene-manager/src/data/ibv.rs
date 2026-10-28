//! Registered manager regions and one signalled RDMA write per window (experimental).

use super::channel::{Pair, Rdma};
use super::{Region, invalid};
use crate::regions::LocalRegions;
use flare_selene::shared::rdma::{self, Handshake, Qp, ffi};
use std::{io, mem, ptr, sync::Arc};

/// One transfer's queue pair and the registrations it keeps alive.
pub struct LocalRdma {
    pair: Qp,
    device: Arc<Device>,
    peer: Option<Handshake>,
}

/// The device context, protection domain and both registered regions. Shared by every
/// queue pair made from it, and closed when the last of them is gone.
pub struct Device {
    /// Keeps the mappings alive while the memory regions refer to them.
    _regions: Arc<LocalRegions>,
    device: Arc<rdma::Device>,
    arena: *mut ffi::ibv_mr,
    staging: *mut ffi::ibv_mr,
}

// SAFETY: libibverbs objects may be used from any thread; the handles are never changed after
// `open` and are released once, when the last queue pair and the device itself are gone.
unsafe impl Send for Device {}
// SAFETY: as above.
unsafe impl Sync for Device {}

impl Device {
    /// Open the first device whose port is active and register both mapped regions.
    ///
    /// The device retains `regions` for its whole lifetime, so the mappings stay alive while the
    /// memory regions refer to them, and every bound is taken from the registered lengths rather
    /// than from a caller-supplied size.
    pub fn open(regions: Arc<LocalRegions>) -> io::Result<Arc<Self>> {
        let arena = regions.mapping(Region::Arena);
        let staging = regions.mapping(Region::Staging);
        if arena.1 == 0 || staging.1 == 0 || arena.0.is_null() || staging.0.is_null() {
            return Err(invalid("both RDMA regions must be mapped"));
        }
        let device = rdma::Device::open()?;
        let pd = device.pd;
        let access = ffi::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0 as i32;
        // SAFETY: both ranges are mapped by `regions`, which this value retains.
        let arena_mr = unsafe { ffi::ibv_reg_mr(pd, arena.0.cast(), arena.1, access) };
        if arena_mr.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: as above for the staging range.
        let staging_mr = unsafe { ffi::ibv_reg_mr(pd, staging.0.cast(), staging.1, access) };
        if staging_mr.is_null() {
            let error = io::Error::last_os_error();
            release_mrs(arena_mr, ptr::null_mut());
            return Err(error);
        }
        Ok(Arc::new(Self {
            _regions: regions,
            device,
            arena: arena_mr,
            staging: staging_mr,
        }))
    }

    /// A queue pair for one transfer, in INIT.
    pub fn pair(self: &Arc<Self>) -> io::Result<LocalRdma> {
        Ok(LocalRdma {
            pair: Qp::new(&self.device, 1, 16, 0)?,
            device: self.clone(),
            peer: None,
        })
    }
}

impl Rdma for Arc<Device> {
    fn pair(&self) -> io::Result<Box<dyn Pair>> {
        Ok(Box::new(Device::pair(self)?))
    }
}

impl Pair for LocalRdma {
    fn handshake(&self) -> Vec<u8> {
        self.pair.handshake(0, 0).encode().to_vec()
    }

    fn connect(&mut self, peer: &[u8]) -> io::Result<()> {
        let peer = Handshake::decode(peer)?;
        if peer.rkey_staging == 0 {
            return Err(invalid("the instance registered no staging"));
        }
        self.pair.connect(&peer)?;
        self.peer = Some(peer);
        Ok(())
    }

    fn write(
        &mut self,
        region: Region,
        offset: u64,
        len: u64,
        remote_offset: u64,
    ) -> io::Result<()> {
        let peer = self
            .peer
            .as_ref()
            .ok_or_else(|| invalid("RDMA pair is not connected"))?;
        if len == 0
            || !len.is_multiple_of(4096)
            || !offset.is_multiple_of(4096)
            || !remote_offset.is_multiple_of(4096)
        {
            return Err(invalid("RDMA range is not 4096-aligned"));
        }
        let mr = match region {
            Region::Arena => self.device.arena,
            Region::Staging => self.device.staging,
        };
        // SAFETY: this pair retains the registration and its mappings.
        let mr = unsafe { &*mr };
        if offset
            .checked_add(len)
            .is_none_or(|end| end > mr.length as u64)
        {
            return Err(invalid("local range is outside the region"));
        }
        let mut sge = ffi::ibv_sge {
            addr: (mr.addr as u64)
                .checked_add(offset)
                .ok_or_else(|| invalid("local address overflow"))?,
            length: u32::try_from(len).map_err(|_| invalid("RDMA write is too large"))?,
            lkey: mr.lkey,
        };
        let mut wr = unsafe { mem::zeroed::<ffi::ibv_send_wr>() };
        wr.sg_list = &mut sge;
        wr.num_sge = 1;
        wr.opcode = ffi::ibv_wr_opcode::IBV_WR_RDMA_WRITE;
        wr.send_flags = ffi::ibv_send_flags::IBV_SEND_SIGNALED.0;
        wr.wr.rdma = ffi::ibv_send_wr__bindgen_ty_2__bindgen_ty_1 {
            remote_addr: peer
                .staging_base
                .checked_add(remote_offset)
                .ok_or_else(|| invalid("remote address overflow"))?,
            rkey: peer.rkey_staging,
        };
        let mut bad: *mut ffi::ibv_send_wr = ptr::null_mut();
        // SAFETY: the queue pair is RTS and the work request refers to registered memory.
        let errno = unsafe {
            let ops = &mut (*self.pair.qp).context.as_mut().unwrap().ops;
            (ops.post_send
                .ok_or_else(|| invalid("libibverbs has no post_send"))?)(
                self.pair.qp,
                &mut wr,
                &mut bad,
            )
        };
        if errno != 0 {
            return Err(io::Error::from_raw_os_error(errno));
        }
        Ok(())
    }

    fn poll_done(&mut self) -> io::Result<bool> {
        // SAFETY: the pair is live until the handler fences it on exit.
        let poll = unsafe { (*self.pair.qp).context.as_ref().unwrap().ops.poll_cq }
            .ok_or_else(|| invalid("libibverbs has no poll_cq"))?;
        let mut completion = unsafe { mem::zeroed::<ffi::ibv_wc>() };
        // SAFETY: the CQ is live and completion holds one entry.
        match unsafe { poll(self.pair.cq, 1, &mut completion) } {
            0 => Ok(false),
            1 if completion.is_valid() => Ok(true),
            1 => Err(io::Error::other("RDMA work completion failed")),
            _ => Err(io::Error::last_os_error()),
        }
    }

    fn close(&mut self) -> io::Result<()> {
        self.pair.close()
    }
}

impl Drop for LocalRdma {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            eprintln!("warning: quarantining an unfenced RDMA channel: {error}");
            mem::forget(self.device.clone());
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        release_mrs(self.arena, self.staging);
    }
}

fn release_mrs(arena: *mut ffi::ibv_mr, staging: *mut ffi::ibv_mr) {
    // SAFETY: each MR is either null or a live registration released once.
    unsafe {
        if !arena.is_null() {
            ffi::ibv_dereg_mr(arena);
        }
        if !staging.is_null() {
            ffi::ibv_dereg_mr(staging);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;
    use std::time::{Duration, Instant};

    const BYTES: usize = 4 << 20;

    fn regions() -> Arc<LocalRegions> {
        Arc::new(LocalRegions::new(
            Arena::new(BYTES).unwrap(),
            Arena::new(BYTES).unwrap(),
        ))
    }

    #[test]
    fn a_loopback_pair_writes_only_the_private_destination() {
        let regions_a = regions();
        let regions_b = regions();
        let Ok(device_a) = Device::open(regions_a.clone()) else {
            assert!(
                std::env::var_os("SELENE_RDMA_REQUIRED").is_none(),
                "no active RDMA device"
            );
            eprintln!("skipping: no active RDMA device");
            return;
        };
        let mut a = device_a.pair().unwrap();
        // SAFETY: this private destination stays mapped until the pair and device are dropped.
        let device_b = unsafe {
            flare_selene::pool::rdma::Device::open(regions_b.mapping(Region::Staging).0, BYTES)
        }
        .unwrap();
        let b = flare_selene::pool::rdma::Pair::new(&device_b).unwrap();
        let handshake_a = Handshake::decode(&a.handshake()).unwrap();
        assert_eq!(handshake_a.rkey_staging, 0);
        assert_eq!(handshake_a.staging_base, 0);
        a.connect(&b.handshake().encode()).unwrap();
        b.connect(&handshake_a).unwrap();
        // SAFETY: the test owns both allocations.
        unsafe {
            for index in 0..4096 {
                *regions_a.mapping(Region::Arena).0.add(index) = (index % 251) as u8;
            }
        }
        for (offset, len, remote) in [
            (1, 4096, 0),
            (0, 1, 0),
            (0, 4096, 1),
            (BYTES as u64, 4096, 0),
            (0, 0, 0),
        ] {
            assert!(a.write(Region::Arena, offset, len, remote).is_err());
        }
        a.write(Region::Arena, 0, 4096, 0).unwrap();
        wait(&mut a);
        assert_eq!(
            unsafe { regions_b.bytes(Region::Staging, 0, 4096).unwrap() },
            unsafe { regions_a.bytes(Region::Arena, 0, 4096).unwrap() }
        );
        // Exercise a write larger than the old piece size.
        let len = 2 * (1 << 20) + 4096;
        unsafe {
            for index in 0..len {
                *regions_a.mapping(Region::Arena).0.add(index) = (index % 251) as u8;
            }
        }
        a.write(Region::Arena, 0, len as u64, 0).unwrap();
        wait(&mut a);
        assert_eq!(
            unsafe { regions_b.bytes(Region::Staging, 0, len as u64).unwrap() },
            unsafe { regions_a.bytes(Region::Arena, 0, len as u64).unwrap() }
        );
    }

    fn wait(channel: &mut LocalRdma) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !channel.poll_done().unwrap() {
            assert!(Instant::now() < deadline, "RDMA transfer timed out");
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}
