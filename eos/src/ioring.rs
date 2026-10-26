//! Pipelined Windows file reads into a fixed, caller-owned registered buffer set.

use std::{fs::File, io, os::windows::io::AsRawHandle, time::Instant};
use windows_sys::Win32::{
    Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0},
    Storage::FileSystem::*,
    System::{
        IO::CancelIoEx,
        Threading::{CreateEventW, INFINITE, WaitForSingleObject},
    },
};

fn check(status: i32) -> io::Result<()> {
    if status >= 0 {
        Ok(())
    } else if status as u32 & 0xffff0000 == 0x80070000 {
        Err(io::Error::from_raw_os_error(status & 0xffff))
    } else {
        Err(io::Error::other(format!(
            "Windows I/O ring HRESULT {status:#010x}"
        )))
    }
}

#[derive(Default)]
pub struct Timings {
    pub submit: u64,
    pub wait: u64,
    pub complete: u64,
}

pub struct ReadRing {
    pub timings: Timings,
    handle: HIORING,
    event: HANDLE,
    owned_files: Vec<File>,
    // Registration descriptors and the referenced storage outlive every completion,
    // including a registration failure during construction.
    files: Box<[HANDLE]>,
    buffers: Box<[IORING_BUFFER_INFO]>,
    pending: Box<[usize]>,
    queued: usize,
    in_flight: usize,
    failed: bool,
}

// SAFETY: One owner serializes the ring. Moving it preserves the boxed descriptors
// and the caller-owned file/buffer addresses covered by new's lifetime contract.
unsafe impl Send for ReadRing {}

impl ReadRing {
    /// # Safety
    /// The depth * chunk_bytes writable bytes at address must remain valid until
    /// this ring is dropped. Do not read or reuse a slot before its completion.
    pub unsafe fn new(
        files: Vec<File>,
        address: usize,
        depth: usize,
        chunk_bytes: u32,
    ) -> io::Result<Self> {
        let capacity = u32::try_from(depth.max(2)).map_err(|_| io::ErrorKind::InvalidInput)?;
        let file_count = u32::try_from(files.len()).map_err(|_| io::ErrorKind::InvalidInput)?;
        if files.is_empty()
            || depth == 0
            || chunk_bytes == 0
            || depth
                .checked_mul(chunk_bytes as usize)
                .and_then(|n| address.checked_add(n))
                .is_none()
        {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        let mut handle = std::ptr::null_mut();
        // Version 2 provides reads, registrations and reliable completion waits.
        check(unsafe {
            CreateIoRing(
                IORING_VERSION_2,
                IORING_CREATE_FLAGS::default(),
                capacity,
                capacity,
                &mut handle,
            )
        })?;
        let mut ring = Self {
            timings: Timings::default(),
            handle,
            event: std::ptr::null_mut(),
            files: files.iter().map(AsRawHandle::as_raw_handle).collect(),
            owned_files: files,
            buffers: (0..depth)
                .map(|slot| IORING_BUFFER_INFO {
                    Address: (address + slot * chunk_bytes as usize) as *mut _,
                    Length: chunk_bytes,
                })
                .collect(),
            pending: vec![usize::MAX; depth].into_boxed_slice(),
            queued: 0,
            in_flight: 0,
            failed: false,
        };
        ring.event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        if ring.event.is_null() {
            return Err(io::Error::last_os_error());
        }
        check(unsafe { SetIoRingCompletionEvent(handle, ring.event) })?;
        for operation in [
            IORING_OP_READ,
            IORING_OP_REGISTER_FILES,
            IORING_OP_REGISTER_BUFFERS,
        ] {
            if unsafe { IsIoRingOpSupported(handle, operation) } == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("Windows I/O ring operation {operation} is unsupported"),
                ));
            }
        }
        check(unsafe {
            BuildIoRingRegisterFileHandles(handle, file_count, ring.files.as_ptr(), 0)
        })?;
        ring.queued += 1;
        check(unsafe {
            BuildIoRingRegisterBuffers(handle, depth as u32, ring.buffers.as_ptr(), 1)
        })?;
        ring.queued += 1;
        ring.flush()?;
        let mut failure = None;
        while ring.in_flight != 0 {
            let completion = ring.completion()?;
            if let Err(error) = check(completion.ResultCode) {
                failure.get_or_insert(error);
            }
        }
        if let Some(error) = failure {
            return Err(error);
        }
        Ok(ring)
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.queued != 0 {
            let mut submitted = 0;
            // A submission failure leaves all entries queued. Drop abandons those
            // entries instead of allowing them to execute after buffer reuse.
            let begin = Instant::now();
            let status = unsafe { SubmitIoRing(self.handle, 0, 0, &mut submitted) };
            self.timings.submit += begin.elapsed().as_nanos() as u64;
            check(status)?;
            self.queued -= submitted as usize;
            self.in_flight += submitted as usize;
            if self.queued != 0 {
                return Err(io::Error::other("incomplete Windows I/O ring submission"));
            }
        }
        Ok(())
    }

    fn try_completion(&mut self) -> io::Result<Option<IORING_CQE>> {
        let mut completion = IORING_CQE::default();
        let begin = Instant::now();
        let status = unsafe { PopIoRingCompletion(self.handle, &mut completion) };
        self.timings.wait += begin.elapsed().as_nanos() as u64;
        check(status)?;
        if status == 0 {
            self.in_flight -= 1;
            Ok(Some(completion))
        } else {
            Ok(None)
        }
    }

    fn completion(&mut self) -> io::Result<IORING_CQE> {
        loop {
            if let Some(completion) = self.try_completion()? {
                return Ok(completion);
            }
            // Waiting on the event never submits queued entries. In particular,
            // shutdown can drain submitted reads and abandon the unsubmitted tail.
            let begin = Instant::now();
            let status = unsafe { WaitForSingleObject(self.event, INFINITE) };
            self.timings.wait += begin.elapsed().as_nanos() as u64;
            if status != WAIT_OBJECT_0 {
                return Err(io::Error::last_os_error());
            }
        }
    }

    fn build_read(&mut self, slot: usize, file: usize, offset: u64, nbytes: u32) -> io::Result<()> {
        let buffer = self.buffers.get(slot).ok_or(io::ErrorKind::InvalidInput)?;
        if file >= self.owned_files.len() || nbytes > buffer.Length {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        let begin = Instant::now();
        let status = unsafe {
            BuildIoRingReadFile(
                self.handle,
                IORING_HANDLE_REF {
                    Kind: IORING_REF_REGISTERED,
                    Handle: IORING_HANDLE_REF_0 { Index: file as u32 },
                },
                IORING_BUFFER_REF {
                    Kind: IORING_REF_REGISTERED,
                    Buffer: IORING_BUFFER_REF_0 {
                        IndexAndOffset: IORING_REGISTERED_BUFFER {
                            BufferIndex: slot as u32,
                            Offset: 0,
                        },
                    },
                },
                nbytes,
                offset,
                slot,
                IOSQE_FLAGS_NONE,
            )
        };
        self.timings.submit += begin.elapsed().as_nanos() as u64;
        check(status)?;
        self.queued += 1;
        Ok(())
    }

    /// Refill completed slots without waiting for the other reads in the window.
    /// The callback receives the read index and its registered buffer slot. It must
    /// consume the bytes before returning, when that slot becomes reusable.
    /// Complete every submitted read, including errors, before returning normally.
    /// An API failure poisons the ring: its slots cannot be reused before Drop drains.
    pub fn read_all(
        &mut self,
        reads: impl ExactSizeIterator<Item = (usize, u64, u32)>,
        mut complete: impl FnMut(usize, usize, io::Result<usize>),
    ) -> io::Result<()> {
        self.timings = Timings::default();
        if self.failed {
            return Err(io::Error::other(
                "Windows I/O ring is unusable after an API failure",
            ));
        }
        // The embedding worker catches panics. Keep the ring poisoned if a
        // completion consumer unwinds with reads still using registered slots.
        self.failed = true;
        let result = (|| {
            let mut reads = reads.enumerate();
            for slot in 0..self.buffers.len() {
                let Some((index, (file, offset, nbytes))) = reads.next() else {
                    break;
                };
                self.build_read(slot, file, offset, nbytes)?;
                self.pending[slot] = index;
            }
            self.flush()?;
            while self.in_flight != 0 {
                let mut cqe = self.completion()?;
                loop {
                    let begin = Instant::now();
                    let slot = cqe.UserData;
                    let index = self.pending.get_mut(slot).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "unknown I/O ring slot")
                    })?;
                    let index = std::mem::replace(index, usize::MAX);
                    if index == usize::MAX {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "inactive I/O ring slot completed",
                        ));
                    }
                    complete(index, slot, check(cqe.ResultCode).map(|_| cqe.Information));
                    self.timings.complete += begin.elapsed().as_nanos() as u64;
                    if let Some((index, (file, offset, nbytes))) = reads.next() {
                        self.build_read(slot, file, offset, nbytes)?;
                        self.pending[slot] = index;
                    }
                    // Batch the completions already available, but submit before
                    // more than half the window becomes idle. Never block with
                    // replacement reads still queued in user space.
                    if self.in_flight == 0 || self.queued >= self.buffers.len().div_ceil(2) {
                        break;
                    }
                    match self.try_completion()? {
                        Some(next) => cqe = next,
                        None => break,
                    }
                }
                self.flush()?;
            }
            Ok(())
        })();
        self.failed = result.is_err();
        result
    }
}

impl ReadRing {
    pub fn cancel_and_drain(&mut self) {
        // CloseIoRing does NOT cancel or drain in-flight I/O. Cancel against the
        // private reader handle and retain all descriptors and buffers until drained.
        if self.in_flight != 0 {
            for file in &self.owned_files {
                unsafe {
                    CancelIoEx(file.as_raw_handle(), std::ptr::null());
                }
            }
        }
        while self.in_flight != 0 {
            if self.completion().is_err() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

impl Drop for ReadRing {
    fn drop(&mut self) {
        self.cancel_and_drain();
        // Unsubmitted entries, including failed registrations, are abandoned here.
        unsafe {
            CloseIoRing(self.handle);
            if !self.event.is_null() {
                CloseHandle(self.event);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panicked_completion_prevents_registered_slot_reuse() {
        let path =
            std::env::temp_dir().join(format!("flare-ioring-panic-{}.bin", std::process::id()));
        std::fs::write(&path, [0x71; 8192]).unwrap();
        let source = File::open(&path).unwrap();
        let file = crate::direct_reopen(&source).unwrap();
        let address = crate::allocate_aligned(8192).unwrap();
        let mut ring = unsafe { ReadRing::new(vec![file], address, 2, 4096) }.unwrap();
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = ring.read_all([(0, 0, 4096), (0, 4096, 4096)].into_iter(), |_, _, _| {
                panic!("completion consumer panicked");
            });
        }));
        assert!(failure.is_err());
        assert_eq!(ring.in_flight, 1);
        assert!(
            ring.read_all([(0, 0, 4096)].into_iter(), |_, _, _| {})
                .is_err()
        );
        drop(ring);
        unsafe {
            crate::free_aligned(address, 8192);
        }
        drop(source);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn registered_reads_preserve_slots_and_drain_failures_before_reuse() {
        let path = std::env::temp_dir().join(format!("flare-ioring-{}.bin", std::process::id()));
        let bytes: Vec<u8> = (0..4096 * 256).map(|i| (i / 4096) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let source = File::open(&path).unwrap();
        let file = crate::direct_reopen(&source).unwrap();
        let depth = 128;
        let address = crate::allocate_aligned(depth * 4096).unwrap();
        let mut ring = unsafe { ReadRing::new(vec![file], address, depth, 4096) }.unwrap();
        for (round, count) in [128, 0, 1, 641, 77, 257].into_iter().enumerate() {
            let mut seen = vec![false; count];
            ring.read_all(
                (0..count).map(|index| (0, ((index * 101 + round) % 256 * 4096) as u64, 4096)),
                |index, slot, result| {
                    assert_eq!(result.unwrap(), 4096);
                    assert!(!seen[index]);
                    seen[index] = true;
                    let expected = ((index * 101 + round) % 256) as u8;
                    assert!(
                        unsafe {
                            std::slice::from_raw_parts((address + slot * 4096) as *const u8, 4096)
                        }
                        .iter()
                        .all(|byte| *byte == expected)
                    );
                    unsafe {
                        std::ptr::write_bytes((address + slot * 4096) as *mut u8, 0xa5, 4096);
                    }
                },
            )
            .unwrap();
            assert!(seen.iter().all(|value| *value));
        }
        let mut completions = 0;
        ring.read_all(
            (0..259).map(|index| {
                (
                    0,
                    if index % 17 == 0 {
                        bytes.len() as u64 + 4096
                    } else {
                        0
                    },
                    4096,
                )
            }),
            |index, _, result| {
                completions += 1;
                if index % 17 != 0 {
                    assert_eq!(result.unwrap(), 4096);
                } else {
                    assert!(result.is_err() || result.unwrap() == 0);
                }
            },
        )
        .unwrap();
        assert_eq!(completions, 259);
        assert_eq!(ring.in_flight, 0);
        ring.read_all([(0, 4096, 4096)].into_iter(), |_, _, result| {
            assert_eq!(result.unwrap(), 4096)
        })
        .unwrap();
        // Exercise shutdown with both submitted and unsubmitted entries. Drop must
        // drain the submitted batch before the caller poisons/frees its buffers.
        ring.build_read(0, 0, 0, 4096).unwrap();
        ring.flush().unwrap();
        ring.build_read(1, 0, 4096, 4096).unwrap();
        drop(ring);
        unsafe {
            std::ptr::write_bytes(address as *mut u8, 0xa5, depth * 4096);
            crate::free_aligned(address, depth * 4096);
        }
        drop(source);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn invalid_refill_poisons_ring_and_drains_before_freeing_slots() {
        let path =
            std::env::temp_dir().join(format!("flare-ioring-refill-{}.bin", std::process::id()));
        std::fs::write(&path, [0x71; 8192]).unwrap();
        let source = File::open(&path).unwrap();
        let file = crate::direct_reopen(&source).unwrap();
        let address = crate::allocate_aligned(8192).unwrap();
        let mut ring = unsafe { ReadRing::new(vec![file], address, 2, 4096) }.unwrap();
        let mut completions = 0;
        assert!(
            ring.read_all(
                [(0, 0, 4096), (0, 4096, 4096), (0, 0, 4096), (0, 0, 8192)].into_iter(),
                |_, _, result| {
                    result.unwrap();
                    completions += 1;
                },
            )
            .is_err()
        );
        assert_eq!(completions, 2);
        assert!(ring.in_flight != 0);
        assert!(
            ring.read_all([(0, 0, 4096)].into_iter(), |_, _, _| {})
                .is_err()
        );
        drop(ring);
        unsafe {
            std::ptr::write_bytes(address as *mut u8, 0xa5, 8192);
            crate::free_aligned(address, 8192);
        }
        drop(source);
        std::fs::remove_file(path).unwrap();
    }
}
