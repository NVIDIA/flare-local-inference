//! Bounded aligned storage transfers used while constructing runtime packages.

use super::ExportResult as Result;
use std::{
    fs::{File, OpenOptions},
    path::Path,
    sync::{Arc, Condvar, Mutex},
};

pub(crate) const ALIGNMENT: usize = 4096;
pub(crate) const BLOCK: usize = 8 << 20;
const CAPACITY: usize = BLOCK + ALIGNMENT;

pub(crate) struct Aligned {
    pointer: std::ptr::NonNull<u8>,
    capacity: usize,
}
unsafe impl Send for Aligned {}
impl Aligned {
    fn new(capacity: usize) -> Result<Self> {
        let layout = std::alloc::Layout::from_size_align(capacity, ALIGNMENT)?;
        let pointer = std::ptr::NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })
            .ok_or("cannot allocate export I/O buffer")?;
        Ok(Self { pointer, capacity })
    }
    pub(crate) fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.capacity) }
    }
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.capacity) }
    }
}
impl Drop for Aligned {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(
                self.pointer.as_ptr(),
                std::alloc::Layout::from_size_align(self.capacity, ALIGNMENT).unwrap(),
            );
        }
    }
}

struct Pool {
    idle: Vec<Aligned>,
    allocated: usize,
}
pub(crate) struct IoBudget {
    pool: Mutex<Pool>,
    ready: Condvar,
    pub(crate) depth: usize,
    read_pool: Mutex<Pool>,
    read_ready: Condvar,
    read_depth: usize,
    #[cfg(test)]
    pub(crate) scan_bytes: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    bounced_bytes: std::sync::atomic::AtomicU64,
}
pub(crate) struct Buffer<'a> {
    owner: &'a IoBudget,
    buffer: Option<Aligned>,
    scratch: bool,
}
impl Buffer<'_> {
    pub(crate) fn bytes(&self) -> &[u8] {
        self.buffer.as_ref().unwrap().bytes()
    }
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        self.buffer.as_mut().unwrap().bytes_mut()
    }
}
impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        let (pool, ready) = if self.scratch {
            (&self.owner.read_pool, &self.owner.read_ready)
        } else {
            (&self.owner.pool, &self.owner.ready)
        };
        pool.lock().unwrap().idle.push(self.buffer.take().unwrap());
        ready.notify_one();
    }
}
fn buffer_depths(ranks: usize, available: u64, capture: bool) -> Result<(usize, usize)> {
    if ranks == 0 {
        return Err("export rank count must be positive".into());
    }
    let total = (1u64 << 30).min(available / 16);
    let reserved = if capture {
        (4 * BLOCK as u64)
            .checked_mul(ranks as u64)
            .ok_or("capture buffer budget overflow")?
    } else {
        0
    };
    let usable = total
        .checked_sub(reserved)
        .ok_or("insufficient host memory for capture transfer buffers")?;
    let slots = ((usable / ranks as u64 / CAPACITY as u64) as usize).min(128 / ranks);
    if slots == 0 {
        return Err("insufficient available RAM for an 8 MiB export buffer per rank".into());
    }
    let scratch = if slots > 1 {
        (slots / 4).max(1).min(8)
    } else {
        0
    };
    Ok((slots - scratch, scratch))
}
impl IoBudget {
    pub(crate) fn for_ranks(ranks: usize) -> Result<Arc<Self>> {
        Self::build(ranks, false)
    }
    pub(crate) fn for_capture(ranks: usize) -> Result<Arc<Self>> {
        Self::build(ranks, true)
    }
    fn build(ranks: usize, capture: bool) -> Result<Arc<Self>> {
        let (depth, read_depth) = buffer_depths(ranks, available_memory()?, capture)?;
        Ok(Arc::new(Self {
            pool: Mutex::new(Pool {
                idle: Vec::new(),
                allocated: 0,
            }),
            ready: Condvar::new(),
            depth,
            read_pool: Mutex::new(Pool {
                idle: Vec::new(),
                allocated: 0,
            }),
            read_ready: Condvar::new(),
            read_depth,
            #[cfg(test)]
            scan_bytes: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            bounced_bytes: std::sync::atomic::AtomicU64::new(0),
        }))
    }
    pub(crate) fn buffer(&self) -> Result<Buffer<'_>> {
        self.acquire(false)
    }
    fn acquire(&self, scratch: bool) -> Result<Buffer<'_>> {
        let (pool_mutex, ready, depth) = if scratch {
            (&self.read_pool, &self.read_ready, self.read_depth)
        } else {
            (&self.pool, &self.ready, self.depth)
        };
        let mut pool = pool_mutex.lock().unwrap();
        loop {
            if let Some(buffer) = pool.idle.pop() {
                return Ok(Buffer {
                    owner: self,
                    buffer: Some(buffer),
                    scratch,
                });
            }
            if pool.allocated < depth {
                pool.allocated += 1;
                drop(pool);
                return match Aligned::new(CAPACITY) {
                    Ok(buffer) => Ok(Buffer {
                        owner: self,
                        buffer: Some(buffer),
                        scratch,
                    }),
                    Err(error) => {
                        pool_mutex.lock().unwrap().allocated -= 1;
                        ready.notify_one();
                        Err(error)
                    }
                };
            }
            pool = ready.wait(pool).unwrap();
        }
    }
    pub(crate) fn read_at(&self, source: &Source, offset: u64, output: &mut [u8]) -> Result<()> {
        if output.len() < ALIGNMENT
            || source.direct_compatible(offset, output)
            || self.read_depth == 0
        {
            return source.read_at(offset, output);
        }
        let mut scratch = self.acquire(true)?;
        let transferred = source.read_bounced(offset, output, scratch.bytes_mut())?;
        #[cfg(test)]
        self.bounced_bytes
            .fetch_add(transferred, std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(test))]
        let _ = transferred;
        Ok(())
    }
    fn try_buffer(&self) -> Result<Option<Buffer<'_>>> {
        let mut pool = self.pool.lock().unwrap();
        if let Some(buffer) = pool.idle.pop() {
            return Ok(Some(Buffer {
                owner: self,
                buffer: Some(buffer),
                scratch: false,
            }));
        }
        if pool.allocated == self.depth {
            return Ok(None);
        }
        pool.allocated += 1;
        drop(pool);
        match Aligned::new(CAPACITY) {
            Ok(buffer) => Ok(Some(Buffer {
                owner: self,
                buffer: Some(buffer),
                scratch: false,
            })),
            Err(error) => {
                self.pool.lock().unwrap().allocated -= 1;
                self.ready.notify_one();
                Err(error)
            }
        }
    }
    pub(crate) fn scan(
        &self,
        source: &Source,
        mut consume: impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        if source.length == 0 {
            return source.check_unchanged();
        }
        let file = direct_reopen(&source.file, false)?;
        std::thread::scope(|scope| {
            let (sender, receiver) = std::sync::mpsc::sync_channel(self.depth);
            let reader = scope.spawn(move || -> Result<()> {
                let mut offset = 0;
                while offset < source.length {
                    let mut buffers = vec![self.buffer()?];
                    let count = ((source.length - offset).div_ceil(BLOCK as u64) as usize)
                        .min(self.depth)
                        .min(32);
                    while buffers.len() < count {
                        match self.try_buffer()? {
                            Some(buffer) => buffers.push(buffer),
                            None => break,
                        }
                    }
                    let mut jobs = Vec::new();
                    let mut spans = Vec::new();
                    for buffer in &mut buffers {
                        let count = (source.length - offset).min(BLOCK as u64) as usize;
                        let position = source.offset + offset;
                        let lead = position as usize % ALIGNMENT;
                        let padded = (lead + count).next_multiple_of(ALIGNMENT);
                        jobs.push((
                            buffer.bytes_mut().as_mut_ptr(),
                            padded,
                            position - lead as u64,
                            lead + count,
                        ));
                        spans.push((lead, count));
                        offset += count as u64;
                    }
                    transfer_batch(&file, &jobs, false)?;
                    #[cfg(test)]
                    self.scan_bytes.fetch_add(
                        jobs.iter().map(|job| job.1 as u64).sum(),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    for (buffer, (lead, count)) in buffers.into_iter().zip(spans) {
                        sender
                            .send((buffer, lead, count))
                            .map_err(|_| "export scan consumer stopped")?;
                    }
                }
                source.check_unchanged()
            });
            let consumed: Result<()> = (|| {
                while let Ok((buffer, lead, count)) = receiver.recv() {
                    consume(&buffer.bytes()[lead..lead + count])?;
                }
                Ok(())
            })();
            drop(receiver);
            let read = reader.join().map_err(|_| "native export reader panicked")?;
            consumed?;
            read
        })
    }
    pub(crate) fn hash_generated(
        &self,
        size: u64,
        prefix: &[u8],
        mut fill: impl FnMut(u64, &mut [u8]) -> Result<()>,
    ) -> Result<[u8; 32]> {
        use sha2::{Digest, Sha256};
        std::thread::scope(|scope| {
            let (sender, receiver) =
                std::sync::mpsc::sync_channel::<(Buffer<'_>, usize)>(self.depth);
            let hasher = scope.spawn(move || {
                let mut digest = Sha256::new();
                digest.update(prefix);
                while let Ok((buffer, count)) = receiver.recv() {
                    digest.update(&buffer.bytes()[..count]);
                }
                <[u8; 32]>::from(digest.finalize())
            });
            let filled: Result<()> = (|| {
                let mut offset = 0;
                while offset < size {
                    let mut buffer = self.buffer()?;
                    let count = (size - offset).min(BLOCK as u64) as usize;
                    fill(offset, &mut buffer.bytes_mut()[..count])?;
                    sender
                        .send((buffer, count))
                        .map_err(|_| "export hash worker stopped")?;
                    offset += count as u64;
                }
                Ok(())
            })();
            drop(sender);
            let digest = hasher.join().map_err(|_| "native export hasher panicked")?;
            filled?;
            Ok(digest)
        })
    }
    /// Fill and submit disjoint blocks, retaining every buffer until its I/O completes.
    pub(crate) fn write(
        &self,
        output: &File,
        offset: u64,
        size: u64,
        fill: impl FnMut(u64, &mut [u8]) -> Result<()>,
    ) -> Result<()> {
        self.write_inner(output, offset, size, None, fill)
            .map(|_| ())
    }
    pub(crate) fn write_hashed(
        &self,
        output: &File,
        offset: u64,
        size: u64,
        prefix: &[u8],
        fill: impl FnMut(u64, &mut [u8]) -> Result<()>,
    ) -> Result<[u8; 32]> {
        self.write_inner(output, offset, size, Some(prefix), fill)?
            .ok_or_else(|| "missing export checksum".into())
    }
    fn write_inner(
        &self,
        output: &File,
        offset: u64,
        size: u64,
        prefix: Option<&[u8]>,
        mut fill: impl FnMut(u64, &mut [u8]) -> Result<()>,
    ) -> Result<Option<[u8; 32]>> {
        let direct = direct_reopen(output, true)?;
        let original_length = output.metadata()?.len();
        let digest = std::thread::scope(|scope| {
            let (sender, filled_receiver) =
                std::sync::mpsc::sync_channel::<(Buffer<'_>, u64, usize)>(self.depth);
            let (receiver, hasher) = if let Some(prefix) = prefix {
                let (sender, receiver) = std::sync::mpsc::sync_channel(self.depth);
                let hasher = scope.spawn(move || -> Result<[u8; 32]> {
                    use sha2::{Digest, Sha256};
                    let mut digest = Sha256::new();
                    digest.update(prefix);
                    while let Ok((buffer, position, count)) = filled_receiver.recv() {
                        digest.update(&buffer.bytes()[..count]);
                        sender
                            .send((buffer, position, count))
                            .map_err(|_| "export disk writer stopped")?;
                    }
                    Ok(digest.finalize().into())
                });
                (receiver, Some(hasher))
            } else {
                (filled_receiver, None)
            };
            let writer = scope.spawn(move || -> Result<()> {
                while let Ok(first) = receiver.recv() {
                    let mut blocks = vec![first];
                    while blocks.len() < 32 {
                        match receiver.try_recv() {
                            Ok(block) => blocks.push(block),
                            Err(_) => break,
                        }
                    }
                    let mut jobs = Vec::new();
                    for (buffer, position, count) in &mut blocks {
                        let position = offset + *position;
                        let lead = position as usize % ALIGNMENT;
                        let prefix = if lead == 0 {
                            0
                        } else {
                            (*count).min(ALIGNMENT - lead)
                        };
                        if prefix > 0 {
                            write_all_at(output, &buffer.bytes()[..prefix], position)?;
                        }
                        let bulk = (*count - prefix) / ALIGNMENT * ALIGNMENT;
                        if bulk > 0 {
                            // Producer splits an unaligned destination prefix before full blocks.
                            if prefix != 0 {
                                return Err("unaligned bulk export buffer".into());
                            }
                            jobs.push((buffer.bytes_mut().as_mut_ptr(), bulk, position, bulk));
                        }
                        if prefix + bulk < *count {
                            write_all_at(
                                output,
                                &buffer.bytes()[prefix + bulk..*count],
                                position + (prefix + bulk) as u64,
                            )?;
                        }
                    }
                    if !jobs.is_empty() {
                        transfer_batch(&direct, &jobs, true)?;
                    }
                }
                Ok(())
            });
            let filled: Result<()> = (|| {
                let mut position = 0;
                let lead = offset as usize % ALIGNMENT;
                if lead != 0 && size > 0 {
                    let mut buffer = self.buffer()?;
                    let count = size.min((ALIGNMENT - lead) as u64) as usize;
                    fill(0, &mut buffer.bytes_mut()[..count])?;
                    sender
                        .send((buffer, 0, count))
                        .map_err(|_| "export writer stopped")?;
                    position = count as u64;
                }
                while position < size {
                    let mut buffer = self.buffer()?;
                    let count = (size - position).min(BLOCK as u64) as usize;
                    fill(position, &mut buffer.bytes_mut()[..count])?;
                    sender
                        .send((buffer, position, count))
                        .map_err(|_| "export writer stopped")?;
                    position += count as u64;
                }
                Ok(())
            })();
            drop(sender);
            let hashed = hasher
                .map(|worker| worker.join().map_err(|_| "native export hasher panicked"))
                .transpose()?;
            let written = writer.join().map_err(|_| "native export writer panicked")?;
            written?;
            filled?;
            hashed.transpose()
        })?;
        output.set_len(
            original_length.max(offset.checked_add(size).ok_or("export length overflow")?),
        )?;
        Ok(digest)
    }
    pub(crate) fn copy(&self, source: &Source, output: &File, offset: u64) -> Result<()> {
        self.write(output, offset, source.length, |position, bytes| {
            self.read_at(source, position, bytes)
        })?;
        source.check_unchanged()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Identity {
    pub(crate) size: u64,
    #[cfg(unix)]
    pub(crate) device: u64,
    #[cfg(unix)]
    pub(crate) inode: u64,
    #[cfg(unix)]
    pub(crate) modified: (i64, i64),
    #[cfg(unix)]
    pub(crate) changed: (i64, i64),
    #[cfg(windows)]
    modified: u64,
    #[cfg(windows)]
    created: u64,
    #[cfg(windows)]
    volume: u32,
    #[cfg(windows)]
    index: u64,
    #[cfg(windows)]
    changed: i64,
}
impl Identity {
    pub(crate) fn same_file(&self, other: &Self) -> bool {
        #[cfg(unix)]
        {
            self.device == other.device && self.inode == other.inode
        }
        #[cfg(windows)]
        {
            self.volume == other.volume && self.index == other.index
        }
    }
    pub(crate) fn same_content(&self, other: &Self) -> bool {
        self.same_file(other) && self.size == other.size && self.modified == other.modified
    }
    pub(crate) fn of(file: &File) -> Result<Self> {
        let info = file.metadata()?;
        if !info.is_file() {
            return Err("export source must be a regular file".into());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Ok(Self {
                size: info.len(),
                device: info.dev(),
                inode: info.ino(),
                modified: (info.mtime(), info.mtime_nsec()),
                changed: (info.ctime(), info.ctime_nsec()),
            })
        }
        #[cfg(windows)]
        {
            use std::os::windows::{fs::MetadataExt, io::AsRawHandle};
            use windows_sys::Win32::Storage::FileSystem::{
                BY_HANDLE_FILE_INFORMATION, FILE_BASIC_INFO, FileBasicInfo,
                GetFileInformationByHandle, GetFileInformationByHandleEx,
            };
            let mut native: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
            let mut basic: FILE_BASIC_INFO = unsafe { std::mem::zeroed() };
            if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut native) } == 0
                || unsafe {
                    GetFileInformationByHandleEx(
                        file.as_raw_handle(),
                        FileBasicInfo,
                        (&mut basic as *mut FILE_BASIC_INFO).cast(),
                        std::mem::size_of_val(&basic) as u32,
                    )
                } == 0
            {
                return Err(std::io::Error::last_os_error().into());
            }
            Ok(Self {
                size: info.len(),
                modified: info.last_write_time(),
                created: info.creation_time(),
                volume: native.dwVolumeSerialNumber,
                index: ((native.nFileIndexHigh as u64) << 32) | native.nFileIndexLow as u64,
                changed: basic.ChangeTime,
            })
        }
    }
}

#[derive(Clone)]
pub(crate) struct Source {
    pub(crate) file: Arc<File>,
    pub(crate) offset: u64,
    pub(crate) length: u64,
    identity: Identity,
    direct: Arc<Mutex<Option<Arc<File>>>>,
}
impl Source {
    pub(crate) fn identity(&self) -> Identity {
        self.identity.clone()
    }
    pub(crate) fn open_checked(
        path: &Path,
        offset: u64,
        length: u64,
        expected: &Identity,
    ) -> Result<Self> {
        let source = Self::open(path, offset, length)?;
        if &source.identity != expected {
            return Err("export source changed before reading".into());
        }
        Ok(source)
    }
    pub(crate) fn pinned_path(&self) -> Result<std::path::PathBuf> {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            Ok(std::path::PathBuf::from(format!(
                "/proc/self/fd/{}",
                self.file.as_raw_fd()
            )))
        }
        #[cfg(windows)]
        {
            use std::{
                ffi::OsString,
                os::windows::{ffi::OsStringExt, io::AsRawHandle},
            };
            use windows_sys::Win32::Storage::FileSystem::GetFinalPathNameByHandleW;
            let mut buffer = vec![0u16; 32768];
            let count = unsafe {
                GetFinalPathNameByHandleW(
                    self.file.as_raw_handle(),
                    buffer.as_mut_ptr(),
                    buffer.len() as u32,
                    0,
                )
            };
            if count == 0 || count as usize >= buffer.len() {
                return Err(std::io::Error::last_os_error().into());
            }
            Ok(std::path::PathBuf::from(OsString::from_wide(
                &buffer[..count as usize],
            )))
        }
    }
    pub(crate) fn open(path: &Path, offset: u64, length: u64) -> Result<Self> {
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC);
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            options.share_mode(windows_sys::Win32::Storage::FileSystem::FILE_SHARE_READ);
        }
        Self::from_file(Arc::new(options.open(path)?), offset, length)
    }
    pub(crate) fn from_file(file: Arc<File>, offset: u64, length: u64) -> Result<Self> {
        let identity = Identity::of(&file)?;
        if offset
            .checked_add(length)
            .is_none_or(|end| end > identity.size)
        {
            return Err("export source range exceeds file".into());
        }
        Ok(Self {
            file,
            offset,
            length,
            identity,
            direct: Arc::new(Mutex::new(None)),
        })
    }
    pub(crate) fn slice(&self, offset: u64, length: u64) -> Result<Self> {
        if offset
            .checked_add(length)
            .is_none_or(|end| end > self.length)
        {
            return Err("export source slice exceeds range".into());
        }
        Ok(Self {
            offset: self.offset + offset,
            length,
            ..self.clone()
        })
    }
    pub(crate) fn check_unchanged(&self) -> Result<()> {
        if Identity::of(&self.file)? != self.identity {
            return Err("export source changed during operation".into());
        }
        Ok(())
    }
    fn direct_file(&self) -> Result<Arc<File>> {
        let mut direct = self.direct.lock().unwrap();
        if direct.is_none() {
            *direct = Some(Arc::new(direct_reopen(&self.file, false)?));
        }
        Ok(direct.as_ref().unwrap().clone())
    }
    fn direct_compatible(&self, offset: u64, output: &[u8]) -> bool {
        self.offset
            .checked_add(offset)
            .is_some_and(|position| position % ALIGNMENT as u64 == 0)
            && output.as_ptr() as usize % ALIGNMENT == 0
    }
    fn read_bounced(&self, offset: u64, mut output: &mut [u8], scratch: &mut [u8]) -> Result<u64> {
        if offset
            .checked_add(output.len() as u64)
            .is_none_or(|end| end > self.length)
        {
            return Err("export read exceeds source".into());
        }
        let file = self.direct_file()?;
        let mut position = self.offset + offset;
        let mut transferred = 0;
        while !output.is_empty() {
            let lead = position as usize % ALIGNMENT;
            let count = output.len().min(scratch.len() - ALIGNMENT);
            let padded = (lead + count).next_multiple_of(ALIGNMENT);
            transfer_batch(
                &file,
                &[(
                    scratch.as_mut_ptr(),
                    padded,
                    position - lead as u64,
                    lead + count,
                )],
                false,
            )?;
            transferred += padded as u64;
            output[..count].copy_from_slice(&scratch[lead..lead + count]);
            output = &mut output[count..];
            position += count as u64;
        }
        Ok(transferred)
    }
    pub(crate) fn read_at(&self, offset: u64, output: &mut [u8]) -> Result<()> {
        if offset
            .checked_add(output.len() as u64)
            .is_none_or(|end| end > self.length)
        {
            return Err("export read exceeds source".into());
        }
        let position = self.offset + offset;
        if output.len() < ALIGNMENT {
            return read_exact_at(&self.file, output, position);
        }
        if self.direct_compatible(offset, output) {
            let file = self.direct_file()?;
            let bulk = output.len() / ALIGNMENT * ALIGNMENT;
            transfer_batch(&file, &[(output.as_mut_ptr(), bulk, position, bulk)], false)?;
            read_exact_at(&self.file, &mut output[bulk..], position + bulk as u64)
        } else {
            #[repr(align(4096))]
            struct Edge([u8; 64 * 1024]);
            let mut scratch = Edge([0; 64 * 1024]);
            self.read_bounced(offset, output, &mut scratch.0)
                .map(|_| ())
        }
    }
}

pub(crate) fn open_read(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Storage::FileSystem::{
            FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_DELETE, FILE_SHARE_READ,
        };
        options
            .share_mode(FILE_SHARE_READ | FILE_SHARE_DELETE)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = options.open(path)?;
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if file.metadata()?.file_attributes()
            & windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT
            != 0
        {
            return Err("cached blob must not be a reparse point".into());
        }
    }
    Identity::of(&file)?;
    Ok(file)
}
pub(crate) fn read_exact_at(file: &File, mut output: &mut [u8], mut offset: u64) -> Result<()> {
    while !output.is_empty() {
        #[cfg(unix)]
        let read = {
            use std::os::unix::fs::FileExt;
            file.read_at(output, offset)
        };
        #[cfg(windows)]
        let read = {
            use std::os::windows::fs::FileExt;
            file.seek_read(output, offset)
        };
        let count = match read {
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if count == 0 {
            return Err("truncated export source".into());
        }
        offset += count as u64;
        output = &mut output[count..];
    }
    Ok(())
}
pub(crate) fn write_all_at(file: &File, mut bytes: &[u8], mut offset: u64) -> Result<()> {
    while !bytes.is_empty() {
        #[cfg(unix)]
        let written = {
            use std::os::unix::fs::FileExt;
            file.write_at(bytes, offset)
        };
        #[cfg(windows)]
        let written = {
            use std::os::windows::fs::FileExt;
            file.seek_write(bytes, offset)
        };
        let count = match written {
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if count == 0 {
            return Err("zero-length export write".into());
        }
        offset += count as u64;
        bytes = &bytes[count..];
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn direct_reopen(file: &File, write: bool) -> Result<File> {
    use std::os::{fd::AsRawFd, unix::fs::OpenOptionsExt};
    Ok(OpenOptions::new()
        .read(!write)
        .write(write)
        .custom_flags(libc::O_DIRECT | libc::O_CLOEXEC)
        .open(format!("/proc/self/fd/{}", file.as_raw_fd()))?)
}
#[cfg(windows)]
fn direct_reopen(file: &File, write: bool) -> Result<File> {
    use std::os::windows::io::{AsRawHandle, FromRawHandle};
    use windows_sys::Win32::{Foundation::INVALID_HANDLE_VALUE, Storage::FileSystem::*};
    let access = if write {
        FILE_GENERIC_WRITE
    } else {
        FILE_GENERIC_READ
    };
    let flags = FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED;
    let handle = unsafe {
        ReOpenFile(
            file.as_raw_handle(),
            access,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            flags,
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(unsafe { File::from_raw_handle(handle) })
}

#[cfg(target_os = "linux")]
fn transfer_batch(file: &File, jobs: &[(*mut u8, usize, u64, usize)], write: bool) -> Result<()> {
    use io_uring::{IoUring, opcode, types};
    use std::os::fd::AsRawFd;
    let mut ring = IoUring::new(jobs.len().next_power_of_two() as u32)?;
    let mut done = vec![0usize; jobs.len()];
    for (i, &(pointer, length, offset, _)) in jobs.iter().enumerate() {
        let entry = if write {
            opcode::Write::new(types::Fd(file.as_raw_fd()), pointer, length as u32)
                .offset(offset)
                .build()
        } else {
            opcode::Read::new(types::Fd(file.as_raw_fd()), pointer, length as u32)
                .offset(offset)
                .build()
        }
        .user_data(i as u64);
        unsafe {
            ring.submission()
                .push(&entry)
                .map_err(|_| "export submission queue is full")?;
        }
    }
    let mut pending = jobs.len();
    let mut failure = None;
    while pending > 0 {
        match ring.submit_and_wait(1) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                // With no SQ polling, entries remaining in SQ have never reached the kernel.
                let queued = ring.submission().len();
                pending = pending.saturating_sub(queued);
                while pending > 0 {
                    let completed = ring.completion().count();
                    pending = pending.saturating_sub(completed);
                    if pending == 0 {
                        break;
                    }
                    let mut descriptor = libc::pollfd {
                        fd: ring.as_raw_fd(),
                        events: libc::POLLIN,
                        revents: 0,
                    };
                    // Submitted buffers remain owned until every completion has drained.
                    unsafe {
                        libc::poll(&mut descriptor, 1, -1);
                    }
                }
                return Err(e.into());
            }
        }
        let completions: Vec<_> = ring
            .completion()
            .map(|entry| (entry.user_data() as usize, entry.result()))
            .collect();
        for (i, result) in completions {
            if result <= 0 {
                failure.get_or_insert_with(|| {
                    if result == 0 {
                        "zero-length direct export write".to_owned()
                    } else {
                        std::io::Error::from_raw_os_error(-result).to_string()
                    }
                });
                pending -= 1;
                continue;
            }
            done[i] += result as usize;
            let (pointer, length, offset, required) = jobs[i];
            if done[i] >= required && done[i] <= length {
                pending -= 1;
            } else if failure.is_some() || done[i] > length || done[i] % ALIGNMENT != 0 {
                failure.get_or_insert_with(|| "unaligned partial direct export write".to_owned());
                pending -= 1;
            } else {
                let pointer = unsafe { pointer.add(done[i]) };
                let entry = if write {
                    opcode::Write::new(
                        types::Fd(file.as_raw_fd()),
                        pointer,
                        (length - done[i]) as u32,
                    )
                    .offset(offset + done[i] as u64)
                    .build()
                } else {
                    opcode::Read::new(
                        types::Fd(file.as_raw_fd()),
                        pointer,
                        (length - done[i]) as u32,
                    )
                    .offset(offset + done[i] as u64)
                    .build()
                }
                .user_data(i as u64);
                unsafe {
                    if ring.submission().push(&entry).is_err() {
                        failure.get_or_insert_with(|| "export submission queue is full".to_owned());
                        pending -= 1;
                    }
                }
            }
        }
    }
    if let Some(error) = failure {
        return Err(error.into());
    }
    Ok(())
}

#[cfg(windows)]
fn transfer_batch(file: &File, jobs: &[(*mut u8, usize, u64, usize)], write: bool) -> Result<()> {
    use std::{mem::zeroed, os::windows::io::AsRawHandle};
    use windows_sys::Win32::{
        Foundation::{CloseHandle, ERROR_IO_PENDING, GetLastError},
        Storage::FileSystem::{ReadFile, WriteFile},
        System::{
            IO::{GetOverlappedResult, OVERLAPPED},
            Threading::CreateEventW,
        },
    };
    let mut operations: Vec<Box<OVERLAPPED>> = Vec::with_capacity(jobs.len());
    let mut failure = None;
    for &(pointer, length, offset, _) in jobs {
        let mut operation: Box<OVERLAPPED> = Box::new(unsafe { zeroed() });
        operation.Anonymous.Anonymous.Offset = offset as u32;
        operation.Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
        operation.hEvent = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
        if operation.hEvent.is_null() {
            failure = Some(std::io::Error::last_os_error().to_string());
            break;
        }
        let result = unsafe {
            if write {
                WriteFile(
                    file.as_raw_handle(),
                    pointer,
                    length as u32,
                    std::ptr::null_mut(),
                    operation.as_mut(),
                )
            } else {
                ReadFile(
                    file.as_raw_handle(),
                    pointer,
                    length as u32,
                    std::ptr::null_mut(),
                    operation.as_mut(),
                )
            }
        };
        if result == 0 && unsafe { GetLastError() } != ERROR_IO_PENDING {
            let error = std::io::Error::last_os_error().to_string();
            unsafe {
                CloseHandle(operation.hEvent);
            }
            failure = Some(error);
            break;
        }
        operations.push(operation);
    }
    // Every submitted request drains before its stable OVERLAPPED and aligned buffer can drop.
    for (i, mut operation) in operations.into_iter().enumerate() {
        let (pointer, length, offset, required) = jobs[i];
        let mut done = 0;
        loop {
            let mut transferred = 0;
            if unsafe {
                GetOverlappedResult(
                    file.as_raw_handle(),
                    operation.as_mut(),
                    &mut transferred,
                    1,
                )
            } == 0
            {
                failure.get_or_insert_with(|| std::io::Error::last_os_error().to_string());
                break;
            }
            done += transferred as usize;
            if done >= required && done <= length {
                break;
            }
            if transferred == 0 || done > length || done % ALIGNMENT != 0 || failure.is_some() {
                failure
                    .get_or_insert_with(|| "unaligned partial native export transfer".to_owned());
                break;
            }
            operation.Internal = 0;
            operation.InternalHigh = 0;
            operation.Anonymous.Anonymous.Offset = (offset + done as u64) as u32;
            operation.Anonymous.Anonymous.OffsetHigh = ((offset + done as u64) >> 32) as u32;
            let result = unsafe {
                if write {
                    WriteFile(
                        file.as_raw_handle(),
                        pointer.add(done),
                        (length - done) as u32,
                        std::ptr::null_mut(),
                        operation.as_mut(),
                    )
                } else {
                    ReadFile(
                        file.as_raw_handle(),
                        pointer.add(done),
                        (length - done) as u32,
                        std::ptr::null_mut(),
                        operation.as_mut(),
                    )
                }
            };
            if result == 0 && unsafe { GetLastError() } != ERROR_IO_PENDING {
                failure.get_or_insert_with(|| std::io::Error::last_os_error().to_string());
                break;
            }
        }
        unsafe {
            CloseHandle(operation.hEvent);
        }
    }
    if let Some(error) = failure {
        return Err(error.into());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn available_memory() -> Result<u64> {
    let text = std::fs::read_to_string("/proc/meminfo")?;
    let kib = text
        .lines()
        .find_map(|line| {
            line.strip_prefix("MemAvailable:")
                .and_then(|v| v.split_whitespace().next())
                .and_then(|v| v.parse::<u64>().ok())
        })
        .ok_or("missing available host memory")?;
    Ok(kib.saturating_mul(1024))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    fn temp() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "flare-io-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ))
    }
    #[test]
    fn direct_tail_reads_and_writes_preserve_exact_length() {
        let path = temp();
        let file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let budget = IoBudget::for_ranks(1).unwrap();
        let size = BLOCK as u64 + 37;
        budget
            .write(&file, 0, size, |offset, bytes| {
                for (i, byte) in bytes.iter_mut().enumerate() {
                    *byte = ((offset + i as u64) % 251) as u8;
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(file.metadata().unwrap().len(), size);
        let source = Source::from_file(Arc::new(file), 3, size - 7).unwrap();
        let mut cursor = 3;
        budget
            .scan(&source, |bytes| {
                for byte in bytes {
                    assert_eq!(*byte, (cursor % 251) as u8);
                    cursor += 1;
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(cursor, size - 4);
        drop(source);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    #[cfg(unix)]
    fn source_mutation_and_overflow_are_rejected() {
        let path = temp();
        std::fs::write(&path, b"abcdef").unwrap();
        let source = Source::open(&path, 1, 4).unwrap();
        assert!(source.slice(u64::MAX, 2).is_err());
        assert!(source.read_at(3, &mut [0; 2]).is_err());
        std::fs::write(&path, b"replacement").unwrap();
        assert!(source.check_unchanged().is_err());
        drop(source);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn inactive_source_reopen_rejects_replacement_inodes() {
        let path = temp();
        std::fs::write(&path, b"abcdef").unwrap();
        let source = Source::open(&path, 1, 4).unwrap();
        let identity = source.identity();
        drop(source);
        let unchanged = Source::open_checked(&path, 1, 4, &identity).unwrap();
        drop(unchanged);
        let replacement = temp();
        std::fs::write(&replacement, b"abcdef").unwrap();
        std::fs::rename(&replacement, &path).unwrap();
        assert!(Source::open_checked(&path, 1, 4, &identity).is_err());
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn callback_error_releases_all_reusable_buffers() {
        let path = temp();
        let file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let budget = IoBudget::for_ranks(1).unwrap();
        assert!(
            budget
                .write(&file, 0, BLOCK as u64 + 1, |_, _| Err(
                    "injected fill error".into()
                ))
                .is_err()
        );
        let pool = budget.pool.lock().unwrap();
        assert_eq!(pool.idle.len(), pool.allocated);
        drop(pool);
        drop(file);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn unaligned_destination_preserves_neighbors_and_existing_extent() {
        let path = temp();
        let file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let size = BLOCK as u64 + 37;
        file.set_len(size + 100).unwrap();
        write_all_at(&file, b"before", 0).unwrap();
        write_all_at(&file, b"after", size + 7).unwrap();
        let budget = IoBudget::for_ranks(1).unwrap();
        budget
            .write(&file, 7, size, |_, bytes| {
                bytes.fill(13);
                Ok(())
            })
            .unwrap();
        assert_eq!(file.metadata().unwrap().len(), size + 100);
        let mut before = [0; 7];
        read_exact_at(&file, &mut before, 0).unwrap();
        assert_eq!(&before, b"before\0");
        let mut after = [0; 5];
        read_exact_at(&file, &mut after, size + 7).unwrap();
        assert_eq!(&after, b"after");
        let source = Source::from_file(Arc::new(file), 7, size).unwrap();
        budget
            .scan(&source, |bytes| {
                assert!(bytes.iter().all(|b| *b == 13));
                Ok(())
            })
            .unwrap();
        drop(source);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn writes_progress_while_producer_fills_subsequent_block() {
        let path = temp();
        let file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let budget = IoBudget::for_ranks(1).unwrap();
        let size = BLOCK as u64 + 13;
        budget
            .write_hashed(&file, 0, size, b"prefix", |offset, bytes| {
                if offset > 0 {
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                    while file.metadata()?.len() < BLOCK as u64 {
                        if std::time::Instant::now() > deadline {
                            return Err("writer did not overlap producer".into());
                        }
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }
                bytes.fill(17);
                Ok(())
            })
            .unwrap();
        drop(file);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn capture_and_publishing_buffers_fit_the_host_budget() {
        for available in [8u64 << 30, 16u64 << 30, 64u64 << 30, 1u64 << 40] {
            for ranks in [1, 2, 4, 8] {
                for capture in [false, true] {
                    let (main, scratch) = buffer_depths(ranks, available, capture).unwrap();
                    let reserved = if capture {
                        4 * BLOCK as u64 * ranks as u64
                    } else {
                        0
                    };
                    assert!((main + scratch) * ranks <= 128);
                    assert!(
                        ((main + scratch) * CAPACITY * ranks) as u64 + reserved
                            <= (1u64 << 30).min(available / 16)
                    );
                    assert!(main > 0);
                }
            }
        }
    }
    #[test]
    fn unaligned_bulk_uses_reserved_direct_scratch_with_one_output_slot() {
        let path = temp();
        let file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let mut budget = IoBudget::for_ranks(1).unwrap();
        let inner = Arc::get_mut(&mut budget).unwrap();
        inner.depth = 1;
        inner.read_depth = 1;
        let size = BLOCK as u64 + 100;
        budget
            .write(&file, 0, size, |offset, bytes| {
                for (i, byte) in bytes.iter_mut().enumerate() {
                    *byte = ((offset + i as u64) % 251) as u8;
                }
                Ok(())
            })
            .unwrap();
        let source = Source::from_file(Arc::new(file), 0, size).unwrap();
        let occupied = budget.buffer().unwrap();
        for offset in [8, 16] {
            for alignment in [1, 8, 4095] {
                let mut output = vec![0; BLOCK + 37 + alignment];
                budget
                    .read_at(&source, offset, &mut output[alignment..])
                    .unwrap();
                assert!(
                    output[alignment..]
                        .iter()
                        .enumerate()
                        .all(|(i, b)| *b == ((offset + i as u64) % 251) as u8)
                );
            }
        }
        assert!(budget.bounced_bytes.load(Ordering::Relaxed) > 6 * BLOCK as u64);
        assert_eq!(budget.pool.lock().unwrap().allocated, 1);
        assert_eq!(budget.read_pool.lock().unwrap().allocated, 1);
        drop(occupied);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        budget
                            .hash_generated(BLOCK as u64 + 37, b"", |offset, bytes| {
                                budget.read_at(&source, offset + 8, bytes)
                            })
                            .unwrap()
                    })
                })
                .collect();
            let hashes: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            assert!(hashes.iter().all(|h| *h == hashes[0]));
        });
        assert_eq!(budget.pool.lock().unwrap().allocated, 1);
        assert_eq!(budget.read_pool.lock().unwrap().allocated, 1);
        drop(source);
        std::fs::remove_file(path).unwrap();
    }
}
#[cfg(windows)]
fn available_memory() -> Result<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of_val(&status) as u32;
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(status.ullAvailPhys)
}
