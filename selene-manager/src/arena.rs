use crate::{Result, invalid};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::OpenOptions,
    os::unix::fs::OpenOptionsExt,
    path::Path,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

struct Mapping {
    pointer: NonNull<u8>,
    length: usize,
    free: Mutex<BTreeMap<usize, BTreeSet<usize>>>,
}

// Allocations never overlap; a mutable slice requires exclusive ownership of its allocation.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.pointer.as_ptr().cast(), self.length);
        }
    }
}

#[derive(Clone)]
pub struct Arena(Arc<Mapping>);

pub struct Extent {
    mapping: Arc<Mapping>,
    offset: usize,
    capacity: usize,
    length: usize,
}

impl Arena {
    pub fn new(length: usize) -> Result<Self> {
        let length = length / 4096 * 4096;
        if length == 0 {
            return Err(invalid("arena must hold at least 4096 bytes"));
        }
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if pointer == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Self::from_pointer(pointer, length)
    }

    /// Maps a manager region file shared with clients (`MAP_SHARED`).
    ///
    /// The file must already exist and be exactly `length` bytes, rounded down to a page.
    pub fn open(path: &Path, length: usize) -> Result<Self> {
        let length = length / 4096 * 4096;
        if length == 0 {
            return Err(invalid("arena must hold at least 4096 bytes"));
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)?;
        let size = file.metadata()?.len();
        if size != length as u64 {
            return Err(invalid(format!(
                "region file {} has size {size}, expected {length}",
                path.display()
            )));
        }
        let pointer = flare_selene::shared::region::map_file(&file, length)?;
        Self::from_pointer(pointer, length)
    }

    fn from_pointer(pointer: *mut libc::c_void, length: usize) -> Result<Self> {
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(pointer, length, libc::MADV_HUGEPAGE);
        }
        let mut free: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
        let mut offset = 0;
        let mut remaining = length;
        while remaining > 0 {
            let size = 1usize << (usize::BITS - 1 - remaining.leading_zeros());
            free.entry(size).or_default().insert(offset);
            offset += size;
            remaining -= size;
        }
        Ok(Self(Arc::new(Mapping {
            pointer: NonNull::new(pointer.cast()).unwrap(),
            length,
            free: Mutex::new(free),
        })))
    }

    pub fn allocate(&self, length: usize) -> Option<Extent> {
        let capacity = length.max(4096).checked_next_power_of_two()?;
        let mut free = self.0.free.lock().unwrap();
        let (&size, _) = free
            .range(capacity..)
            .find(|(_, offsets)| !offsets.is_empty())?;
        let offset = free.get_mut(&size).unwrap().pop_first().unwrap();
        let mut split = size;
        while split > capacity {
            split /= 2;
            free.entry(split).or_default().insert(offset + split);
        }
        Some(Extent {
            mapping: self.0.clone(),
            offset,
            capacity,
            length,
        })
    }

    /// Takes everything that is free, so nothing can be allocated until the extents drop.
    pub fn reserve_all(&self) -> Vec<Extent> {
        let mut held = Vec::new();
        let mut size = 1usize << (usize::BITS - 1 - self.0.length.leading_zeros());
        while size >= 4096 {
            while let Some(extent) = self.allocate(size) {
                held.push(extent);
            }
            size /= 2;
        }
        held
    }

    /// Whether an allocation of `length` bytes could be served without freeing anything.
    pub fn can_allocate(&self, length: usize) -> bool {
        let Some(capacity) = length.max(4096).checked_next_power_of_two() else {
            return false;
        };
        let free = self.0.free.lock().unwrap();
        free.range(capacity..)
            .any(|(_, offsets)| !offsets.is_empty())
    }

    pub fn used(&self) -> usize {
        let free = self.0.free.lock().unwrap();
        self.0.length
            - free
                .iter()
                .map(|(size, offsets)| size * offsets.len())
                .sum::<usize>()
    }

    pub fn capacity(&self) -> usize {
        self.0.length
    }
    /// Base pointer and full mapped length of the region. Valid for the life of the
    /// mapping and stable across allocations.
    pub fn mapping(&self) -> (*mut u8, usize) {
        (self.0.pointer.as_ptr(), self.0.length)
    }
}

/// A window-sized view into a shared extent. Holding it keeps the extent
/// allocation alive, so a queued write can read a staging hold slice after the
/// window itself has been acknowledged.
#[derive(Clone)]
pub struct ExtentSlice {
    extent: std::sync::Arc<Extent>,
    within: usize,
    length: usize,
}

impl ExtentSlice {
    pub fn new(extent: std::sync::Arc<Extent>, within: usize, length: usize) -> Self {
        Self {
            extent,
            within,
            length,
        }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.extent.bytes()[self.within..self.within + self.length]
    }
}

impl Extent {
    pub fn bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.mapping.pointer.as_ptr().add(self.offset), self.length)
        }
    }
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.mapping.pointer.as_ptr().add(self.offset),
                self.length,
            )
        }
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Absolute offset of this extent inside its region.
    pub fn offset(&self) -> usize {
        self.offset
    }
    fn free(&self) {
        let mut free = self.mapping.free.lock().unwrap();
        let mut offset = self.offset;
        let mut size = self.capacity;
        loop {
            let buddy = offset ^ size;
            if !free
                .get_mut(&size)
                .is_some_and(|offsets| offsets.remove(&buddy))
            {
                break;
            }
            offset = offset.min(buddy);
            size *= 2;
        }
        free.entry(size).or_default().insert(offset);
    }
}

impl Drop for Extent {
    fn drop(&mut self) {
        self.free();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buddy_allocation_and_reuse() {
        let arena = Arena::new(8192).unwrap();
        let first = arena.allocate(4096).unwrap();
        let second = arena.allocate(4096).unwrap();
        assert_eq!(first.capacity(), 4096);
        assert!(!arena.can_allocate(4096));
        drop(first);
        assert!(arena.can_allocate(4096));
        drop(second);
        assert_eq!(arena.used(), 0);
        assert!(arena.allocate(8192).is_some());
    }

    #[test]
    fn offsets_are_page_aligned_and_bounded() {
        let arena = Arena::new(16384).unwrap();
        let extent = arena.allocate(9000).unwrap();
        assert_eq!(extent.offset() % 4096, 0);
        assert_eq!(extent.capacity(), 16384);
        assert!(extent.offset() + extent.capacity() <= arena.capacity());
    }

    #[test]
    fn independent_anonymous_arenas_do_not_alias() {
        let left = Arena::new(4096).unwrap();
        let right = Arena::new(4096).unwrap();
        let mut a = left.allocate(4096).unwrap();
        let mut b = right.allocate(4096).unwrap();
        assert_eq!(a.offset(), b.offset());
        a.bytes_mut()[0] = 0xaa;
        assert_eq!(b.bytes()[0], 0);
        b.bytes_mut()[0] = 0xbb;
        assert_eq!(a.bytes()[0], 0xaa);
        assert_eq!(b.bytes()[0], 0xbb);
    }

    #[test]
    fn arc_holds_the_allocation() {
        let arena = Arena::new(8192).unwrap();
        let extent = Arc::new(arena.allocate(4096).unwrap());
        let held = extent.clone();
        assert_eq!(Arc::strong_count(&extent), 2);
        drop(extent);
        // The holder still keeps the allocation alive even though the store dropped its handle.
        assert_eq!(arena.used(), 4096);
        assert!(!arena.can_allocate(8192));
        drop(held);
        assert_eq!(arena.used(), 0);
    }
}
