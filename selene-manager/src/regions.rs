//! Stable mappings of the manager's shared arena and staging.

use crate::{arena::Arena, data::invalid};
use flare_selene::shared::region::Region;
use std::io;

#[derive(Clone)]
pub struct LocalRegions {
    arena: Arena,
    staging: Arena,
}

impl LocalRegions {
    pub fn new(arena: Arena, staging: Arena) -> Self {
        Self { arena, staging }
    }

    pub fn mapping(&self, region: Region) -> (*mut u8, usize) {
        match region {
            Region::Arena => self.arena.mapping(),
            Region::Staging => self.staging.mapping(),
        }
    }

    /// # Safety
    /// The caller must own the range exclusively while the slice lives, and keep
    /// this mapping alive. The store leases windows until their handler is fenced.
    pub unsafe fn bytes<'a>(
        &self,
        region: Region,
        offset: u64,
        length: u64,
    ) -> io::Result<&'a mut [u8]> {
        let (base, size) = self.mapping(region);
        let end = offset
            .checked_add(length)
            .ok_or_else(|| invalid("window range overflow"))?;
        if end > size as u64 {
            return Err(invalid("a window lies outside its region"));
        }
        // SAFETY: bounds are checked; ownership and lifetime are the caller's contract.
        Ok(unsafe { std::slice::from_raw_parts_mut(base.add(offset as usize), length as usize) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regions() -> LocalRegions {
        LocalRegions::new(
            Arena::new(64 * 1024).unwrap(),
            Arena::new(16 * 1024).unwrap(),
        )
    }

    #[test]
    fn reads_and_writes_hit_the_mapped_bytes() {
        let regions = regions();
        // SAFETY: this test exclusively owns the mappings and each borrow ends before the next.
        unsafe {
            regions
                .bytes(Region::Arena, 4096, 3)
                .unwrap()
                .copy_from_slice(b"abc");
            assert_eq!(regions.bytes(Region::Arena, 4096, 3).unwrap(), b"abc");
        }
    }

    #[test]
    fn the_mapping_is_stable_across_accesses() {
        let regions = regions();
        assert_eq!(
            regions.mapping(Region::Arena),
            regions.mapping(Region::Arena)
        );
        assert_eq!(regions.mapping(Region::Arena).1, 64 * 1024);
        assert_eq!(regions.mapping(Region::Staging).1, 16 * 1024);
    }

    #[test]
    fn ranges_outside_a_region_are_rejected() {
        let regions = regions();
        // SAFETY: the test owns both mappings; these invalid ranges never yield a slice.
        unsafe {
            assert!(regions.bytes(Region::Arena, 64 * 1024, 1).is_err());
            assert!(regions.bytes(Region::Staging, 16 * 1024, 1).is_err());
            assert!(regions.bytes(Region::Arena, u64::MAX, 2).is_err());
            assert!(regions.bytes(Region::Staging, 32 * 1024, 1).is_err());
        }
    }
}
