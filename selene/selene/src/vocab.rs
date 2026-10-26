//! Identifiers and keys shared with the GPU tier that owns the checkpoints.

/// Tokens per hash block; aligned boundaries and bucket digests use this unit.
pub const HASH_BLOCK_TOKENS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CheckpointId(u64);

impl CheckpointId {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn value(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateSlotId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrefixKey {
    pub consumed_tokens: u64,
    pub digest: [u8; 16],
}

/// A request's token and conditioning history as the index reads it.
pub trait History {
    fn token_count(&self) -> u64;
    fn key_at(&self, consumed_tokens: u64) -> Option<PrefixKey>;
    /// Keys at every multiple of [`HASH_BLOCK_TOKENS`] up to the history's length.
    fn aligned_boundaries(&self) -> &[PrefixKey];
}
