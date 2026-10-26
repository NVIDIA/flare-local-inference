use blake2::digest::consts::U16;
use blake2::{Blake2b, Digest};

use crate::Result;

pub use flare_selene::{HASH_BLOCK_TOKENS, PrefixKey};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConditioningEvent {
    pub position: u64,
    pub identity: [u8; 32],
}

/// Incremental token and conditioning identity for a single model/recipe.
/// Hash blocks contain 64 tokens independently of physical cache page geometry.
///
/// Events belong to token positions in the appended segment. Events at the same
/// position retain caller order. A boundary at N contains positions less than N;
/// appending conditioning at N cannot change that boundary's digest.
#[derive(Clone, Debug)]
pub struct PrefixHistory {
    identity: [u8; 32],
    boundaries: Vec<PrefixKey>,
    tail: Vec<u32>,
    tail_events: Vec<ConditioningEvent>,
    tokens: Vec<u32>,
    events: Vec<ConditioningEvent>,
}

impl PrefixHistory {
    pub fn new(identity: [u8; 32]) -> Self {
        let mut hash = Blake2b::<U16>::new();
        hash.update(b"flare-prefix-root-v1\0");
        hash.update(identity);
        Self {
            identity,
            boundaries: vec![PrefixKey {
                consumed_tokens: 0,
                digest: hash.finalize().into(),
            }],
            tail: Vec::with_capacity(HASH_BLOCK_TOKENS),
            tail_events: Vec::new(),
            tokens: Vec::new(),
            events: Vec::new(),
        }
    }

    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }

    pub fn token_count(&self) -> u64 {
        self.boundaries.last().unwrap().consumed_tokens + self.tail.len() as u64
    }

    pub fn key_at(&self, consumed_tokens: u64) -> Option<PrefixKey> {
        let end = usize::try_from(consumed_tokens).ok()?;
        if end > self.tokens.len() {
            return None;
        }
        let index = end / HASH_BLOCK_TOKENS;
        let parent = *self.boundaries.get(index)?;
        if end.is_multiple_of(HASH_BLOCK_TOKENS) {
            return Some(parent);
        }
        let start = index * HASH_BLOCK_TOKENS;
        let mut hash = Blake2b::<U16>::new();
        hash.update(b"flare-prefix-tail-v1\0");
        hash.update(self.identity);
        hash.update(parent.digest);
        hash.update(consumed_tokens.to_le_bytes());
        for token in &self.tokens[start..end] {
            hash.update(token.to_le_bytes());
        }
        let first = self.events.partition_point(|e| e.position < start as u64);
        let last = self
            .events
            .partition_point(|e| e.position < consumed_tokens);
        hash.update(((last - first) as u64).to_le_bytes());
        for event in &self.events[first..last] {
            hash.update(event.position.to_le_bytes());
            hash.update(event.identity);
        }
        Some(PrefixKey {
            consumed_tokens,
            digest: hash.finalize().into(),
        })
    }

    pub fn aligned_boundaries(&self) -> &[PrefixKey] {
        &self.boundaries[1..]
    }

    /// Digest of the aligned boundary at or below `consumed_tokens`, which keys
    /// every endpoint inside that hash block in host-store buckets.
    pub fn bucket_at(&self, consumed_tokens: u64) -> Option<[u8; 16]> {
        let aligned = consumed_tokens - consumed_tokens % HASH_BLOCK_TOKENS as u64;
        self.key_at(aligned).map(|key| key.digest)
    }

    pub fn append(&mut self, tokens: &[u32], events: &[ConditioningEvent]) -> Result<()> {
        let start = self.token_count();
        let end = start
            .checked_add(tokens.len() as u64)
            .ok_or("token length overflow")?;
        let mut last_position = start;
        for event in events {
            if event.position < last_position || event.position >= end {
                return Err(
                    "conditioning events must be ordered within the appended token segment".into(),
                );
            }
            last_position = event.position;
        }
        self.tokens.extend_from_slice(tokens);
        self.events.extend_from_slice(events);
        let mut event_index = 0;
        for (index, token) in tokens.iter().enumerate() {
            let position = start + index as u64;
            while event_index < events.len() && events[event_index].position == position {
                self.tail_events.push(events[event_index]);
                event_index += 1;
            }
            self.tail.push(*token);
            if self.tail.len() == HASH_BLOCK_TOKENS {
                let parent = *self.boundaries.last().unwrap();
                let consumed_tokens = parent.consumed_tokens + HASH_BLOCK_TOKENS as u64;
                let mut hash = Blake2b::<U16>::new();
                hash.update(b"flare-prefix-block-v1\0");
                hash.update(self.identity);
                hash.update(parent.digest);
                hash.update(consumed_tokens.to_le_bytes());
                for token in &self.tail {
                    hash.update(token.to_le_bytes());
                }
                hash.update((self.tail_events.len() as u64).to_le_bytes());
                for event in &self.tail_events {
                    hash.update(event.position.to_le_bytes());
                    hash.update(event.identity);
                }
                self.boundaries.push(PrefixKey {
                    consumed_tokens,
                    digest: hash.finalize().into(),
                });
                self.tail.clear();
                self.tail_events.clear();
            }
        }
        Ok(())
    }
}

impl flare_selene::History for PrefixHistory {
    fn token_count(&self) -> u64 {
        PrefixHistory::token_count(self)
    }

    fn key_at(&self, consumed_tokens: u64) -> Option<PrefixKey> {
        PrefixHistory::key_at(self, consumed_tokens)
    }

    fn aligned_boundaries(&self) -> &[PrefixKey] {
        PrefixHistory::aligned_boundaries(self)
    }
}
