//! Shared copy proposals from a bounded window of committed response tokens.

use std::collections::{HashMap, hash_map::Entry};
use std::hash::{BuildHasherDefault, Hash, Hasher};

const CAPACITY: usize = 65_536;
const END: u64 = u64::MAX;

// Keys make one write containing their entire initialized token array.
#[derive(Default)]
struct TokenHasher(u64);

impl Hasher for TokenHasher {
    fn finish(&self) -> u64 { self.0 }
    fn write(&mut self, bytes: &[u8]) {
        self.0 = twox_hash::XxHash3_64::oneshot(bytes);
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct Key<const N: usize>([u32; N]);

impl<const N: usize> Hash for Key<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // SAFETY: u32 arrays have no padding and every element is initialized.
        let bytes = unsafe {
            std::slice::from_raw_parts(self.0.as_ptr().cast::<u8>(), N * size_of::<u32>())
        };
        state.write(bytes);
    }
}

type Index<const N: usize> = HashMap<Key<N>, u64, BuildHasherDefault<TokenHasher>>;

#[derive(Clone, Copy)]
struct Token {
    next: u64,
    id: u32,
}

/// Tail of one generated response, retained through in-memory preemption.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DecodedSequence {
    positions: [u64; 8],
    tokens: [u32; 8],
    len: usize,
}

impl DecodedSequence {
    fn push(&mut self, position: u64, token: u32) {
        if self.len == 8 {
            self.positions.copy_within(1.., 0);
            self.tokens.copy_within(1.., 0);
        } else {
            self.len += 1;
        }
        self.positions[self.len - 1] = position;
        self.tokens[self.len - 1] = token;
    }
}

pub(crate) struct DecodedHistory {
    ring: Vec<Token>,
    eight: Index<8>,
    four: Index<4>,
    end: u64,
}

impl DecodedHistory {
    pub(crate) fn new() -> Self { Self::with_capacity(CAPACITY) }

    fn with_capacity(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two());
        Self {
            ring: vec![Token { next: END, id: 0 }; capacity],
            // Leave room for deleted buckets so churn can reclaim them without growing.
            eight: Index::with_capacity_and_hasher(capacity * 2, Default::default()),
            four: Index::with_capacity_and_hasher(capacity * 2, Default::default()),
            end: 0,
        }
    }

    fn start(&self) -> u64 { self.end.saturating_sub(self.ring.len() as u64) }
    fn slot(&self, position: u64) -> usize { position as usize & (self.ring.len() - 1) }

    fn token(&self, position: u64) -> Option<Token> {
        (self.start() <= position && position < self.end).then(|| self.ring[self.slot(position)])
    }

    fn expire(&mut self) {
        let expired = self.start();
        let mut position = expired;
        let mut key = [0; 8];
        let mut count = 0;
        while count < 8 {
            let Some(token) = self.token(position) else { break };
            key[count] = token.id;
            count += 1;
            position = token.next;
        }
        if count >= 4 {
            if let Entry::Occupied(entry) = self.four.entry(Key(key[..4].try_into().unwrap())) {
                if *entry.get() == expired { entry.remove(); }
            }
        }
        if count == 8 {
            if let Entry::Occupied(entry) = self.eight.entry(Key(key)) {
                if *entry.get() == expired { entry.remove(); }
            }
        }
    }

    pub(crate) fn append(&mut self, sequence: &mut DecodedSequence, token: u32) {
        if self.end >= self.ring.len() as u64 { self.expire(); }
        let position = self.end;
        let slot = self.slot(position);
        let start = (position + 1).saturating_sub(self.ring.len() as u64);
        self.ring[slot] = Token { next: END, id: token };
        if sequence.len > 0 {
            let previous = sequence.positions[sequence.len - 1];
            if previous >= start {
                let slot = self.slot(previous);
                self.ring[slot].next = position;
            }
        }
        // Index only contexts with an observed continuation. A delayed response
        // must not replace a more recent occurrence of the same context.
        if sequence.len >= 4 && sequence.positions[sequence.len - 4] >= start {
            let position = sequence.positions[sequence.len - 4];
            let key = Key(sequence.tokens[sequence.len - 4..sequence.len].try_into().unwrap());
            self.four.entry(key).and_modify(|old| *old = (*old).max(position)).or_insert(position);
        }
        if sequence.len == 8 && sequence.positions[0] >= start {
            let position = sequence.positions[0];
            self.eight.entry(Key(sequence.tokens))
                .and_modify(|old| *old = (*old).max(position)).or_insert(position);
        }
        sequence.push(position, token);
        self.end += 1;
    }

    fn continuation(&self, mut position: u64, length: usize, maximum: usize) -> ([i64; 7], usize) {
        for _ in 0..length {
            position = self.token(position).expect("indexed context is retained").next;
        }
        let mut output = [-1; 7];
        let mut count = 0;
        while count < maximum.min(7) {
            let Some(token) = self.token(position) else { break };
            output[count] = i64::from(token.id);
            count += 1;
            position = token.next;
        }
        (output, count)
    }

    pub(crate) fn lookup(&self, context: &[u32], maximum: usize) -> ([i64; 7], usize) {
        if maximum == 0 { return ([-1; 7], 0) }
        if context.len() >= 8 {
            if let Some(&position) = self.eight.get(&Key(context[context.len() - 8..].try_into().unwrap())) {
                return self.continuation(position, 8, maximum);
            }
        }
        if context.len() >= 4 {
            if let Some(&position) = self.four.get(&Key(context[context.len() - 4..].try_into().unwrap())) {
                return self.continuation(position, 4, maximum);
            }
        }
        ([-1; 7], 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn append(cache: &mut DecodedHistory, sequence: &mut DecodedSequence, tokens: &[u32]) {
        for &token in tokens { cache.append(sequence, token); }
    }

    #[test]
    fn matches_longest_context_then_most_recent_occurrence_and_bounds_output() {
        let mut cache = DecodedHistory::with_capacity(64);
        let mut first = DecodedSequence::default();
        append(&mut cache, &mut first, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(cache.lookup(&[1, 2, 3, 4, 5, 6, 7, 8], 7).1, 0);
        append(&mut cache, &mut first, &[9, 10, 11, 12, 13, 14, 15, 16]);
        let mut second = DecodedSequence::default();
        append(&mut cache, &mut second, &[5, 6, 7, 8, 90, 91]);
        assert_eq!(cache.lookup(&[1, 2, 3, 4, 5, 6, 7, 8], 99), ([9, 10, 11, 12, 13, 14, 15], 7));
        assert_eq!(cache.lookup(&[5, 6, 7, 8], 7), ([90, 91, -1, -1, -1, -1, -1], 2));
        assert_eq!(cache.lookup(&[5, 6, 7, 8], 1), ([90, -1, -1, -1, -1, -1, -1], 1));
        assert_eq!(cache.lookup(&[5, 6, 7, 8], 0), ([-1; 7], 0));
        assert_eq!(cache.lookup(&[6, 7, 8], 7), ([-1; 7], 0));
    }

    #[test]
    fn delayed_continuation_does_not_replace_a_more_recent_context() {
        let mut cache = DecodedHistory::with_capacity(32);
        let mut delayed = DecodedSequence::default();
        let mut recent = DecodedSequence::default();
        append(&mut cache, &mut delayed, &[1, 2, 3, 4]);
        append(&mut cache, &mut recent, &[1, 2, 3, 4, 90]);
        cache.append(&mut delayed, 50);
        assert_eq!(cache.lookup(&[1, 2, 3, 4], 7), ([90, -1, -1, -1, -1, -1, -1], 1));
        let mut noise = DecodedSequence::default();
        append(&mut cache, &mut noise, &[99; 26]);
        assert_eq!(cache.lookup(&[1, 2, 3, 4], 7).0[0], 90);
        cache.append(&mut noise, 99);
        assert_eq!(cache.lookup(&[1, 2, 3, 4], 7).1, 0);
        append(&mut cache, &mut delayed, &[1, 2, 3, 4, 60]);
        assert_eq!(cache.lookup(&[1, 2, 3, 4], 7).0[0], 60);
    }

    fn oracle(events: &[(u64, u32)], context: &[u32], capacity: usize) -> ([i64; 7], usize) {
        let mut sequences = std::collections::BTreeMap::<u64, Vec<(usize, u32)>>::new();
        for (position, &(sequence, token)) in events.iter().enumerate().skip(events.len().saturating_sub(capacity)) {
            sequences.entry(sequence).or_default().push((position, token));
        }
        for length in [8, 4] {
            if context.len() < length { continue; }
            let mut best = None;
            for sequence in sequences.values() {
                for start in 0..sequence.len().saturating_sub(length) {
                    if sequence[start..start + length].iter().map(|v| v.1)
                        .eq(context[context.len() - length..].iter().copied())
                        && best.as_ref().is_none_or(|(position, _, _)| sequence[start].0 > *position)
                    {
                        let count = 7.min(sequence.len() - start - length);
                        let mut output = [-1; 7];
                        for i in 0..count { output[i] = i64::from(sequence[start + length + i].1); }
                        best = Some((sequence[start].0, output, count));
                    }
                }
            }
            if let Some((_, output, count)) = best { return (output, count); }
        }
        ([-1; 7], 0)
    }

    #[test]
    fn interleaving_eviction_and_slot_reuse_match_segmented_window_search() {
        for capacity in [16, 32, 128, 512] {
            let mut cache = DecodedHistory::with_capacity(capacity);
            let mut sequences = [DecodedSequence::default(); 4];
            let mut ids = [0, 1, 2, 3];
            let mut next_id = 4;
            let mut counts = [0; 4];
            let mut events = Vec::new();
            for step in 0..5000 {
                let lane = if step % 173 < 97 { 0 } else { step % 4 };
                if counts[lane] > 0 && counts[lane] % 107 == 0 {
                    sequences[lane] = DecodedSequence::default();
                    ids[lane] = next_id;
                    next_id += 1;
                }
                let token = if counts[lane] % 32 < 16 { counts[lane] % 16 } else { step + 100 } as u32;
                cache.append(&mut sequences[lane], token);
                counts[lane] += 1;
                events.push((ids[lane], token));
                for source in [lane, (lane + 1) % 4] {
                    let query = &sequences[source].tokens[..sequences[source].len];
                    assert_eq!(cache.lookup(query, 7), oracle(&events, query, capacity),
                        "capacity={capacity}, step={step}, source={source}");
                }
                assert!(cache.eight.len() <= capacity && cache.four.len() <= capacity);
            }
        }
    }

    #[test]
    fn production_window_evicts_without_growing_storage() {
        let mut cache = DecodedHistory::new();
        let ring_capacity = cache.ring.capacity();
        let index_capacity = (cache.eight.capacity(), cache.four.capacity());
        let mut sequences = [DecodedSequence::default(); 4];
        for position in 0..CAPACITY * 4 {
            cache.append(&mut sequences[position % 4], position as u32);
        }
        assert_eq!(cache.ring.capacity(), ring_capacity);
        assert!(cache.eight.capacity() <= index_capacity.0 && cache.four.capacity() <= index_capacity.1);
        assert_eq!(cache.lookup(&[0, 4, 8, 12], 7).1, 0);
        let start = (CAPACITY * 3) as u32;
        assert_eq!(cache.lookup(&[start, start + 4, start + 8, start + 12], 7).0[0], i64::from(start + 16));
    }
}
