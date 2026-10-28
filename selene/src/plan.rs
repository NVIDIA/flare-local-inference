//! One source per restore: longest match, then closest copy level.

use std::cmp::Reverse;

use crate::levels::Level;
use crate::{Error, History, PrefixKey, Result};

/// Whose cache a prefix belongs to. It is checked by the sources; the plan carries
/// it so every part is validated against the same identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    pub model: [u8; 32],
    pub layout: [u8; 16],
}

/// Names a place that holds records: a local tier, a neighbour or a manager.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SourceId(pub u64);

/// One record of a candidate's chain: its key and the pages of its own tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainNode {
    pub key: PrefixKey,
    pub tail_pages: usize,
}

/// A source that can supply the pages from `begin_tokens` to the endpoint `key`
/// and the endpoint's row, from a local tier or from a pool lookup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub level: Level,
    pub source: SourceId,
    pub key: PrefixKey,
    /// The first token this candidate supplies, exact, not necessarily page aligned.
    pub begin_tokens: u64,
    /// Pages of `nodes[0]`'s tail before the page that holds `begin_tokens`.
    pub skip_pages: usize,
    /// Records from the one holding `begin_tokens` up to the endpoint, ascending.
    pub nodes: Vec<ChainNode>,
}

/// What the plan is built for.
pub struct Request<'a> {
    pub history: &'a dyn History,
    pub identity: &'a Identity,
    pub page_tokens: u64,
    /// Tokens already in GPU memory.
    pub resident: u64,
    /// The endpoint must be strictly below this.
    pub before: u64,
}

/// One source's share of the restore.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Part {
    pub level: Level,
    pub source: SourceId,
    /// The first token this part supplies.
    pub begin_tokens: u64,
    /// The last record's key; the part supplies the tokens before it.
    pub end: PrefixKey,
    /// Pages of the first record in `chain` to skip.
    pub first_page: usize,
    /// Offset into the ticket's pages where this part's pages go.
    pub dest_page: usize,
    pub pages: usize,
    /// Whether this part fetches the endpoint's continuation row.
    pub row: bool,
    /// Keys of the records read, ascending, each verified against the history.
    pub chain: Vec<PrefixKey>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub identity: Identity,
    pub resident: u64,
    pub part: Part,
}
impl Plan {
    pub fn endpoint(&self) -> PrefixKey {
        self.part.end
    }
    pub fn pages(&self) -> usize {
        self.part.pages
    }
    pub fn validate(&self, history: &dyn History, page_tokens: u64) -> Result<()> {
        let part = &self.part;
        let expected =
            (part.end.consumed_tokens / page_tokens).checked_sub(self.resident / page_tokens);
        if part.begin_tokens != self.resident
            || part.dest_page != 0
            || !part.row
            || expected != Some(part.pages as u64)
            || part.end.consumed_tokens <= self.resident
            || history.key_at(part.end.consumed_tokens) != Some(part.end)
            || part.chain.last() != Some(&part.end)
            || part
                .chain
                .iter()
                .any(|key| history.key_at(key.consumed_tokens) != Some(*key))
        {
            return Err(Error::new("restore source is inconsistent"));
        }
        Ok(())
    }
}

pub fn plan(request: &Request, candidates: &[Candidate]) -> Option<Plan> {
    let source = candidates
        .iter()
        .filter(|candidate| {
            candidate.begin_tokens <= request.resident
                && candidate.key.consumed_tokens > request.resident
                && candidate.key.consumed_tokens < request.before
                && valid(request, candidate)
        })
        .min_by_key(|candidate| {
            (
                Reverse(candidate.key.consumed_tokens),
                candidate.level as usize,
                candidate.source.0,
            )
        })?;
    Some(Plan {
        identity: request.identity.clone(),
        resident: request.resident,
        part: part(request, source, request.resident, 0, true),
    })
}

/// The page where each node's tail starts, if the chain is consistent.
fn tail_starts(page_tokens: u64, c: &Candidate) -> Option<Vec<usize>> {
    let mut starts = Vec::with_capacity(c.nodes.len());
    let mut previous: Option<usize> = None;
    for node in &c.nodes {
        let end = (node.key.consumed_tokens / page_tokens) as usize;
        let start = end.checked_sub(node.tail_pages)?;
        if previous.is_some_and(|p| p != start) {
            return None;
        }
        starts.push(start);
        previous = Some(end);
    }
    Some(starts)
}

/// A candidate is a hint: its keys must be the history's and its records must
/// chain page by page from the page holding `begin_tokens` to the endpoint.
fn valid(request: &Request, c: &Candidate) -> bool {
    let pt = request.page_tokens;
    let end = c.key.consumed_tokens;
    let Some(starts) = tail_starts(pt, c).filter(|starts| !starts.is_empty()) else {
        return false;
    };
    c.level != Level::ThisGpu
        && c.begin_tokens <= end
        && end <= request.history.token_count()
        && request.history.key_at(end) == Some(c.key)
        && c.nodes.last().is_some_and(|node| node.key == c.key)
        && c.nodes
            .windows(2)
            .all(|w| w[0].key.consumed_tokens < w[1].key.consumed_tokens)
        && c.nodes
            .iter()
            .all(|n| request.history.key_at(n.key.consumed_tokens) == Some(n.key))
        && starts[0] + c.skip_pages == (c.begin_tokens / pt) as usize
}

/// The part reading `c` from token `begin`, which `c` must cover.
fn part(request: &Request, c: &Candidate, begin: u64, dest_page: usize, row: bool) -> Part {
    let pt = request.page_tokens;
    let start = (begin / pt) as usize;
    let end_page = (c.key.consumed_tokens / pt) as usize;
    let starts = tail_starts(pt, c).expect("validated candidate");
    // The record holding the first page; at the end of the chain, the last record.
    let first = c
        .nodes
        .iter()
        .position(|n| (n.key.consumed_tokens / pt) as usize > start)
        .unwrap_or(c.nodes.len() - 1);
    Part {
        level: c.level,
        source: c.source,
        begin_tokens: begin,
        end: c.key,
        first_page: start - starts[first],
        dest_page,
        pages: end_page - start,
        row,
        chain: c.nodes[first..].iter().map(|n| n.key).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestHistory;
    struct Fixture {
        history: TestHistory,
        identity: Identity,
    }
    impl Fixture {
        fn new() -> Self {
            let mut history = TestHistory::new([7; 32]);
            history
                .append(&(0..2000).collect::<Vec<u32>>(), &[])
                .unwrap();
            Self {
                history,
                identity: Identity {
                    model: [7; 32],
                    layout: [1; 16],
                },
            }
        }
        fn candidate(&self, level: Level, source: u64, ends: &[u64]) -> Candidate {
            let mut previous = 0;
            let nodes = ends
                .iter()
                .map(|end| {
                    let tail_pages = (*end / 16 - previous) as usize;
                    previous = *end / 16;
                    ChainNode {
                        key: self.history.key_at(*end).unwrap(),
                        tail_pages,
                    }
                })
                .collect();
            Candidate {
                level,
                source: SourceId(source),
                key: self.history.key_at(*ends.last().unwrap()).unwrap(),
                begin_tokens: 0,
                skip_pages: 0,
                nodes,
            }
        }
        fn request(&self, resident: u64) -> Request<'_> {
            Request {
                history: &self.history,
                identity: &self.identity,
                page_tokens: 16,
                resident,
                before: 2001,
            }
        }
    }
    #[test]
    fn longest_match_wins_then_closest_level_and_stable_source() {
        let f = Fixture::new();
        let candidates = [
            f.candidate(Level::HostDram, 1, &[800]),
            f.candidate(Level::RemoteNvme, 2, &[1600]),
        ];
        assert_eq!(
            plan(&f.request(0), &candidates).unwrap().part.source,
            SourceId(2)
        );
        let candidates = [
            f.candidate(Level::HostNvme, 3, &[1600]),
            f.candidate(Level::HostDram, 2, &[1600]),
            f.candidate(Level::HostDram, 1, &[1600]),
        ];
        assert_eq!(
            plan(&f.request(0), &candidates).unwrap().part.source,
            SourceId(1)
        );
    }
    #[test]
    fn a_resident_prefix_can_start_inside_a_record() {
        let f = Fixture::new();
        let c = f.candidate(Level::HostDram, 1, &[640, 1600]);
        let p = plan(&f.request(800), &[c]).unwrap();
        p.validate(&f.history, 16).unwrap();
        assert_eq!(
            (p.part.first_page, p.pages(), p.part.dest_page),
            (10, 50, 0)
        );
        assert_eq!(p.part.chain, [f.history.key_at(1600).unwrap()]);
        assert!(p.part.row);
    }
    #[test]
    fn row_only_restore_keeps_the_endpoint_and_no_pages() {
        let f = Fixture::new();
        let c = f.candidate(Level::HostDram, 1, &[808, 815]);
        let p = plan(&f.request(808), &[c]).unwrap();
        p.validate(&f.history, 16).unwrap();
        assert_eq!(p.pages(), 0);
        assert!(p.part.row);
    }
    #[test]
    fn holes_foreign_keys_and_inconsistent_tails_are_misses() {
        let f = Fixture::new();
        let good = f.candidate(Level::HostDram, 1, &[800]);
        let mut hole = good.clone();
        hole.begin_tokens = 16;
        hole.skip_pages = 1;
        let mut foreign = good.clone();
        foreign.key.digest[0] ^= 1;
        let mut gap = f.candidate(Level::HostDram, 2, &[400, 800]);
        gap.nodes[1].tail_pages += 1;
        let mut no_nodes = good.clone();
        no_nodes.nodes.clear();
        for bad in [hole, foreign, gap, no_nodes] {
            assert!(plan(&f.request(0), &[bad]).is_none());
        }
        assert!(plan(&f.request(0), &[]).is_none());
    }
    #[test]
    fn endpoints_stay_below_the_bound_and_above_the_resident_prefix() {
        let f = Fixture::new();
        let c = f.candidate(Level::HostDram, 1, &[800]);
        let mut request = f.request(0);
        request.before = 800;
        assert!(plan(&request, std::slice::from_ref(&c)).is_none());
        assert!(plan(&f.request(800), &[c]).is_none());
    }
    #[test]
    fn validate_rejects_bad_destination_row_and_page_count() {
        let f = Fixture::new();
        let c = f.candidate(Level::HostDram, 1, &[800]);
        let p = plan(&f.request(0), &[c]).unwrap();
        let mut bad = p.clone();
        bad.part.dest_page = 1;
        assert!(bad.validate(&f.history, 16).is_err());
        let mut bad = p.clone();
        bad.part.row = false;
        assert!(bad.validate(&f.history, 16).is_err());
        let mut bad = p;
        bad.part.pages += 1;
        assert!(bad.validate(&f.history, 16).is_err());
    }
}
