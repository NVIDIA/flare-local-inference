//! The ticket a queued request holds for its asynchronous pool lookup.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

/// UDS lookup p99 was 0.55 ms; allow host scheduling headroom.
pub const LOOKUP_GRACE: Duration = Duration::from_millis(2);

use crate::plan;

use super::types::PoolCandidate;

/// What a lookup produced for one request.
#[derive(Debug)]
pub struct Answer {
    pub candidates: Vec<PoolCandidate>,
}

impl Answer {
    /// The candidates in the planner's terms. The planner checks each against
    /// the request's history and drops one that disagrees.
    pub fn plan_candidates(&self) -> Vec<plan::Candidate> {
        self.candidates
            .iter()
            .map(|candidate| candidate.plan.clone())
            .collect()
    }
}

#[derive(Clone, Debug)]
pub enum LookupState {
    /// No reply yet.
    Pending,
    /// The lookup finished with an answer, possibly with no candidates.
    Ready(Arc<Answer>),
    /// There is no answer and none will come: the pool is not ready, the lookup
    /// was refused or timed out, or the session was lost.
    None,
}

pub(crate) struct TicketInner {
    state: Mutex<LookupState>,
}

impl TicketInner {
    pub(crate) fn new(state: LookupState) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(state),
        })
    }

    pub(crate) fn set(&self, state: LookupState) {
        *self.state.lock().unwrap() = state;
    }

    pub(crate) fn get(&self) -> LookupState {
        self.state.lock().unwrap().clone()
    }
}

/// The handle one request holds for its lookup. Dropping it withdraws the
/// lookup still on the wire; a late reply is discarded.
pub struct LookupTicket {
    inner: Arc<TicketInner>,
    withdraw: Option<Box<dyn FnOnce() + Send>>,
    grace: Duration,
}

impl LookupTicket {
    pub(crate) fn new(
        inner: Arc<TicketInner>,
        withdraw: Box<dyn FnOnce() + Send>,
        grace: Duration,
    ) -> Self {
        Self {
            inner,
            withdraw: Some(withdraw),
            grace,
        }
    }

    /// A ticket that will never have an answer and owns nothing.
    pub fn none() -> Self {
        Self {
            inner: TicketInner::new(LookupState::None),
            withdraw: None,
            grace: LOOKUP_GRACE,
        }
    }

    /// Maximum admission hold for this lookup.
    pub fn grace(&self) -> Duration {
        self.grace
    }

    /// The current state. Never waits for a reply.
    pub fn state(&self) -> LookupState {
        self.inner.get()
    }
}

#[cfg(any(test, feature = "test-support"))]
impl LookupTicket {
    /// Complete a watched lookup without a manager.
    pub fn set_state(&self, state: LookupState) {
        self.inner.set(state);
    }

    /// A ticket in `state` that runs `withdraw` when it is dropped, for tests of
    /// the code that holds tickets.
    pub fn watched(state: LookupState, withdraw: impl FnOnce() + Send + 'static) -> Self {
        Self {
            inner: TicketInner::new(state),
            withdraw: Some(Box::new(withdraw)),
            grace: LOOKUP_GRACE,
        }
    }
}

impl Drop for LookupTicket {
    fn drop(&mut self) {
        if let Some(withdraw) = self.withdraw.take() {
            withdraw();
        }
    }
}

impl std::fmt::Debug for LookupTicket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LookupTicket")
            .field("state", &self.state())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ticket_without_a_lookup_has_no_answer() {
        let ticket = LookupTicket::none();
        assert!(matches!(ticket.state(), LookupState::None));
    }
}
