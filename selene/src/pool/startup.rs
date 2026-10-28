//! The registration window: the one time a pool client may register host memory.
//!
//! Registering host memory with the GPU consumes device memory, which a serving
//! instance must not do. So a client attaches to the manager's memory once, at
//! startup, and the window closes: either the first attach finished (and
//! [`super::engine::Engine::close_registration`] then makes the registrar refuse
//! everything), or the manager was not reached in time and the client runs on its
//! local tiers for the life of the process.

use std::{
    sync::{Condvar, Mutex},
    time::{Duration, Instant},
};

/// A range this client registered at startup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegisteredRegion {
    /// What the range is: `"arena"` and `"staging"` are the manager's shared files, mapped from
    /// their paths; `"data-staging"` is this process's own staging for a manager on another host.
    pub name: &'static str,
    /// The start of the range in this process.
    pub address: usize,
    /// Its length in bytes.
    pub bytes: usize,
}

/// A range the client is about to register, offered to the approval callback before any is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedRegion {
    /// As [`RegisteredRegion::name`].
    pub name: &'static str,
    /// Its length in bytes.
    pub bytes: u64,
}

/// Judges the complete registration plan. `Err(reason)` registers nothing.
pub type Approve = std::sync::Arc<dyn Fn(&[PlannedRegion]) -> Result<(), String> + Send + Sync>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// The I/O thread has not reached the manager yet.
    Waiting,
    /// It has, and is registering.
    Registering,
    /// Attached (or nothing to attach): the window has closed behind the registrations.
    Done,
    /// Nothing was or will be registered.
    Closed,
}

pub(crate) struct Startup {
    state: Mutex<(Phase, Vec<RegisteredRegion>)>,
    changed: Condvar,
}

impl Startup {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new((Phase::Waiting, Vec::new())),
            changed: Condvar::new(),
        }
    }

    /// The I/O thread asks to register. `false` once the window is closed.
    pub(crate) fn begin(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        if state.0 == Phase::Waiting {
            state.0 = Phase::Registering;
            true
        } else {
            false
        }
    }

    /// The first attach is over, with `regions` registered.
    pub(crate) fn done(&self, regions: Vec<RegisteredRegion>) {
        let mut state = self.state.lock().unwrap();
        if matches!(state.0, Phase::Waiting | Phase::Registering) {
            *state = (Phase::Done, regions);
        }
        self.changed.notify_all();
    }

    /// The pool is off before it attached: nothing will be registered.
    pub(crate) fn close(&self) {
        let mut state = self.state.lock().unwrap();
        if state.0 == Phase::Waiting {
            state.0 = Phase::Closed;
        }
        self.changed.notify_all();
    }

    /// Whether the client gave up or was disabled before attaching.
    pub(crate) fn is_closed(&self) -> bool {
        self.state.lock().unwrap().0 == Phase::Closed
    }

    /// Waits up to `limit` for the first attach. `Ok` with what it registered, or `Err(true)`
    /// when `limit` passed (the window is now closed) and `Err(false)` when the pool was
    /// disabled first.
    pub(crate) fn wait(&self, limit: Duration) -> Result<Vec<RegisteredRegion>, bool> {
        let deadline = Instant::now() + limit;
        let mut state = self.state.lock().unwrap();
        while state.0 == Phase::Waiting {
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                state.0 = Phase::Closed;
                return Err(true);
            };
            state = self.changed.wait_timeout(state, left).unwrap().0;
        }
        // Registering is bounded by the devices, not by the manager; it gets as long again.
        let deadline = Instant::now() + limit.max(Duration::from_secs(10));
        while state.0 == Phase::Registering {
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                return Err(false);
            };
            state = self.changed.wait_timeout(state, left).unwrap().0;
        }
        match state.0 {
            Phase::Done => Ok(state.1.clone()),
            _ => Err(false),
        }
    }
}

/// Registers nothing: the window is closed.
pub(crate) struct ClosedRegistrar;

impl crate::HostRegistrar for ClosedRegistrar {
    fn register(
        &self,
        _context: usize,
        _address: usize,
        _bytes: usize,
    ) -> crate::Result<Box<dyn crate::Registration>> {
        Err(crate::Error::new(
            "host memory can be registered only at startup, and the pool's registration is closed",
        ))
    }

    fn map_split(
        &self,
        _requests: &[(usize, i32, usize)],
    ) -> crate::Result<Box<dyn crate::SplitMapping>> {
        Err(crate::Error::new(
            "host memory can be mapped only at startup, and the pool's registration is closed",
        ))
    }
}
