//! The worker for host window fills, rank-record reads and hashing.
//! Started on first use; jobs retain what they read until completion.

use std::{
    sync::{Arc, mpsc},
    thread,
};

use tokio::sync::Notify;

use crate::Result;

use super::super::exporter::Checksums;

/// Whose window a fill is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FillOwner {
    Export(u64),
    Put(u64),
}

/// Fills a window with the bytes the owner serves and returns their checksums.
pub(crate) type Work = Box<dyn FnOnce() -> Result<Checksums> + Send>;

pub(crate) struct FillJob<O = FillOwner> {
    pub owner: O,
    pub window: u32,
    pub work: Work,
}

pub(crate) struct FillDone<O = FillOwner> {
    pub owner: O,
    pub window: u32,
    pub result: Result<Checksums>,
}

pub(crate) struct HostFiller<O = FillOwner> {
    jobs: Option<mpsc::Sender<FillJob<O>>>,
    done: mpsc::Receiver<FillDone<O>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl<O: Send + 'static> HostFiller<O> {
    pub(crate) fn start(wake: Arc<Notify>) -> std::io::Result<Self> {
        let (jobs, rx) = mpsc::channel::<FillJob<O>>();
        let (tx, done) = mpsc::channel();
        let thread = thread::Builder::new()
            .name("selene-host-fill".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    let result = (job.work)();
                    let _ = tx.send(FillDone {
                        owner: job.owner,
                        window: job.window,
                        result,
                    });
                    wake.notify_one();
                }
            })?;
        Ok(Self {
            jobs: Some(jobs),
            done,
            thread: Some(thread),
        })
    }

    pub(crate) fn submit(&self, job: FillJob<O>) -> bool {
        self.jobs
            .as_ref()
            .is_some_and(|jobs| jobs.send(job).is_ok())
    }

    /// Wait for reads of slabs that their owner is about to free.
    pub(crate) fn join(&mut self) {
        self.jobs.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }

    pub(crate) fn collect(&self, out: &mut Vec<FillDone<O>>) {
        while let Ok(done) = self.done.try_recv() {
            out.push(done);
        }
    }
}
