//! Coordinated CUDA ranks sharing one process address space.

use super::{ExecutionManifest, FrameCompletion, GraphExecutor, OwnedUpload, Upload};
use crate::{Error, ExecutionError, Result, Shutdown, cache::Fence};
use std::{
    sync::{Arc, mpsc},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const TIMEOUT: Duration = Duration::from_secs(120);
const HANDOFF_POLL: Duration = Duration::from_micros(500);

// Prepare/commit and completion/prepare exchanges are brief. Poll that interval
// before parking a rank, while retaining blocking waits for idle serving.
fn receive<T>(receiver: &mpsc::Receiver<T>, timeout: Option<Duration>) -> std::result::Result<T, mpsc::RecvTimeoutError> {
    let started = Instant::now();
    let poll = timeout.map_or(HANDOFF_POLL, |limit| limit.min(HANDOFF_POLL));
    loop {
        match receiver.try_recv() {
            Ok(value) => return Ok(value),
            Err(mpsc::TryRecvError::Disconnected) => return Err(mpsc::RecvTimeoutError::Disconnected),
            Err(mpsc::TryRecvError::Empty) => {}
        }
        if started.elapsed() >= poll { break; }
        std::hint::spin_loop();
    }
    match timeout {
        Some(limit) => receiver.recv_timeout(limit.saturating_sub(started.elapsed())),
        None => receiver.recv().map_err(|_| mpsc::RecvTimeoutError::Disconnected),
    }
}
fn invalid(message: impl Into<String>) -> Error {
    Error::execution(ExecutionError::Worker, message)
}

struct Step {
    fence: Fence,
    graph: usize,
    grammar: bool,
    uploads: Vec<OwnedUpload>,
}
impl Step {
    fn uploads(&self) -> Vec<Upload<'_>> {
        self.uploads
            .iter()
            .map(|u| Upload {
                buffer: &u.buffer,
                offset: u.offset,
                bytes: &u.bytes,
            })
            .collect()
    }
}
enum Command {
    Prepare(Arc<Step>),
    Commit,
    Drain,
}
enum Reply {
    Ready,
    Complete(FrameCompletion),
    Drained,
}
struct Peer {
    sender: mpsc::Sender<Command>,
    receiver: mpsc::Receiver<Result<Reply>>,
    thread: Option<JoinHandle<Shutdown>>,
    completion: Option<FrameCompletion>,
}
impl Peer {
    fn send(&self, command: Command) -> Result<()> {
        self.sender
            .send(command)
            .map_err(|_| invalid("rank thread disconnected"))
    }
    fn receive(&self) -> Result<Reply> {
        receive(&self.receiver, Some(TIMEOUT))
            .map_err(|_| invalid("rank thread response deadline expired"))?
    }
}

pub(crate) struct ThreadedExecutor {
    local: GraphExecutor,
    peers: Vec<Peer>,
    pending: Option<Instant>,
    completion: Option<FrameCompletion>,
    poison: Option<Error>,
    shutdown: Option<Shutdown>,
}
impl ThreadedExecutor {
    pub unsafe fn new(manifest: ExecutionManifest, ranks: Vec<ExecutionManifest>) -> Result<Self> {
        if ranks.len() >= 16 {
            return Err(invalid("tensor parallelism supports at most sixteen ranks"));
        }
        if ranks.is_empty() {
            return Err(invalid("tensor parallelism requires at least two ranks"));
        }
        for rank in &ranks {
            rank.validate()?;
            if !same_layout(&manifest, rank) {
                return Err(invalid("rank graph I/O schemas differ"));
            }
        }
        let local = unsafe { GraphExecutor::new(manifest)? };
        let mut result = Self {
            local,
            peers: Vec::new(),
            pending: None,
            completion: None,
            poison: None,
            shutdown: None,
        };
        for (index, manifest) in ranks.into_iter().enumerate() {
            let (sender, commands) = mpsc::channel();
            let (replies, receiver) = mpsc::channel();
            let thread = thread::Builder::new()
                .name(format!("flare-rank-{}", index + 1))
                .spawn(move || {
                    let mut executor = match unsafe { GraphExecutor::new(manifest) } {
                        Ok(executor) => executor,
                        Err(error) => {
                            let _ = replies.send(Err(error.clone()));
                            return Shutdown {
                                drained: true,
                                error: Some(error),
                            };
                        }
                    };
                    executor.copy_logprobs = false;
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        rank_loop(&mut executor, commands, &replies)
                    }))
                    .unwrap_or_else(|_| Err(invalid("rank execution thread panicked")));
                    match result {
                        Ok(()) => executor.close(),
                        Err(error) => {
                            let _ = replies.send(Err(error.clone()));
                            executor.abandon(error)
                        }
                    }
                })
                .map_err(|e| invalid(format!("cannot start rank thread: {e}")))?;
            let mut peer = Peer {
                sender,
                receiver,
                thread: Some(thread),
                completion: None,
            };
            let ready = match peer.receiver.recv_timeout(TIMEOUT) {
                Ok(reply) => reply,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    Err(invalid("rank initialization thread exited"))
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // A thread inside CUDA may still use resources supplied by the caller.
                    eprintln!("rank initialization deadline expired; terminating the CUDA process");
                    std::process::abort();
                }
            };
            if !matches!(ready, Ok(Reply::Ready)) {
                drop(peer.sender);
                let _ = peer.thread.take().unwrap().join();
                return Err(ready
                    .err()
                    .unwrap_or_else(|| invalid("rank startup did not report readiness")));
            }
            result.peers.push(peer);
        }
        Ok(result)
    }
    pub fn manifest(&self) -> &ExecutionManifest {
        self.local.manifest()
    }
    pub fn outstanding(&self) -> usize {
        usize::from(self.pending.is_some())
    }
    pub fn can_submit(&self, grammar: bool) -> bool {
        self.poison.is_none()
            && self.shutdown.is_none()
            && self.pending.is_none()
            && self.local.can_submit(grammar)
    }
    pub fn next_fence(&self) -> Result<Fence> {
        if let Some(error) = &self.poison {
            return Err(error.clone());
        }
        if self.shutdown.is_some() {
            return Err(invalid("rank executor is closed"));
        }
        self.local.next_fence()
    }
    pub fn submit(
        &mut self,
        graph: usize,
        uploads: &[Upload<'_>],
        grammar: bool,
    ) -> Result<Option<Fence>> {
        self.next_fence()?;
        if self.pending.is_some() {
            return Ok(None);
        }
        let Some(fence) = self.local.validate_submission(graph, uploads, grammar)? else {
            return Ok(None);
        };
        let result = (|| {
            let dispatched = Instant::now();
            let step = Arc::new(Step {
                fence,
                graph,
                grammar,
                uploads: uploads
                    .iter()
                    .map(|u| OwnedUpload {
                        buffer: u.buffer.to_owned(),
                        offset: u.offset,
                        bytes: u.bytes.to_vec(),
                    })
                    .collect(),
            });
            for peer in &self.peers {
                peer.send(Command::Prepare(step.clone()))?;
            }
            for peer in &self.peers {
                if !matches!(peer.receive()?, Reply::Ready) {
                    return Err(invalid("rank did not validate the step"));
                }
            }
            self.pending = Some(dispatched);
            for peer in &self.peers {
                peer.send(Command::Commit)?;
            }
            if self.local.submit(graph, uploads, grammar)? != Some(fence) {
                return Err(invalid("rank-zero fence changed after validation"));
            }
            Ok(Some(fence))
        })();
        if let Err(error) = &result {
            self.poison = Some(error.clone());
        }
        result
    }
    pub fn poll(&mut self) -> Result<Option<FrameCompletion>> {
        if let Some(error) = &self.poison {
            return Err(error.clone());
        }
        match self.poll_ranks() {
            Ok(None) => Ok(None),
            Ok(Some(completion)) => Ok(Some(completion)),
            Err(error) => {
                self.poison = Some(error.clone());
                Err(error)
            }
        }
    }
    fn poll_ranks(&mut self) -> Result<Option<FrameCompletion>> {
        let Some(started) = self.pending else {
            return Ok(None);
        };
        if started.elapsed() > TIMEOUT {
            return Err(invalid("tensor-parallel graph completion deadline expired"));
        }
        if self.completion.is_none() {
            // Inspect both tags before moving the large inline completion.
            // Empty polls must not copy its inactive record storage.
            match self.local.poll() {
                Ok(None) => {}
                Ok(Some(completion)) => self.completion = Some(completion),
                Err(error) => return Err(error),
            }
        }
        for peer in &mut self.peers {
            if peer.completion.is_some() {
                continue;
            }
            match peer.receiver.try_recv() {
                Ok(Ok(Reply::Complete(completion))) => peer.completion = Some(completion),
                Ok(Err(error)) => return Err(error),
                Err(mpsc::TryRecvError::Empty) => {}
                _ => return Err(invalid("rank thread failed to return graph completion")),
            }
        }
        if self.completion.is_none() || self.peers.iter().any(|p| p.completion.is_none()) {
            return Ok(None);
        }
        for (index, peer) in self.peers.iter_mut().enumerate() {
            let agreement = if self.local.manifest().graphs[self.completion.as_ref().unwrap().graph_index].kind
                == super::GraphKind::PackedSpeculative { super::parallel::agree_packed } else { super::parallel::agree };
            agreement(
                self.completion.as_ref().unwrap(),
                peer.completion.as_ref().unwrap(),
            )
            .map_err(|error| invalid(format!("rank {}: {error}", index + 1)))?;
            self.completion.as_mut().unwrap().timing.merge_rank(index + 1,
                peer.completion.as_ref().unwrap().timing)?;
            peer.completion = None;
        }
        self.completion.as_mut().unwrap().timing.wall_ns =
            started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        self.pending = None;
        Ok(self.completion.take())
    }
    pub fn close(&mut self) -> Shutdown {
        if let Some(shutdown) = &self.shutdown {
            return shutdown.clone();
        }
        let result = (|| {
            while self.pending.is_some() {
                self.poll()?;
                thread::sleep(Duration::from_micros(50));
            }
            if let Some(error) = &self.poison {
                return Err(error.clone());
            }
            for peer in &self.peers {
                peer.send(Command::Drain)?;
            }
            for peer in &mut self.peers {
                if !matches!(peer.receive()?, Reply::Drained) {
                    return Err(invalid("rank thread did not drain"));
                }
                let shutdown = peer
                    .thread
                    .take()
                    .unwrap()
                    .join()
                    .map_err(|_| invalid("rank thread panicked"))?;
                if !shutdown.drained {
                    return Err(invalid("rank thread did not confirm drain"));
                }
                if let Some(error) = shutdown.error {
                    return Err(error);
                }
            }
            Ok(())
        })();
        let shutdown = match result {
            Ok(()) => self.local.close(),
            Err(error) => self.local.abandon(error),
        };
        self.shutdown = Some(shutdown.clone());
        shutdown
    }
}
impl Drop for ThreadedExecutor {
    fn drop(&mut self) {
        self.close();
    }
}

fn rank_loop(
    executor: &mut GraphExecutor,
    commands: mpsc::Receiver<Command>,
    replies: &mpsc::Sender<Result<Reply>>,
) -> Result<()> {
    let send = |reply| {
        replies
            .send(Ok(reply))
            .map_err(|_| invalid("rank coordinator disconnected"))
    };
    send(Reply::Ready)?;
    while let Ok(command) = receive(&commands, None) {
        match command {
            Command::Prepare(step) => {
                let uploads = step.uploads();
                if executor.validate_submission(step.graph, &uploads, step.grammar)?
                    != Some(step.fence)
                {
                    return Err(invalid("rank submission fence differs"));
                }
                send(Reply::Ready)?;
                if !matches!(receive(&commands, Some(TIMEOUT)), Ok(Command::Commit)) {
                    return Err(invalid("rank commit missing after preparation"));
                }
                if executor.submit(step.graph, &uploads, step.grammar)? != Some(step.fence) {
                    return Err(invalid("rank fence changed after validation"));
                }
                let started = Instant::now();
                loop {
                    match executor.poll() {
                        Ok(None) => {}
                        Ok(Some(completion)) => {
                            send(Reply::Complete(completion))?;
                            break;
                        }
                        Err(error) => return Err(error),
                    }
                    if started.elapsed() > TIMEOUT {
                        return Err(invalid("rank graph completion deadline expired"));
                    }
                    thread::yield_now();
                }
            }
            Command::Drain => {
                let shutdown = executor.close();
                if !shutdown.drained {
                    return Err(invalid("rank streams did not drain"));
                }
                if let Some(error) = shutdown.error {
                    return Err(error);
                }
                send(Reply::Drained)?;
                return Ok(());
            }
            Command::Commit => return Err(invalid("rank commit arrived without preparation")),
        }
    }
    Ok(())
}

fn same_layout(a: &ExecutionManifest, b: &ExecutionManifest) -> bool {
    match (&a.engram, &b.engram) {
        // Partitioned tables have independent delivery groups, each numbered from zero.
        (Some(a), Some(b)) if Arc::ptr_eq(&a.shared, &b.shared) && a.rank == b.rank => {
            return false;
        }
        (Some(_), None) | (None, Some(_)) => return false,
        _ => {}
    }
    match (&a.offload, &b.offload) {
        (Some(a), Some(b)) if !a.shares(b) || a.rank() == b.rank() => return false,
        (Some(_), None) | (None, Some(_)) => return false,
        _ => {}
    }
    match (&a.cache, &b.cache) {
        (Some(a), Some(b)) if !a.same_layout(b) => return false,
        (Some(_), None) | (None, Some(_)) => return false,
        _ => {}
    }
    a.abi_version == b.abi_version
        && a.max_active == b.max_active
        && a.input_frame_bytes == b.input_frame_bytes
        && a.buffers.len() == b.buffers.len()
        && a.graphs.len() == b.graphs.len()
        && a.buffers.iter().zip(&b.buffers).all(|(a, b)| {
            a.name == b.name
                && a.dtype == b.dtype
                && a.nbytes == b.nbytes
                && a.shape == b.shape
                && a.strides == b.strides
        })
        && a.graphs.iter().zip(&b.graphs).all(|(a, b)| {
            a.name == b.name
                && a.kind == b.kind
                && a.sampling_mode == b.sampling_mode
                && a.batch_capacity == b.batch_capacity
                && a.token_capacity == b.token_capacity
                && a.draft_tokens == b.draft_tokens
                && a.packed_token_capacity == b.packed_token_capacity
                && a.prefill_capacity == b.prefill_capacity
                && a.input_bindings.len() == b.input_bindings.len()
                && a.input_bindings
                    .iter()
                    .zip(&b.input_bindings)
                    .all(|(a, b)| {
                        a.buffer_name == b.buffer_name
                            && a.host_offset == b.host_offset
                            && a.nbytes == b.nbytes
                    })
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handoff_receive_preserves_queued_messages_and_disconnects() {
        let (sender, receiver) = mpsc::channel();
        sender.send(7).unwrap();
        sender.send(9).unwrap();
        drop(sender);
        assert_eq!(receive(&receiver, Some(Duration::ZERO)), Ok(7));
        assert_eq!(receive(&receiver, None), Ok(9));
        assert_eq!(receive(&receiver, None), Err(mpsc::RecvTimeoutError::Disconnected));
    }

    #[test]
    fn handoff_receive_parks_for_late_messages_and_respects_deadlines() {
        let (sender, receiver) = mpsc::channel();
        assert_eq!(receive(&receiver, Some(Duration::ZERO)), Err(mpsc::RecvTimeoutError::Timeout));
        let worker = thread::spawn(move || {
            thread::sleep(HANDOFF_POLL * 4);
            sender.send(17).unwrap();
        });
        assert_eq!(receive(&receiver, Some(Duration::from_secs(1))), Ok(17));
        worker.join().unwrap();
    }
    #[test]
    fn rank_layout_compares_bindings_but_not_cuda_addresses() {
        let root = super::super::tests::manifest();
        let mut peer = root.clone();
        peer.context += 1;
        peer.device += 1;
        for buffer in &mut peer.buffers {
            buffer.address += 0x10000;
        }
        for graph in &mut peer.graphs {
            graph.handle += 0x10000;
        }
        assert!(same_layout(&root, &peer));
        peer.graphs[0].input_bindings[0].host_offset += 8;
        assert!(!same_layout(&root, &peer));
        peer = root.clone();
        peer.buffers.pop();
        assert!(!same_layout(&root, &peer));
        peer = root.clone();
        peer.graphs[0].token_capacity += 1;
        assert!(!same_layout(&root, &peer));
        peer = root.clone();
        peer.graphs[0].prefill_capacity = 64;
        assert!(!same_layout(&root, &peer));
        let geometry = crate::offload::geometry::tests::geometry(64, &[("k", 8192)], 4096);
        peer = root.clone();
        peer.cache = Some(geometry.clone());
        assert!(!same_layout(&root, &peer));
        let mut root = root;
        root.cache = Some(geometry.clone());
        for plane in &mut peer.cache.as_mut().unwrap().planes {
            plane.address += 0x1000;
        }
        assert!(same_layout(&root, &peer));
        peer.cache.as_mut().unwrap().planes[0].unit_bytes += 1;
        assert!(!same_layout(&root, &peer));
    }

    #[test]
    fn rank_layout_accepts_shared_and_partitioned_engram_delivery() {
        use crate::engram::{Binding, Index, Plane, Shared, Table};

        for partitioned in [false, true] {
            let groups: Vec<_> = (0..if partitioned { 4 } else { 1 })
                .map(|rank| {
                    Shared::layout_fixture(
                        Index {
                            tables: vec![Table {
                                planes: [256, 8].map(|row_bytes| Plane {
                                    file: rank,
                                    offset: rank as u64 * 4096,
                                    nbytes: (100 + rank) * row_bytes,
                                    row_bytes,
                                }),
                                heads: 24,
                                weights: "weights".into(),
                                scales: "scales".into(),
                                hashes: "hashes".into(),
                                event: "ready".into(),
                                request: Some("requests".into()),
                            }],
                        },
                        if partitioned { 1 } else { 4 },
                    )
                })
                .collect();
            let ranks: Vec<_> = (0..4)
                .map(|rank| {
                    let mut manifest = super::super::tests::manifest();
                    manifest.context += rank;
                    manifest.device = rank as i32;
                    for buffer in &mut manifest.buffers {
                        buffer.address += rank * 0x100000;
                    }
                    for graph in &mut manifest.graphs {
                        graph.handle += rank * 0x100000;
                    }
                    manifest.engram = Some(Binding {
                        shared: groups[if partitioned { rank } else { 0 }].clone(),
                        rank: if partitioned { 0 } else { rank },
                        events: vec![rank + 1],
                    });
                    manifest.validate().unwrap();
                    manifest
                })
                .collect();
            for peer in &ranks[1..] {
                assert!(same_layout(&ranks[0], peer));
            }
            // A delivery group cannot count the same member twice.
            let mut duplicate = ranks[1].clone();
            duplicate.engram = ranks[0].engram.clone();
            assert!(!same_layout(&ranks[0], &duplicate));
            let mut missing = ranks[1].clone();
            missing.engram = None;
            assert!(!same_layout(&ranks[0], &missing));
            let mut mismatch = ranks[1].clone();
            mismatch.graphs[0].input_bindings[0].host_offset += 8;
            assert!(!same_layout(&ranks[0], &mismatch));
        }
    }

    #[test]
    fn tp4_catalog_accepts_shared_and_partitioned_embedding_delivery() {
        use super::super::{BufferDescriptor, Dtype, GraphKind, InputBinding};
        use crate::engram::{Binding, Index, Plane, Shared, Table};

        for logprobs in [false, true] {
            let mut root = if logprobs {
                super::super::tests::with_logprobs(20)
            } else {
                super::super::tests::manifest()
            };
            root.abi_version = 3;
            let records = root.buffers.iter_mut().find(|b| b.name == "completion_records").unwrap();
            records.nbytes = 2 * 4 * 24 * 8;
            records.shape = vec![2, 4, 24];
            records.strides = vec![4 * 24, 24, 1];
            root.buffers.push(BufferDescriptor {
                name: "proposal_confidence".into(), address: 0x30000, nbytes: 4 * 5 * 4,
                dtype: Dtype::Float32, shape: vec![4, 5], strides: vec![5, 1],
            });
            root.buffers.push(BufferDescriptor {
                name: "prefill_row".into(), address: 0x40000, nbytes: 4,
                dtype: Dtype::Int32, shape: vec![1], strides: vec![1],
            });
            let template = root.graphs[0].clone();
            root.graphs = [
                ("prefill", GraphKind::Prefill, 1, 4096, 0, 0),
                ("decode", GraphKind::Decode, 4, 1, 0, 0),
                ("resume", GraphKind::Resume, 4, 1, 0, 0),
                ("speculative", GraphKind::Speculative, 4, 6, 0, 0),
                ("proposal", GraphKind::Proposal, 4, 6, 0, 0),
                ("verification", GraphKind::Verification, 4, 6, 0, 0),
                ("packed", GraphKind::Verification, 4, 6, 0, 12),
                ("mixed", GraphKind::Mixed, 4, 6, 4096, 0),
            ].into_iter().enumerate().map(|(index, (name, kind, batch, width, prefill, packed))| {
                let mut graph = template.clone();
                graph.name = name.into();
                graph.kind = kind;
                graph.handle += index;
                graph.batch_capacity = batch;
                graph.token_capacity = width;
                graph.prefill_capacity = prefill;
                graph.packed_token_capacity = packed;
                if kind == GraphKind::Mixed {
                    graph.input_bindings.push(InputBinding {
                        buffer_name: "prefill_row".into(), host_offset: root.input_frame_bytes, nbytes: 4,
                    });
                }
                graph
            }).collect();
            root.input_frame_bytes += 8;
            root.validate().unwrap();

            for partitioned in [false, true] {
                let groups: Vec<_> = (0..if partitioned { 4 } else { 1 }).map(|rank| {
                    Shared::layout_fixture(Index { tables: vec![Table {
                        planes: [256, 8].map(|row_bytes| Plane {
                            file: rank, offset: rank as u64 * 4096,
                            nbytes: (100 + rank) * row_bytes, row_bytes,
                        }),
                        heads: 24, weights: "weights".into(), scales: "scales".into(),
                        hashes: "hashes".into(), event: "ready".into(), request: Some("requests".into()),
                    }] }, if partitioned { 1 } else { 4 })
                }).collect();
                let ranks: Vec<_> = (0..4).map(|rank| {
                    let mut manifest = root.clone();
                    manifest.context += rank;
                    manifest.device = rank as i32;
                    for buffer in &mut manifest.buffers { buffer.address += rank * 0x100000; }
                    for graph in &mut manifest.graphs { graph.handle += rank * 0x100000; }
                    manifest.engram = Some(Binding {
                        shared: groups[if partitioned { rank } else { 0 }].clone(),
                        rank: if partitioned { 0 } else { rank }, events: vec![rank + 1],
                    });
                    manifest.validate().unwrap();
                    manifest
                }).collect();
                for peer in &ranks[1..] { assert!(same_layout(&ranks[0], peer)); }
                // A repeated member of one delivery group would corrupt its completion accounting.
                assert!(!same_layout(&ranks[0], &ranks[0]));
                let mut bad = ranks[1].clone();
                bad.graphs[6].input_bindings[0].host_offset += 8;
                assert!(!same_layout(&ranks[0], &bad));
                let mut bad = ranks[1].clone();
                bad.buffers[2].shape[1] += 1;
                assert!(!same_layout(&ranks[0], &bad));
            }
        }
    }
}
