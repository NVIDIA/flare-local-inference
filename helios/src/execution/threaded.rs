//! Coordinated CUDA ranks sharing one process address space.

use super::{ExecutionManifest, FrameCompletion, GraphExecutor, OwnedUpload, Upload};
use crate::{Error, ExecutionError, Result, Shutdown, cache::Fence};
use std::{
    collections::VecDeque,
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
    Commit(Fence, usize),
    Drain,
}
enum Reply {
    Initialized,
    Ready(Fence, usize),
    Complete(FrameCompletion),
    Drained,
}
struct Peer {
    sender: mpsc::Sender<Command>,
    receiver: mpsc::Receiver<Result<Reply>>,
    thread: Option<JoinHandle<Shutdown>>,
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

struct Pending {
    step: Arc<Step>,
    dispatched: Instant,
    committed: bool,
    queued: bool,
    ready: Vec<bool>,
    completions: Vec<Option<FrameCompletion>>,
}

pub(crate) struct ThreadedExecutor {
    local: GraphExecutor,
    peers: Vec<Peer>,
    pending: VecDeque<Pending>,
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
            pending: VecDeque::with_capacity(2),
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
            if !matches!(ready, Ok(Reply::Initialized)) {
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
        self.pending.len()
    }
    pub fn can_submit(&self, grammar: bool) -> bool {
        self.poison.is_none()
            && self.shutdown.is_none()
            && self.pending.len() < 2
            && self.pending.back().is_none_or(|frame| frame.committed)
            && (!grammar || self.pending.is_empty())
            && !self.pending.iter().any(|frame| frame.step.grammar || !matches!(
                self.local.manifest().graphs[frame.step.graph].kind,
                super::GraphKind::Decode | super::GraphKind::Speculative))
            && self.local.can_submit(grammar)
    }
    pub fn next_fence(&self) -> Result<Fence> {
        if let Some(error) = &self.poison {
            return Err(error.clone());
        }
        if self.shutdown.is_some() {
            return Err(invalid("rank executor is closed"));
        }
        match self.pending.back() {
            Some(frame) if !frame.committed => Ok(Fence(frame.step.fence.0 + 1)),
            _ => self.local.next_fence(),
        }
    }
    pub fn submit(
        &mut self,
        graph: usize,
        uploads: &[Upload<'_>],
        grammar: bool,
    ) -> Result<Option<Fence>> {
        self.next_fence()?;
        let barrier = grammar || !matches!(self.local.manifest().graphs.get(graph).map(|g| g.kind),
            Some(super::GraphKind::Decode | super::GraphKind::Speculative));
        if !self.can_submit(barrier) { return Ok(None); }
        let Some(fence) = self.local.validate_submission(graph, uploads, grammar)? else {
            return Ok(None);
        };
        let step = Arc::new(Step {
            fence, graph, grammar,
            uploads: uploads.iter().map(|u| OwnedUpload {
                buffer: u.buffer.to_owned(), offset: u.offset, bytes: u.bytes.to_vec(),
            }).collect(),
        });
        let queued = !self.pending.is_empty();
        self.pending.push_back(Pending {
            step: step.clone(), dispatched: Instant::now(), committed: false, queued,
            ready: vec![false; self.peers.len()],
            completions: (0..=self.peers.len()).map(|_| None).collect(),
        });
        for peer in &self.peers {
            if let Err(error) = peer.send(Command::Prepare(step.clone())) {
                self.poison = Some(error.clone());
                return Err(error);
            }
        }
        Ok(Some(fence))
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
        if self.pending.iter().any(|frame| frame.dispatched.elapsed() > TIMEOUT) {
            return Err(invalid("tensor-parallel graph completion deadline expired"));
        }
        if let Some(completion) = self.local.poll()? {
            let frame = self.pending.iter_mut().find(|frame| frame.step.fence == completion.fence)
                .ok_or_else(|| invalid("unexpected rank-zero completion fence"))?;
            if frame.completions[0].replace(completion).is_some() {
                return Err(invalid("duplicate rank-zero completion"));
            }
        }
        for (index, peer) in self.peers.iter_mut().enumerate() {
            loop {
                let reply = match peer.receiver.try_recv() {
                    Ok(reply) => reply?,
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(_) => return Err(invalid("rank thread disconnected")),
                };
                match reply {
                    Reply::Ready(fence, graph) => {
                        let frame = self.pending.iter_mut().find(|f| f.step.fence == fence)
                            .ok_or_else(|| invalid("unexpected rank readiness fence"))?;
                        if frame.step.graph != graph || frame.ready[index] || frame.committed {
                            return Err(invalid("rank readiness identity or state differs"));
                        }
                        frame.ready[index] = true;
                    }
                    Reply::Complete(completion) => {
                        let expected = self.pending.iter_mut()
                            .find(|f| f.completions[index + 1].is_none())
                            .ok_or_else(|| invalid("unexpected rank completion"))?;
                        if !expected.committed || expected.step.fence != completion.fence
                            || expected.step.graph != completion.graph_index {
                            return Err(invalid("rank completion order or identity differs"));
                        }
                        expected.completions[index + 1] = Some(completion);
                    }
                    _ => return Err(invalid("unexpected rank thread reply")),
                }
            }
        }
        for frame in &mut self.pending {
            if frame.committed { continue; }
            if !frame.ready.iter().all(|ready| *ready) { break; }
            let uploads = frame.step.uploads();
            if self.local.validate_submission(frame.step.graph, &uploads, frame.step.grammar)?
                != Some(frame.step.fence) {
                return Err(invalid("rank-zero reservation changed before commit"));
            }
            for peer in &self.peers {
                peer.send(Command::Commit(frame.step.fence, frame.step.graph))?;
            }
            if self.local.submit(frame.step.graph, &uploads, frame.step.grammar)? != Some(frame.step.fence) {
                return Err(invalid("rank-zero fence changed after validation"));
            }
            frame.committed = true;
        }
        if !self.pending.front().is_some_and(|frame| frame.completions.iter().all(Option::is_some)) {
            return Ok(None);
        }
        let mut frame = self.pending.pop_front().unwrap();
        let mut root = frame.completions[0].take().unwrap();
        for (index, completion) in frame.completions.into_iter().skip(1).enumerate() {
            let completion = completion.unwrap();
            let agreement = if self.local.manifest().graphs[root.graph_index].kind == super::GraphKind::Speculative {
                super::parallel::agree_speculative
            } else { super::parallel::agree };
            agreement(&root, &completion).map_err(|error| invalid(format!("rank {}: {error}", index + 1)))?;
            root.timing.merge_rank(index + 1, completion.timing)?;
        }
        root.timing.queued = frame.queued;
        root.timing.wall_ns = frame.dispatched.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        Ok(Some(root))
    }
    pub fn close(&mut self) -> Shutdown {
        if let Some(shutdown) = &self.shutdown {
            return shutdown.clone();
        }
        let result = (|| {
            while !self.pending.is_empty() {
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
    send(Reply::Initialized)?;
    let mut prepared: Option<Arc<Step>> = None;
    let mut prepared_at: Option<Instant> = None;
    let mut started = VecDeque::new();
    loop {
        let command = if executor.outstanding() == 0 && prepared.is_none() {
            match receive(&commands, None) { Ok(command) => Some(command), Err(_) => return Ok(()) }
        } else {
            match commands.try_recv() {
                Ok(command) => Some(command),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(_) => return Err(invalid("rank coordinator disconnected during execution")),
            }
        };
        match command {
            Some(Command::Prepare(step)) => {
                if prepared.is_some() { return Err(invalid("rank already has a prepared frame")); }
                if executor.validate_submission(step.graph, &step.uploads(), step.grammar)? != Some(step.fence) {
                    return Err(invalid("rank submission fence differs"));
                }
                send(Reply::Ready(step.fence, step.graph))?;
                prepared = Some(step);
                prepared_at = Some(Instant::now());
            }
            Some(Command::Commit(fence, graph)) => {
                prepared_at = None;
                let step = prepared.take().ok_or_else(|| invalid("rank commit arrived without preparation"))?;
                if step.fence != fence || step.graph != graph { return Err(invalid("rank commit identity differs")); }
                if executor.submit(graph, &step.uploads(), step.grammar)? != Some(fence) {
                    return Err(invalid("rank fence changed after validation"));
                }
                started.push_back(Instant::now());
            }
            Some(Command::Drain) => {
                if prepared.is_some() || executor.outstanding() != 0 { return Err(invalid("rank drain arrived with outstanding frames")); }
                let shutdown = executor.close();
                if !shutdown.drained { return Err(invalid("rank streams did not drain")); }
                if let Some(error) = shutdown.error { return Err(error); }
                send(Reply::Drained)?;
                return Ok(());
            }
            None => {}
        }
        if let Some(completion) = executor.poll()? {
            started.pop_front();
            send(Reply::Complete(completion))?;
        }
        if prepared_at.is_some_and(|time| time.elapsed() > TIMEOUT) {
            return Err(invalid("rank commit deadline expired after preparation"));
        }
        if started.front().is_some_and(|time| time.elapsed() > TIMEOUT) {
            return Err(invalid("rank graph completion deadline expired"));
        }
        if executor.outstanding() != 0 || prepared.is_some() { thread::yield_now(); }
    }
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
                && a.confidence_width == b.confidence_width
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
        let geometry = flare_selene::testing::geometry(64, &[("k", 8192)], 4096);
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
            root.abi_version = super::super::ABI_VERSION;
            let records = root.buffers.iter_mut().find(|b| b.name == "completion_records").unwrap();
            records.nbytes = 2 * 4 * 24 * 8;
            records.shape = vec![2, 4, 24];
            records.strides = vec![4 * 24, 24, 1];
            root.buffers.push(BufferDescriptor {
                name: "proposal_confidence".into(), address: 0x30000, nbytes: 2 * 4 * 5 * 4,
                dtype: Dtype::Float32, shape: vec![2, 4, 5], strides: vec![20, 5, 1],
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
                ("speculative", GraphKind::Speculative, 4, 6, 0, 12),
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
                if kind == GraphKind::Speculative { graph.draft_tokens = 5; graph.confidence_width = 5; }
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
                bad.graphs[3].input_bindings[0].host_offset += 8;
                assert!(!same_layout(&ranks[0], &bad));
                let mut bad = ranks[1].clone();
                bad.buffers[2].shape[1] += 1;
                assert!(!same_layout(&ranks[0], &bad));
            }
        }
    }
}

#[cfg(all(test, feature = "runtime-package"))]
#[path = "threaded/cuda_tests.rs"]
mod cuda_tests;
