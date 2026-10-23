//! Native rank coordination over process-local CUDA executors and control sockets.

#[cfg(test)]
mod tests;
mod wire;

use std::time::Instant;

use super::{ExecutionManifest, FrameCompletion, GraphExecutor, RankStream, Upload};
use crate::{Error, Result, Shutdown, cache::Fence};
use wire::*;

// Rank exchanges within a step are brief; poll that long before blocking, as threaded ranks do.
const HANDOFF_SPIN: std::time::Duration = std::time::Duration::from_micros(500);

struct Peer {
    channel: Channel,
    completion: Option<FrameCompletion>,
}

/// Rank zero owns request scheduling; peer processes execute identical graph steps.
/// One step can be outstanding. Its fence retires only after every rank completes.
pub struct TensorParallelExecutor {
    local: GraphExecutor,
    peers: Vec<Peer>,
    pending: Option<(Fence, usize, Instant)>,
    completion: Option<FrameCompletion>,
    poison: Option<Error>,
    shutdown: Option<Shutdown>,
}

impl TensorParallelExecutor {
    /// # Safety
    /// GraphExecutor's ownership contract applies. Each socket must connect to a
    /// distinct prepared rank in order 1..world; every owner includes its b12x IPC
    /// runtime and remains retained until coordinated shutdown confirms drain.
    pub unsafe fn new(manifest: ExecutionManifest, sockets: Vec<RankStream>) -> Result<Self> {
        let world = sockets.len() + 1;
        if world > 16 {
            return Err(invalid("tensor parallelism supports at most sixteen ranks"));
        }
        if world < 2 {
            return Err(invalid("tensor parallelism requires at least two ranks"));
        }
        manifest.validate()?;
        let expected = layout(&manifest);
        let mut peers = Vec::new();
        for (index, socket) in sockets.into_iter().enumerate() {
            let mut channel = Channel::new(socket)?;
            let hello = channel.receive(HELLO)?;
            let mut reader = Reader(&hello);
            if reader.number()? != index + 1 || reader.number()? != world || reader.0 != expected {
                return Err(invalid(
                    "tensor-parallel rank identity, graph I/O schema, or cache ownership layout differs",
                ));
            }
            peers.push(Peer {
                channel,
                completion: None,
            });
        }
        // SAFETY: The caller retains all rank-local prepared resources through drain.
        let local = unsafe { GraphExecutor::new(manifest)? };
        for peer in &mut peers {
            peer.channel.send(READY, &[])?;
        }
        Ok(Self {
            local,
            peers,
            pending: None,
            completion: None,
            poison: None,
            shutdown: None,
        })
    }

    pub fn outstanding(&self) -> usize {
        usize::from(self.pending.is_some())
    }

    pub fn manifest(&self) -> &ExecutionManifest {
        self.local.manifest()
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
            return Err(invalid("tensor-parallel executor is closed"));
        }
        self.local.next_fence()
    }

    pub fn submit(
        &mut self,
        graph: usize,
        uploads: &[Upload<'_>],
        grammar: bool,
    ) -> Result<Option<Fence>> {
        if let Some(error) = &self.poison {
            return Err(error.clone());
        }
        if self.shutdown.is_some() {
            return Err(invalid("tensor-parallel executor is closed"));
        }
        if self.pending.is_some() {
            return Ok(None);
        }
        let Some(fence) = self.local.validate_submission(graph, uploads, grammar)? else {
            return Ok(None);
        };
        let result = self.prepare_and_commit(graph, uploads, grammar, fence);
        if let Err(error) = &result {
            self.poison = Some(error.clone());
        }
        result.map(Some)
    }

    fn prepare_and_commit(
        &mut self,
        graph: usize,
        uploads: &[Upload<'_>],
        grammar: bool,
        fence: Fence,
    ) -> Result<Fence> {
        let dispatched = Instant::now();
        for peer in &mut self.peers {
            peer.channel.nonblocking(false)?;
        }
        wire::prepare(self.local.manifest(), fence, graph, grammar, uploads, |tag, bytes| {
            for peer in &mut self.peers {
                peer.channel.send(tag, bytes)?;
            }
            Ok(())
        })?;
        for peer in &mut self.peers {
            if !peer.channel.handoff(READY, HANDOFF_SPIN)?.is_empty() {
                return Err(invalid("unexpected rank readiness payload"));
            }
        }
        // Every rank has validated the same inputs and reserved the same next fence.
        self.pending = Some((fence, graph, dispatched));
        for peer in &mut self.peers {
            peer.channel.send(COMMIT, &[])?;
        }
        if self.local.submit(graph, uploads, grammar)? != Some(fence) {
            return Err(invalid(
                "rank-zero submission changed after coordinated validation",
            ));
        }
        for peer in &self.peers {
            peer.channel.nonblocking(true)?;
        }
        Ok(fence)
    }

    pub fn poll(&mut self) -> Result<Option<FrameCompletion>> {
        if let Some(error) = &self.poison {
            return Err(error.clone());
        }
        if self.shutdown.is_some() {
            return Err(invalid("tensor-parallel executor is closed"));
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
        let Some((fence, graph, started)) = self.pending else {
            return Ok(None);
        };
        if started.elapsed() > TIMEOUT {
            return Err(invalid("tensor-parallel graph completion deadline expired"));
        }
        if self.completion.is_none() {
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
            if let Some((tag, bytes)) = peer.channel.poll()? {
                if tag != COMPLETE {
                    return Err(invalid("expected rank graph completion"));
                }
                let mut reader = Reader(&bytes);
                if reader.number()? != fence.0 as usize || reader.number()? != graph {
                    return Err(invalid("rank completion fence or graph differs"));
                }
                let timing = wire::timing(&mut reader)?;
                let proposal_confidence = wire::proposal_confidence(&mut reader)?;
                let speculative_survival = wire::speculative_survival(&mut reader)?;
                if reader.0.len() != self.local.manifest().completion_bytes() {
                    return Err(invalid("rank completion size differs from its ABI"));
                }
                peer.completion = Some(FrameCompletion {
                    logprobs: None,
                    prompt_logprobs: None,
                    proposal_confidence,
                    speculative_survival,
                    fence,
                    graph_index: graph,
                    timing,
                    records: super::transport::decode_records(
                        reader.0,
                        fence,
                        self.local.manifest().request_capacity(),
                        self.local.manifest().completion_fields(),
                    )?,
                });
            }
        }
        if self.completion.is_none() || self.peers.iter().any(|peer| peer.completion.is_none()) {
            return Ok(None);
        }
        let root = self.completion.as_ref().unwrap();
        for (index, peer) in self.peers.iter().enumerate() {
            let agreement = if self.local.manifest().graphs[root.graph_index].kind == super::GraphKind::PackedSpeculative { agree_packed } else { agree };
            agreement(root, peer.completion.as_ref().unwrap())
                .map_err(|error| invalid(format!("rank {}: {error}", index + 1)))?;
        }
        for (index, peer) in self.peers.iter_mut().enumerate() {
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
        let result = (|| -> Result<()> {
            while self.pending.is_some() {
                self.poll()?;
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
            if let Some(error) = &self.poison {
                return Err(error.clone());
            }
            for peer in &mut self.peers {
                peer.channel.nonblocking(false)?;
                peer.channel.send(DRAIN, &[])?;
            }
            for peer in &mut self.peers {
                if !peer.channel.receive(DRAINED)?.is_empty() {
                    return Err(invalid("unexpected rank drain payload"));
                }
            }
            Ok(())
        })();
        let shutdown = match result {
            Err(error) => self.local.abandon(error),
            Ok(()) => self.local.close(),
        };
        if shutdown.drained {
            // All GPU streams have drained. A lost release can retain a peer's
            // owner, but cannot make any rank free resources still used by CUDA.
            for peer in &mut self.peers {
                let _ = peer.channel.send(RELEASE, &[]);
            }
        }
        for peer in &self.peers {
            let _ = peer.channel.stream.shutdown(std::net::Shutdown::Both);
        }
        self.shutdown = Some(shutdown.clone());
        shutdown
    }
}

impl Drop for TensorParallelExecutor {
    fn drop(&mut self) {
        self.close();
    }
}

pub(super) fn agree(root: &FrameCompletion, peer: &FrameCompletion) -> Result<()> {
    if root.fence != peer.fence || root.graph_index != peer.graph_index {
        return Err(invalid("rank completion identity differs"));
    }
    if root.records.len() != peer.records.len() {
        return Err(invalid("rank completion record counts differ"));
    }
    match (&root.proposal_confidence, &peer.proposal_confidence) {
        (None, None) => {}
        (Some(root_values), Some(peer_values)) => {
            // Only the root completion is returned to the shared scheduler.
            // Peer estimates do not select execution or enter the root's curve;
            // their floating-point agreement is not a control-plane invariant.
            // Validate every reported probability and keep exact request-state
            // agreement below, without requiring replicated arithmetic identity.
            for (row, (a, b)) in root_values.iter().zip(peer_values).enumerate() {
                for (draft, (&a, &b)) in a.iter().zip(b).enumerate() {
                    if !(0.0..=1.0).contains(&a) || !(0.0..=1.0).contains(&b) {
                        return Err(invalid(format!(
                            "tensor-parallel proposal confidence is invalid at row {row}, draft {draft}: root {a:?}, peer {b:?}; graph {}, fence {}",
                            root.graph_index, root.fence.0,
                        )));
                    }
                }
            }
        }
        _ => return Err(invalid("tensor-parallel proposal confidence presence differs")),
    }
    match (&root.speculative_survival, &peer.speculative_survival) {
        (None, None) => {}
        (Some(root_values), Some(peer_values)) => {
            super::validate_speculative_survival(root_values, &root.records)?;
            super::validate_speculative_survival(peer_values, &peer.records)?;
            if root_values.iter().flatten().zip(peer_values.iter().flatten())
                .any(|(a, b)| (*a == -1.0) != (*b == -1.0)) {
                return Err(invalid("tensor-parallel speculative survival observation extents differ"));
            }
        }
        _ => return Err(invalid("tensor-parallel speculative survival presence differs")),
    }
    for (lane, (a, b)) in root.records.iter().zip(peer.records.iter()).enumerate() {
        if a.generation != b.generation
            || a.submission_sequence != b.submission_sequence
            || a.consumed_tokens != b.consumed_tokens
            || a.error_code != b.error_code
            || !(a.status == b.status || a.status == 2 && b.status == 1)
        {
            return Err(invalid(format!(
                "tensor-parallel request epochs, consumption, or status differ at graph {}, fence {}, lane {lane}: root {a:?}, peer {b:?}",
                root.graph_index, root.fence.0,
            )));
        }
    }
    Ok(())
}

pub(super) fn agree_packed(root: &FrameCompletion, peer: &FrameCompletion) -> Result<()> {
    agree(root, peer)?;
    for (a, b) in root.records.iter().zip(peer.records.iter()) {
        // ABI3 leaves inactive token/vector payload undefined. Retain the base
        // transport agreement above, and compare only committed active values.
        if a.status == 0 && b.status == 0 { continue; }
        if a.generation != b.generation || a.submission_sequence != b.submission_sequence
            || a.consumed_tokens != b.consumed_tokens || a.generated_tokens != b.generated_tokens
            || a.token_id != b.token_id || a.status != b.status || a.error_code != b.error_code
            || a.reserved != b.reserved || a.vector != b.vector {
            return Err(invalid("packed combined rank completion prefixes or committed state differ"));
        }
    }
    Ok(())
}

/// Run a peer's native control loop until coordinated drain and release.
///
/// # Safety
/// The caller retains all graph owners and b12x IPC mappings when drained is
/// false. A failed collective requires coordinated process termination before
/// reclaiming those resources; this function never synchronizes a stranded graph.
pub unsafe fn serve_rank(
    manifest: ExecutionManifest,
    socket: RankStream,
    rank: usize,
    world: usize,
) -> Shutdown {
    let mut executor = match unsafe { GraphExecutor::new(manifest) } {
        Ok(executor) => executor,
        Err(error) => {
            return Shutdown {
                drained: true,
                error: Some(error),
            };
        }
    };
    executor.copy_logprobs = false;
    let result = (|| -> Result<()> {
        if world < 2 || !(1..world).contains(&rank) {
            return Err(invalid("invalid tensor-parallel peer identity"));
        }
        let mut channel = Channel::new(socket)?;
        let result = (|| -> Result<()> {
            let mut hello = Vec::new();
            number(&mut hello, rank);
            number(&mut hello, world);
            hello.extend(layout(executor.manifest()));
            channel.send(HELLO, &hello)?;
            if !channel.receive(READY)?.is_empty() {
                return Err(invalid("unexpected coordinator readiness payload"));
            }
            loop {
                let (tag, bytes) = match channel.spin(HANDOFF_SPIN)? {
                    Some(message) => message,
                    None => channel.wait(true)?,
                };
                if tag == DRAIN && bytes.is_empty() {
                    let shutdown = executor.close();
                    if !shutdown.drained {
                        return Err(shutdown
                            .error
                            .unwrap_or_else(|| invalid("rank did not drain")));
                    }
                    channel.send(DRAINED, &[])?;
                    if !channel.receive(RELEASE)?.is_empty() {
                        return Err(invalid("unexpected rank release payload"));
                    }
                    return Ok(());
                }
                if ![PREPARE, PREPARE_MORE].contains(&tag) {
                    return Err(invalid("expected rank prepare or drain"));
                }
                let (fence, graph, grammar, owned) =
                    receive_prepare(&mut channel, executor.manifest(), tag, bytes)?;
                let uploads: Vec<_> = owned
                    .iter()
                    .map(|upload| Upload {
                        buffer: &upload.buffer,
                        offset: upload.offset,
                        bytes: &upload.bytes,
                    })
                    .collect();
                if executor.validate_submission(graph, &uploads, grammar)? != Some(fence) {
                    return Err(invalid("rank submission sequence or capacity differs"));
                }
                channel.send(READY, &[])?;
                if !channel.handoff(COMMIT, HANDOFF_SPIN)?.is_empty() {
                    return Err(invalid("unexpected rank commit payload"));
                }
                if executor.submit(graph, &uploads, grammar)? != Some(fence) {
                    return Err(invalid("rank lost its prepared submission"));
                }
                let started = Instant::now();
                channel.nonblocking(true)?;
                let completed = loop {
                    match executor.poll() {
                        Ok(None) => {}
                        Ok(Some(completion)) => break completion,
                        Err(error) => return Err(error),
                    }
                    if channel.poll()?.is_some() {
                        return Err(invalid("rank command arrived before graph completion"));
                    }
                    if started.elapsed() > TIMEOUT {
                        return Err(invalid("rank graph completion deadline expired"));
                    }
                    std::thread::yield_now();
                };
                channel.nonblocking(false)?;
                channel.send(COMPLETE, &completion(&completed))?;
            }
        })();
        if let Err(error) = &result {
            let _ = channel.nonblocking(false);
            let _ = channel.send(FAILED, error.to_string().as_bytes());
        }
        result
    })();
    match result {
        Ok(()) => executor.close(),
        Err(error) => executor.abandon(error),
    }
}
