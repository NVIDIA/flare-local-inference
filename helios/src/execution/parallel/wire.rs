use std::io::{ErrorKind, Read, Write};
use std::time::Duration;

use crate::cache::Fence;
use crate::execution::{ExecutionManifest, FrameCompletion, OwnedUpload, RankStream, Upload};
use crate::{Error, ExecutionError, Failure, Result};

pub(super) const HELLO: u8 = 1;
pub(super) const PREPARE: u8 = 2;
pub(super) const READY: u8 = 3;
pub(super) const COMMIT: u8 = 4;
pub(super) const COMPLETE: u8 = 5;
pub(super) const DRAIN: u8 = 6;
pub(super) const DRAINED: u8 = 7;
pub(super) const RELEASE: u8 = 8;
pub(super) const FAILED: u8 = 9;
pub(super) const PREPARE_MORE: u8 = 10;
const MAX_PACKET: usize = 64 * 1024 * 1024;
pub(super) const TIMEOUT: Duration = Duration::from_secs(120);

pub(super) fn invalid(message: impl Into<String>) -> Error {
    Error::execution(ExecutionError::Worker, message)
}

fn io(error: std::io::Error) -> Error {
    ExecutionError::Worker(Failure::caused_by(
        "tensor-parallel control socket failed",
        error,
    ))
    .into()
}

pub(super) struct Channel {
    pub stream: RankStream,
    received: Vec<u8>,
}

impl Channel {
    pub fn new(stream: impl Into<RankStream>) -> Result<Self> {
        let stream = stream.into();
        stream.set_read_timeout(Some(TIMEOUT)).map_err(io)?;
        stream.set_write_timeout(Some(TIMEOUT)).map_err(io)?;
        stream.set_nonblocking(false).map_err(io)?;
        Ok(Self {
            stream,
            received: Vec::new(),
        })
    }

    pub fn nonblocking(&self, enabled: bool) -> Result<()> {
        self.stream.set_nonblocking(enabled).map_err(io)
    }

    pub fn idle(&self, enabled: bool) -> Result<()> {
        self.stream
            .set_read_timeout(if enabled { None } else { Some(TIMEOUT) })
            .map_err(io)
    }

    pub fn send(&mut self, tag: u8, data: &[u8]) -> Result<()> {
        if data.len() > MAX_PACKET {
            return Err(invalid(
                "tensor-parallel control packet exceeds its size limit",
            ));
        }
        let mut header = [0; 5];
        header[0] = tag;
        header[1..].copy_from_slice(&(data.len() as u32).to_le_bytes());
        self.stream.write_all(&header).map_err(io)?;
        self.stream.write_all(data).map_err(io)
    }

    pub fn poll(&mut self) -> Result<Option<(u8, Vec<u8>)>> {
        loop {
            let needed = if self.received.len() < 5 {
                5
            } else {
                let size = u32::from_le_bytes(self.received[1..5].try_into().unwrap()) as usize;
                if size > MAX_PACKET {
                    return Err(invalid("oversized tensor-parallel control packet"));
                }
                5 + size
            };
            if self.received.len() == needed && needed >= 5 {
                let bytes = std::mem::take(&mut self.received);
                if bytes[0] == FAILED {
                    return Err(invalid(format!(
                        "tensor-parallel peer failed: {}",
                        String::from_utf8_lossy(&bytes[5..])
                    )));
                }
                return Ok(Some((bytes[0], bytes[5..].to_vec())));
            }
            let mut bytes = [0; 8192];
            let count = bytes.len().min(needed - self.received.len());
            match self.stream.read(&mut bytes[..count]) {
                Ok(0) => return Err(invalid("tensor-parallel peer disconnected")),
                Ok(count) => self.received.extend_from_slice(&bytes[..count]),
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == ErrorKind::WouldBlock => return Ok(None),
                Err(error) => return Err(io(error)),
            }
        }
    }

    /// Wait for the next message, without a deadline when `idle`.
    pub fn wait(&mut self, idle: bool) -> Result<(u8, Vec<u8>)> {
        self.idle(idle)?;
        let message = self.poll();
        self.idle(false)?;
        message?.ok_or_else(|| invalid("tensor-parallel control deadline expired"))
    }

    /// Poll without blocking for up to `limit` before a caller's blocking wait.
    pub fn spin(&mut self, limit: std::time::Duration) -> Result<Option<(u8, Vec<u8>)>> {
        self.nonblocking(true)?;
        let started = std::time::Instant::now();
        let result = loop {
            match self.poll() {
                Ok(None) if started.elapsed() < limit => std::hint::spin_loop(),
                other => break other,
            }
        };
        self.nonblocking(false)?;
        result
    }

    /// Receive `expected`, polling briefly before blocking.
    pub fn handoff(&mut self, expected: u8, limit: std::time::Duration) -> Result<Vec<u8>> {
        match self.spin(limit)? {
            Some((tag, data)) if tag == expected => Ok(data),
            Some((tag, _)) => Err(invalid(format!("expected rank message {expected}, received {tag}"))),
            None => self.receive(expected),
        }
    }

    pub fn receive(&mut self, expected: u8) -> Result<Vec<u8>> {
        let (tag, data) = self.wait(false)?;
        if tag != expected {
            return Err(invalid(format!(
                "expected rank message {expected}, received {tag}"
            )));
        }
        Ok(data)
    }
}

pub(super) fn number(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}

fn string(bytes: &mut Vec<u8>, value: &str) {
    number(bytes, value.len());
    bytes.extend_from_slice(value.as_bytes());
}

pub(super) struct Reader<'a>(pub &'a [u8]);
impl<'a> Reader<'a> {
    pub fn bytes(&mut self, count: usize) -> Result<&'a [u8]> {
        if count > self.0.len() {
            return Err(invalid("truncated tensor-parallel control packet"));
        }
        let (value, rest) = self.0.split_at(count);
        self.0 = rest;
        Ok(value)
    }
    pub fn number(&mut self) -> Result<usize> {
        usize::try_from(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
            .map_err(|_| invalid("unaddressable tensor-parallel control value"))
    }
    pub fn end(self) -> Result<()> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(invalid("trailing tensor-parallel control bytes"))
        }
    }
}

/// Logical I/O schema only; CUDA addresses and handles never cross processes.
pub(super) fn layout(manifest: &ExecutionManifest) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in [
        4,
        manifest.abi_version as usize,
        manifest.max_active,
        manifest.input_frame_bytes,
        manifest.buffers.len(),
        manifest.graphs.len(),
    ] {
        number(&mut bytes, value);
    }
    for buffer in &manifest.buffers {
        string(&mut bytes, &buffer.name);
        string(&mut bytes, &format!("{:?}", buffer.dtype));
        number(&mut bytes, buffer.nbytes);
        number(&mut bytes, buffer.shape.len());
        for value in buffer.shape.iter().chain(&buffer.strides) {
            number(&mut bytes, *value);
        }
    }
    for graph in &manifest.graphs {
        string(&mut bytes, &graph.name);
        string(
            &mut bytes,
            &format!("{:?}/{:?}", graph.kind, graph.sampling_mode),
        );
        for value in [
            graph.batch_capacity,
            graph.token_capacity,
            graph.packed_token_capacity,
            graph.prefill_capacity,
            graph.input_bindings.len(),
        ] {
            number(&mut bytes, value);
        }
        if graph.kind == super::super::GraphKind::PackedSpeculative {
            number(&mut bytes, graph.draft_tokens);
        }
        for binding in &graph.input_bindings {
            string(&mut bytes, &binding.buffer_name);
            number(&mut bytes, binding.host_offset);
            number(&mut bytes, binding.nbytes);
        }
    }
    number(&mut bytes, usize::from(manifest.cache.is_some()));
    if let Some(cache) = &manifest.cache {
        for value in [cache.page_tokens, cache.page_capacity, cache.row_capacity] {
            number(&mut bytes, value as usize);
        }
        number(&mut bytes, cache.snapshot_pages.len());
        for range in &cache.snapshot_pages {
            number(&mut bytes, range.start as usize);
            number(&mut bytes, range.end as usize);
        }
    }
    bytes
}

/// Split validated uploads into bounded packets while preserving element alignment.
pub(super) fn prepare(
    manifest: &ExecutionManifest,
    fence: Fence,
    graph: usize,
    grammar: bool,
    uploads: &[Upload<'_>],
    mut send: impl FnMut(u8, &[u8]) -> Result<()>,
) -> Result<()> {
    let capacity = uploads.iter().fold(32usize, |size, upload| {
        size.saturating_add(24).saturating_add(upload.bytes.len())
    }).min(MAX_PACKET);
    let mut packet = Vec::with_capacity(capacity);
    for value in [fence.0 as usize, graph, usize::from(grammar), 0] {
        number(&mut packet, value);
    }
    let mut count = 0;
    let mut flush = |packet: &mut Vec<u8>, count: &mut usize, tag| -> Result<()> {
        packet[24..32].copy_from_slice(&(*count as u64).to_le_bytes());
        send(tag, packet)?;
        packet.truncate(32);
        *count = 0;
        Ok(())
    };
    for upload in uploads {
        let index = manifest.buffers.iter()
            .position(|buffer| buffer.name == upload.buffer)
            .ok_or_else(|| invalid("rank upload buffer is absent"))?;
        let alignment = manifest.buffers[index].dtype.size();
        let mut offset = upload.offset;
        let mut bytes = upload.bytes;
        while !bytes.is_empty() {
            let available = MAX_PACKET.saturating_sub(packet.len() + 24);
            let size = (available / alignment * alignment).min(bytes.len());
            if size == 0 {
                flush(&mut packet, &mut count, PREPARE_MORE)?;
                continue;
            }
            for value in [index, offset, size] {
                number(&mut packet, value);
            }
            packet.extend_from_slice(&bytes[..size]);
            count += 1;
            offset += size;
            bytes = &bytes[size..];
        }
    }
    flush(&mut packet, &mut count, PREPARE)
}

pub(super) fn receive_prepare(
    channel: &mut Channel,
    manifest: &ExecutionManifest,
    mut tag: u8,
    mut bytes: Vec<u8>,
) -> Result<(Fence, usize, bool, Vec<OwnedUpload>)> {
    let mut identity = None;
    let mut remaining = manifest.input_frame_bytes;
    let mut owned = Vec::new();
    loop {
        if ![PREPARE, PREPARE_MORE].contains(&tag) {
            return Err(invalid("expected rank prepare packet"));
        }
        let mut reader = Reader(&bytes);
        let fence = Fence(reader.number()? as u64);
        let graph = reader.number()?;
        let grammar = match reader.number()? {
            0 => false,
            1 => true,
            _ => return Err(invalid("invalid grammar flag")),
        };
        let received = (fence, graph, grammar);
        if identity.is_some_and(|expected| expected != received) {
            return Err(invalid("rank prepare packet identity differs"));
        }
        identity = Some(received);
        let part = uploads(manifest, &mut reader, &mut remaining)?;
        reader.end()?;
        if tag == PREPARE_MORE && part.is_empty() {
            return Err(invalid("empty rank prepare continuation"));
        }
        owned.extend(part);
        if tag == PREPARE {
            return Ok((fence, graph, grammar, owned));
        }
        (tag, bytes) = channel.wait(false)?;
    }
}

pub(super) fn uploads(
    manifest: &ExecutionManifest,
    data: &mut Reader<'_>,
    remaining: &mut usize,
) -> Result<Vec<OwnedUpload>> {
    let count = data.number()?;
    if count > manifest.input_frame_bytes || count > data.0.len() / 24 {
        return Err(invalid("invalid rank upload count"));
    }
    let mut uploads = Vec::with_capacity(count);
    for _ in 0..count {
        let buffer = manifest
            .buffers
            .get(data.number()?)
            .ok_or_else(|| invalid("rank upload buffer index out of range"))?;
        let offset = data.number()?;
        let count = data.number()?;
        if count == 0 || count > *remaining {
            return Err(invalid("rank uploads exceed the input-frame extent"));
        }
        *remaining -= count;
        uploads.push(OwnedUpload {
            buffer: buffer.name.clone(),
            offset,
            bytes: data.bytes(count)?.to_vec(),
        });
    }
    Ok(uploads)
}

pub(super) fn completion(value: &FrameCompletion) -> Vec<u8> {
    let mut bytes = Vec::new();
    number(&mut bytes, value.fence.0 as usize);
    number(&mut bytes, value.graph_index);
    number(&mut bytes, usize::from(value.timing.gpu_ns[0].is_some()));
    number(&mut bytes, value.timing.gpu_ns[0].unwrap_or(0) as usize);
    number(&mut bytes, value.timing.wall_ns as usize);
    number(&mut bytes, usize::from(value.timing.queued));
    number(&mut bytes, value.timing.readiness.map_or(0, |samples| samples.len));
    if let Some(samples) = value.timing.readiness {
        for pair in &samples.intervals[..samples.len] {
            number(&mut bytes, pair[0] as usize);
            number(&mut bytes, pair[1] as usize);
        }
    }
    number(&mut bytes, usize::from(value.proposal_confidence.is_some()));
    if let Some(rows) = &value.proposal_confidence {
        for value in rows.iter().flatten() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    number(&mut bytes, usize::from(value.speculative_survival.is_some()));
    if let Some(rows) = &value.speculative_survival {
        for value in rows.iter().flatten() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    for record in value.records.iter() {
        for field in record.fields() {
            bytes.extend_from_slice(&field.to_le_bytes());
        }
    }
    bytes
}

pub(super) fn proposal_confidence(reader: &mut Reader<'_>) -> Result<Option<[[f32; 7]; 64]>> {
    match reader.number()? {
        0 => Ok(None),
        1 => {
            let mut rows = [[0.0; 7]; 64];
            for value in rows.iter_mut().flatten() {
                *value = f32::from_le_bytes(reader.bytes(4)?.try_into().unwrap());
            }
            Ok(Some(rows))
        }
        _ => Err(invalid("invalid proposal confidence presence")),
    }
}

pub(super) fn speculative_survival(reader: &mut Reader<'_>) -> Result<Option<[[f32; 7]; 64]>> {
    match reader.number()? {
        0 => Ok(None),
        1 => {
            let mut rows = [[-1.0; 7]; 64];
            for value in rows.iter_mut().flatten() {
                *value = f32::from_le_bytes(reader.bytes(4)?.try_into().unwrap());
            }
            Ok(Some(rows))
        }
        _ => Err(invalid("invalid speculative survival presence")),
    }
}

pub(super) fn timing(reader: &mut Reader<'_>) -> Result<super::super::ExecutionTiming> {
    let present = reader.number()?;
    let gpu_ns = reader.number()? as u64;
    let wall_ns = reader.number()? as u64;
    let queued = reader.number()?;
    if present > 1 || queued > 1 || (present == 0 && gpu_ns != 0) {
        return Err(invalid("invalid rank timing interval"));
    }
    let count = reader.number()?;
    if count > super::super::MAX_READINESS_INTERVALS {
        return Err(invalid("rank readiness interval count exceeds capacity"));
    }
    let readiness = if count == 0 { None } else {
        let mut samples = super::super::ReadinessIntervals { len: count, ..Default::default() };
        for pair in &mut samples.intervals[..count] {
            *pair = [reader.number()? as u64, reader.number()? as u64];
            if pair[0] < pair[1] {
                return Err(invalid("rank readiness interval excludes part of its barrier"));
            }
        }
        Some(samples)
    };
    Ok(super::super::ExecutionTiming {
        readiness,
        gpu_ns: [
            (present == 1).then_some(gpu_ns),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        wall_ns,
        queued: queued == 1,
    })
}
