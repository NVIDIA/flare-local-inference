/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bounded safetensors containers for payload-based packages.
//!
//! Both canonical forms use the same schema: a primary with metadata tensors and
//! optional payload tensors, and directory companions that each hold exactly one
//! `flare.payload` U8 tensor. Containers carry bytes only; signatures bind
//! payloads, so packing and unpacking copy bytes without cryptographic work.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::File,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(not(unix))]
use std::fs::{self, OpenOptions};

use super::{PayloadId, invalid};
use crate::{Result, os::FileReadAt};
use ring::digest;
use serde::de::{self, Deserialize, Deserializer, MapAccess, Visitor};

pub use super::PayloadId as Identity;

/// Header metadata format for companion and packed payload containers.
pub const PAYLOAD_FORMAT: &str = "flare-payload";
/// Frozen tensor name inside a directory companion.
pub const COMPANION_TENSOR: &str = "flare.payload";
/// Reserved prefix for alignment tensors the writer inserts between payloads.
pub const ALIGNMENT_PREFIX: &str = "align.";
/// Payload bases and container data starts are aligned to this boundary.
pub const ALIGNMENT: u64 = 4096;
/// Metadata bytes are bounded; bulk payload bytes are not.
pub const MAX_HEADER: u64 = 128 << 20;
/// Bound on header tensor entries, including alignment tensors.
pub const MAX_TENSORS: usize = 1 << 20;
/// Bounded streaming buffer for payload copies and digests.
pub const IO_CHUNK: usize = 1 << 20;
/// Maximum bytes in one relative package path.
pub const MAX_PATH: usize = 1024;

fn io(context: &str, error: io::Error) -> crate::Error {
    invalid(format!("{context}: {error}"))
}

#[derive(Debug, Clone, Copy)]
struct TensorSpan {
    offset: u64,
    len: u64,
}

/// A pinned byte slice inside one package file.
///
/// The retained handle is the one used for direct I/O, so a caller that opened a
/// companion keeps that exact file for the slice's lifetime.
#[derive(Clone)]
pub struct Slice {
    file: Arc<File>,
    offset: u64,
    length: u64,
}

#[cfg(all(unix, any(test, feature = "test-tools")))]
static READ_COUNTER: std::sync::Mutex<BTreeMap<(u64, u64), u64>> =
    std::sync::Mutex::new(BTreeMap::new());

/// Count payload bytes read from selected files, including reads on worker threads.
#[cfg(all(unix, any(test, feature = "test-tools")))]
pub fn count_payload_reads<T>(paths: &[PathBuf], operation: impl FnOnce() -> T) -> (T, u64) {
    use std::os::unix::fs::MetadataExt;
    let files: BTreeSet<_> = paths
        .iter()
        .map(|path| {
            let metadata = std::fs::metadata(path).unwrap();
            (metadata.dev(), metadata.ino())
        })
        .collect();
    struct Reset(BTreeSet<(u64, u64)>);
    impl Drop for Reset {
        fn drop(&mut self) {
            let mut counter = READ_COUNTER.lock().unwrap();
            for file in &self.0 {
                counter.remove(file);
            }
        }
    }
    {
        let mut counter = READ_COUNTER.lock().unwrap();
        for file in &files {
            assert!(counter.insert(*file, 0).is_none());
        }
    }
    let reset = Reset(files);
    let result = operation();
    let bytes = {
        let counter = READ_COUNTER.lock().unwrap();
        reset.0.iter().map(|file| counter[file]).sum()
    };
    (result, bytes)
}

impl Slice {
    pub fn new(file: Arc<File>, offset: u64, length: u64) -> Result<Self> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| invalid("payload slice overflow"))?;
        let size = file
            .metadata()
            .map_err(|e| io("read payload length", e))?
            .len();
        if end > size {
            return Err(invalid(format!(
                "payload slice {offset}+{length} exceeds file size {size}"
            )));
        }
        Ok(Self {
            file,
            offset,
            length,
        })
    }
    pub fn file(&self) -> &Arc<File> {
        &self.file
    }
    pub fn offset(&self) -> u64 {
        self.offset
    }
    pub fn len(&self) -> u64 {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    pub fn read_at(&self, offset: u64, output: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(output.len() as u64)
            .ok_or_else(|| invalid("payload read overflow"))?;
        if end > self.length {
            return Err(invalid(format!(
                "payload read {offset}+{} exceeds slice length {}",
                output.len(),
                self.length
            )));
        }
        #[cfg(all(unix, any(test, feature = "test-tools")))]
        {
            use std::os::unix::fs::MetadataExt;
            let mut counter = READ_COUNTER.lock().unwrap();
            if !counter.is_empty() {
                let metadata = self.file.metadata().unwrap();
                if let Some(bytes) = counter.get_mut(&(metadata.dev(), metadata.ino())) {
                    *bytes += output.len() as u64;
                }
            }
        }
        self.file
            .read_exact_at(output, self.offset + offset)
            .map_err(|e| io("read payload bytes", e))
    }
    /// Streams the whole slice into `output` through a bounded buffer.
    pub fn copy_to(&self, output: &mut impl Write) -> Result<()> {
        let mut buffer = vec![0u8; IO_CHUNK];
        let mut offset = 0;
        while offset < self.length {
            let count = (self.length - offset).min(buffer.len() as u64) as usize;
            self.read_at(offset, &mut buffer[..count])?;
            output
                .write_all(&buffer[..count])
                .map_err(|e| io("write payload bytes", e))?;
            offset += count as u64;
        }
        Ok(())
    }
    /// Hashes the exact stored payload bytes, excluding any container framing.
    pub fn digest(&self) -> Result<PayloadId> {
        self.digest_with(None)
    }
    /// Hashes the stored payload bytes, recording each read as verification
    /// work so a long payload reports progress while it is being checked.
    pub fn digest_with(
        &self,
        progress: Option<&crate::package::progress::AcquisitionProgress>,
    ) -> Result<PayloadId> {
        let mut context = digest::Context::new(&digest::SHA256);
        let mut buffer = vec![0u8; IO_CHUNK];
        let mut offset = 0;
        while offset < self.length {
            let count = (self.length - offset).min(buffer.len() as u64) as usize;
            self.read_at(offset, &mut buffer[..count])?;
            context.update(&buffer[..count]);
            if let Some(progress) = progress {
                progress.record_verification_bytes(count as u64);
            }
            offset += count as u64;
        }
        Ok(PayloadId::from_digest(context.finish()))
    }
}

/// Parsed, validated container header with pinned file access.
#[derive(Debug)]
pub struct Container {
    file: Arc<File>,
    data_start: u64,
    data_len: u64,
    format: String,
    tensors: BTreeMap<String, TensorSpan>,
}

impl Container {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| io("open package container", e))?;
        Self::open_pinned(Arc::new(file))
    }

    /// Validates an already-open handle and retains it for all slices.
    pub fn open_pinned(file: Arc<File>) -> Result<Self> {
        let size = file
            .metadata()
            .map_err(|e| io("read container length", e))?
            .len();
        let mut prefix = [0u8; 8];
        file.read_exact_at(&mut prefix, 0)
            .map_err(|e| io("read container header length", e))?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len == 0 || header_len > MAX_HEADER || header_len % ALIGNMENT != ALIGNMENT - 8 {
            return Err(invalid("invalid package header length"));
        }
        let data_start = header_len
            .checked_add(8)
            .ok_or_else(|| invalid("package header overflow"))?;
        if data_start > size {
            return Err(invalid("package header exceeds file size"));
        }
        let mut header = vec![0u8; header_len as usize];
        file.read_exact_at(&mut header, 8)
            .map_err(|e| io("read container header", e))?;
        if header.contains(&0) {
            return Err(invalid("package header contains a zero byte"));
        }
        let raw: RawHeader = serde_json::from_slice(&header)
            .map_err(|e| invalid(format!("invalid package header: {e}")))?;
        let data_len = size - data_start;
        let mut spans: Vec<(String, u64, u64)> = Vec::with_capacity(raw.tensors.len());
        for (name, tensor) in raw.tensors {
            let [begin, end] = tensor.data_offsets[..] else {
                return Err(invalid(format!(
                    "tensor {name:?} requires two data offsets"
                )));
            };
            if tensor.dtype != "U8" {
                return Err(invalid(format!("tensor {name:?} is not U8")));
            }
            if tensor.shape.len() != 1 || tensor.shape[0] != end.saturating_sub(begin) {
                return Err(invalid(format!("tensor {name:?} is not one-dimensional")));
            }
            if begin > end || end > data_len {
                return Err(invalid(format!("tensor {name:?} exceeds container data")));
            }
            if !name.starts_with(ALIGNMENT_PREFIX) && begin % ALIGNMENT != 0 {
                return Err(invalid(format!("tensor {name:?} is not aligned")));
            }
            spans.push((name, begin, end - begin));
        }
        spans.sort_by_key(|(_, offset, _)| *offset);
        let mut cursor = 0u64;
        for (name, offset, len) in &spans {
            if *offset != cursor {
                return Err(invalid(format!(
                    "tensor {name:?} leaves a container gap or overlap"
                )));
            }
            cursor = cursor
                .checked_add(*len)
                .ok_or_else(|| invalid("container extent overflow"))?;
        }
        if cursor != data_len {
            return Err(invalid(
                "container tensors do not cover the payload exactly",
            ));
        }
        Ok(Self {
            file,
            data_start,
            data_len,
            format: raw.format,
            tensors: spans
                .into_iter()
                .map(|(name, offset, len)| (name, TensorSpan { offset, len }))
                .collect(),
        })
    }

    pub fn file(&self) -> &Arc<File> {
        &self.file
    }
    pub fn format(&self) -> &str {
        &self.format
    }
    pub fn data_len(&self) -> u64 {
        self.data_len
    }
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
    pub fn payload_names(&self) -> impl Iterator<Item = &str> {
        self.tensor_names()
            .filter(|name| !name.starts_with(ALIGNMENT_PREFIX))
    }
    pub fn len(&self, name: &str) -> Result<u64> {
        self.tensors
            .get(name)
            .map(|span| span.len)
            .ok_or_else(|| invalid(format!("container lacks tensor {name:?}")))
    }
    pub fn tensor(&self, name: &str) -> Result<Slice> {
        let span = self
            .tensors
            .get(name)
            .ok_or_else(|| invalid(format!("container lacks tensor {name:?}")))?;
        let offset = self
            .data_start
            .checked_add(span.offset)
            .ok_or_else(|| invalid("container tensor offset overflow"))?;
        Ok(Slice {
            file: Arc::clone(&self.file),
            offset,
            length: span.len,
        })
    }
}

/// Bytes supplied to the writer for one tensor.
pub enum Source<'a> {
    Slice(Slice),
    Bytes(Vec<u8>),
    /// Fills the declared length sequentially from `reader(offset, buffer)`.
    /// Streaming keeps acquisition from buffering a whole payload in memory.
    Reader(Box<dyn FnMut(u64, &mut [u8]) -> Result<()> + 'a>),
    Zero(u64),
}

/// Deterministic safetensors writer with 4096-aligned payload bases.
pub struct Writer<'a, W: Write> {
    output: W,
    format: String,
    entries: Vec<(String, u64, Source<'a>)>,
    names: BTreeSet<String>,
}

impl<'a, W: Write> Writer<'a, W> {
    pub fn new(output: W, format: impl Into<String>) -> Result<Self> {
        let format = format.into();
        validate_format(&format)?;
        Ok(Self {
            output,
            format,
            entries: Vec::new(),
            names: BTreeSet::new(),
        })
    }

    pub fn add(&mut self, name: &str, len: u64, source: Source<'a>) -> Result<()> {
        validate_tensor_name(name)?;
        if name.starts_with(ALIGNMENT_PREFIX) {
            return Err(invalid(
                "tensor names beginning with the alignment prefix are reserved",
            ));
        }
        if len == 0 {
            return Err(invalid("package payload tensors must not be empty"));
        }
        if let Source::Slice(slice) = &source
            && slice.len() != len
        {
            return Err(invalid(
                "declared tensor length differs from its source slice",
            ));
        }
        if let Source::Bytes(bytes) = &source
            && bytes.len() as u64 != len
        {
            return Err(invalid(
                "declared tensor length differs from its source bytes",
            ));
        }
        if !self.names.insert(name.to_owned()) {
            return Err(invalid(format!("duplicate tensor {name:?}")));
        }
        self.entries.push((name.to_owned(), len, source));
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        let mut cursor = 0u64;
        let mut layout: Vec<(String, u64, u64, Source<'a>)> =
            Vec::with_capacity(self.entries.len());
        let mut padding = 0usize;
        for (name, len, source) in std::mem::take(&mut self.entries) {
            let pad = (ALIGNMENT - cursor % ALIGNMENT) % ALIGNMENT;
            if pad > 0 {
                let name = loop {
                    let candidate = format!("{ALIGNMENT_PREFIX}{padding:06}");
                    padding += 1;
                    if !self.names.contains(&candidate) {
                        break candidate;
                    }
                };
                self.names.insert(name.clone());
                layout.push((name, cursor, pad, Source::Zero(pad)));
                cursor += pad;
            }
            layout.push((name, cursor, len, source));
            cursor = cursor
                .checked_add(len)
                .ok_or_else(|| invalid("container payload overflow"))?;
        }
        if cursor == 0 {
            return Err(invalid("container requires at least one payload tensor"));
        }
        let header = header_bytes(&self.format, &layout)?;
        self.output
            .write_all(&((header.len() as u64).to_le_bytes()))
            .map_err(|e| io("write container header length", e))?;
        self.output
            .write_all(&header)
            .map_err(|e| io("write container header", e))?;
        let mut buffer = vec![0u8; IO_CHUNK];
        for (name, _, len, mut source) in layout {
            write_source(&mut self.output, &name, len, &mut source, &mut buffer)?;
        }
        self.output.flush().map_err(|e| io("flush container", e))
    }
}

fn header_json(format: &str, layout: &[(String, u64, u64, Source<'_>)]) -> Result<Vec<u8>> {
    let mut tensors: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    tensors.insert(
        "__metadata__".into(),
        serde_json::json!({ "format": format }),
    );
    for (name, offset, len, _) in layout {
        let end = offset
            .checked_add(*len)
            .ok_or_else(|| invalid("container tensor overflow"))?;
        tensors.insert(
            name.clone(),
            serde_json::json!({ "dtype": "U8", "shape": [len], "data_offsets": [offset, end] }),
        );
    }
    serde_json::to_vec(&tensors).map_err(|e| invalid(format!("encode container header: {e}")))
}

fn padded_header_len(json: u64) -> Result<u64> {
    let mut length = json;
    while length % ALIGNMENT != ALIGNMENT - 8 {
        length = length
            .checked_add(1)
            .ok_or_else(|| invalid("container header overflow"))?;
        if length > MAX_HEADER {
            return Err(invalid("container header exceeds the metadata limit"));
        }
    }
    Ok(length)
}

fn header_bytes(format: &str, layout: &[(String, u64, u64, Source<'_>)]) -> Result<Vec<u8>> {
    let mut header = header_json(format, layout)?;
    let length = padded_header_len(header.len() as u64)?;
    header.resize(length as usize, b' ');
    Ok(header)
}

fn write_source<W: Write>(
    output: &mut W,
    name: &str,
    len: u64,
    source: &mut Source<'_>,
    buffer: &mut [u8],
) -> Result<()> {
    match source {
        Source::Slice(slice) => {
            if slice.len() != len {
                return Err(invalid(format!("tensor {name:?} source length changed")));
            }
            slice.copy_to(output)
        }
        Source::Bytes(bytes) => {
            if bytes.len() as u64 != len {
                return Err(invalid(format!("tensor {name:?} source length changed")));
            }
            output
                .write_all(bytes)
                .map_err(|e| io("write tensor bytes", e))
        }
        Source::Reader(reader) => {
            let chunk = buffer.len() as u64;
            let mut offset = 0;
            while offset < len {
                let count = (len - offset).min(chunk) as usize;
                reader(offset, &mut buffer[..count])?;
                output
                    .write_all(&buffer[..count])
                    .map_err(|e| io("write tensor bytes", e))?;
                offset += count as u64;
            }
            Ok(())
        }
        Source::Zero(count) => {
            if *count != len {
                return Err(invalid(format!("tensor {name:?} padding length changed")));
            }
            buffer.fill(0);
            let mut remaining = len;
            while remaining > 0 {
                let chunk = remaining.min(buffer.len() as u64) as usize;
                output
                    .write_all(&buffer[..chunk])
                    .map_err(|e| io("write tensor padding", e))?;
                remaining -= chunk as u64;
            }
            Ok(())
        }
    }
}

/// Writes the deterministic one-payload companion wrapper for `payload`.
pub fn write_companion(output: &mut impl Write, payload: &Slice) -> Result<()> {
    let mut writer = Writer::new(&mut *output, PAYLOAD_FORMAT)?;
    writer.add(
        COMPANION_TENSOR,
        payload.len(),
        Source::Slice(payload.clone()),
    )?;
    writer.finish()
}

/// Maps a companion path to its packed tensor name.
pub fn packed_tensor_name(relative: &str) -> Result<String> {
    validate_relative_path(relative)?;
    let stem = relative
        .strip_suffix(".safetensors")
        .ok_or_else(|| invalid("companion path must end in .safetensors"))?;
    if stem.split('/').any(|segment| segment.contains('.')) {
        return Err(invalid("companion path segments must not contain dots"));
    }
    Ok(stem.replace('/', "."))
}

/// Maps a packed tensor name to its companion path.
pub fn companion_path(tensor: &str) -> Result<String> {
    validate_tensor_name(tensor)?;
    if tensor.starts_with(ALIGNMENT_PREFIX) || tensor.contains('/') {
        return Err(invalid("invalid packed payload tensor name"));
    }
    let mut segments: Vec<&str> = tensor.split('.').collect();
    let name = segments
        .pop()
        .ok_or_else(|| invalid("empty packed payload tensor name"))?;
    if name.is_empty() || segments.is_empty() {
        return Err(invalid("packed payload tensor needs a file and a group"));
    }
    let mut path = segments.join("/");
    path.push('/');
    path.push_str(name);
    path.push_str(".safetensors");
    validate_relative_path(&path)?;
    Ok(path)
}

/// True when a candidate path is equal to, an ancestor of, or beneath an
/// allocated entry path. Ancestor checks use a set range and the candidate's
/// own path components, so large catalogs do not require quadratic scans.
pub fn path_conflicts(candidate: &str, used: &BTreeSet<String>) -> bool {
    if used.contains(candidate) {
        return true;
    }
    let mut prefix = candidate;
    while let Some((parent, _)) = prefix.rsplit_once('/') {
        if used.contains(parent) {
            return true;
        }
        prefix = parent;
    }
    let descendant = format!("{candidate}/");
    used.range(descendant.clone()..)
        .next()
        .is_some_and(|path| path.starts_with(&descendant))
}

/// Normalize one authenticated payload location to a deterministic companion
/// candidate. Directory companions keep their path; packed tensors use their
/// packed-to-path mapping when valid. Arbitrary packed tensor names normalize
/// under `packed/` without failing installation.
pub fn normalized_payload_companion(file: &str, tensor: &str, identity: PayloadId) -> String {
    let leaf = |tensor: &str| {
        let mut leaf: String = tensor
            .chars()
            .map(|byte| {
                if byte.is_ascii_alphanumeric() || matches!(byte, '.' | '_' | '-') {
                    byte
                } else {
                    '_'
                }
            })
            .collect();
        while leaf.starts_with('.') {
            leaf.remove(0);
        }
        while leaf.ends_with('.') {
            leaf.pop();
        }
        if leaf.is_empty() {
            leaf = format!("payload-{}", identity.to_hex());
        }
        leaf.truncate(96);
        leaf
    };
    if file.is_empty() {
        if let Ok(path) = companion_path(tensor) {
            return path;
        }
        return format!("packed/{}.safetensors", leaf(tensor));
    }
    if tensor == COMPANION_TENSOR && file.ends_with(".safetensors") {
        return file.to_owned();
    }
    if tensor == COMPANION_TENSOR {
        return format!("companions/{}", identity.file_name());
    }
    let (parent, name) = match file.rsplit_once('/') {
        Some((parent, name)) => (parent, name),
        None => ("", file),
    };
    let stem = name.strip_suffix(".safetensors").unwrap_or(name);
    let path = if parent.is_empty() {
        format!("{stem}/{}.safetensors", leaf(tensor))
    } else {
        format!("{parent}/{stem}/{}.safetensors", leaf(tensor))
    };
    if path != "model.safetensors" && validate_relative_path(&path).is_ok() {
        path
    } else {
        format!("companions/{}", identity.file_name())
    }
}

fn validate_format(format: &str) -> Result<()> {
    if format.is_empty()
        || format.len() > 64
        || !format
            .bytes()
            .all(|b| b.is_ascii_graphic() && b != b'"' && b != b'\\')
    {
        return Err(invalid("invalid container format identifier"));
    }
    Ok(())
}

pub fn validate_tensor_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 255
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        || name.starts_with('.')
        || name.ends_with('.')
    {
        return Err(invalid(format!("invalid tensor name {name:?}")));
    }
    Ok(())
}

/// Accepts package-relative paths without traversal or platform ambiguity.
pub fn validate_relative_path(path: &str) -> Result<()> {
    if path.is_empty()
        || path.len() > MAX_PATH
        || path.starts_with('/')
        || path.contains('\\')
        || path.contains('\0')
        || path.starts_with("./")
    {
        return Err(invalid(format!("invalid package path {path:?}")));
    }
    for segment in path.split('/') {
        if segment.is_empty()
            || segment == "."
            || segment == ".."
            || segment.len() > 255
            || !segment.bytes().all(|b| b.is_ascii_graphic())
        {
            return Err(invalid(format!("invalid package path {path:?}")));
        }
    }
    Ok(())
}

// Secure resolution: the returned handle, not a re-opened pathname, is what the
// caller reads. Unix walks components with openat(O_NOFOLLOW); other platforms
// open without following the final component and verify the opened handle's
// resolved location stays beneath the root.

#[cfg(unix)]
fn open_directory(path: &Path) -> Result<std::os::fd::OwnedFd> {
    use std::os::fd::{FromRawFd, OwnedFd};
    use std::os::unix::ffi::OsStrExt;

    let name = std::ffi::CString::new(path.as_os_str().as_bytes())
        .map_err(|_| invalid("package root contains a zero byte"))?;
    let raw = unsafe {
        libc::open(
            name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC,
        )
    };
    if raw < 0 {
        return Err(io("open package root", io::Error::last_os_error()));
    }
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

#[cfg(unix)]
fn open_relative_impl(root: &Path, relative: &str) -> Result<(PathBuf, Arc<File>)> {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

    let segments: Vec<&str> = relative.split('/').collect();
    let mut directory = open_directory(root)?;
    for (index, segment) in segments.iter().enumerate() {
        let name = std::ffi::CString::new(segment.as_bytes())
            .map_err(|_| invalid("package path contains a zero byte"))?;
        let intermediate = index + 1 < segments.len();
        let flags = libc::O_RDONLY
            | libc::O_NOFOLLOW
            | libc::O_CLOEXEC
            | if intermediate { libc::O_DIRECTORY } else { 0 };
        let raw = unsafe { libc::openat(directory.as_raw_fd(), name.as_ptr(), flags) };
        if raw < 0 {
            let context = if intermediate {
                "open package directory"
            } else {
                "open package payload"
            };
            return Err(io(context, io::Error::last_os_error()));
        }
        let opened = unsafe { OwnedFd::from_raw_fd(raw) };
        if intermediate {
            directory = opened;
        } else {
            return Ok((root.join(relative), Arc::new(File::from(opened))));
        }
    }
    Err(invalid("package path must name a file"))
}

#[cfg(not(unix))]
fn open_relative_impl(root: &Path, relative: &str) -> Result<(PathBuf, Arc<File>)> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x0020_0000); // FILE_FLAG_OPEN_REPARSE_POINT
    }
    let path = root.join(relative);
    let file = options
        .open(&path)
        .map_err(|e| io("open package payload", e))?;
    let canonical_root = comparable_root(root)?;
    let actual = opened_path(&file)?;
    if !actual.starts_with(&canonical_root) {
        return Err(invalid(format!(
            "package path escapes its root: {relative:?}"
        )));
    }
    if !file
        .metadata()
        .map_err(|e| io("read package payload metadata", e))?
        .is_file()
    {
        return Err(invalid(format!(
            "package path is not a regular file: {relative:?}"
        )));
    }
    Ok((path, Arc::new(file)))
}

#[cfg(not(unix))]
fn opened_path(file: &File) -> Result<PathBuf> {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::{
            FILE_NAME_NORMALIZED, GetFinalPathNameByHandleW, VOLUME_NAME_DOS,
        };

        let handle = file.as_raw_handle();
        let flags = VOLUME_NAME_DOS | FILE_NAME_NORMALIZED;
        let required = unsafe { GetFinalPathNameByHandleW(handle, std::ptr::null_mut(), 0, flags) };
        if required == 0 {
            return Err(io("resolve package handle", io::Error::last_os_error()));
        }
        let mut buffer = vec![0u16; required as usize];
        let written =
            unsafe { GetFinalPathNameByHandleW(handle, buffer.as_mut_ptr(), required, flags) };
        if written == 0 || written >= required {
            return Err(io("resolve package handle", io::Error::last_os_error()));
        }
        buffer.truncate(written as usize);
        let text = String::from_utf16(&buffer)
            .map_err(|_| invalid("package handle path is not UTF-16"))?;
        let text = text.strip_prefix(r"\\?\").unwrap_or(&text);
        return Ok(PathBuf::from(text));
    }
    #[cfg(not(windows))]
    {
        let _ = file;
        Err(invalid(
            "handle-confined package opens require a supported platform",
        ))
    }
}

/// The package root in the form `opened_path` reports.
///
/// Both paths come from the Windows handle resolver, which returns verbatim paths; `opened_path`
/// drops the verbatim prefix, so the root is compared in the same form.
#[cfg(not(unix))]
fn comparable_root(root: &Path) -> Result<PathBuf> {
    let canonical = fs::canonicalize(root).map_err(|e| io("resolve package root", e))?;
    #[cfg(windows)]
    {
        let text = canonical.to_string_lossy();
        if let Some(stripped) = text.strip_prefix(r"\\?\") {
            return Ok(PathBuf::from(stripped));
        }
    }
    Ok(canonical)
}

/// Opens one package file beneath `root`, confined to the opened handle.
pub fn open_relative(root: &Path, relative: &str) -> Result<(PathBuf, Arc<File>)> {
    validate_relative_path(relative)?;
    open_relative_impl(root, relative)
}

#[cfg(unix)]
fn create_relative(root: &Path, relative: &str) -> Result<(PathBuf, File)> {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

    let segments: Vec<&str> = relative.split('/').collect();
    let mut directory = open_directory(root)?;
    for (index, segment) in segments.iter().enumerate() {
        let name = std::ffi::CString::new(segment.as_bytes())
            .map_err(|_| invalid("package path contains a zero byte"))?;
        if index + 1 == segments.len() {
            let raw = unsafe {
                libc::openat(
                    directory.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_WRONLY
                        | libc::O_CREAT
                        | libc::O_EXCL
                        | libc::O_NOFOLLOW
                        | libc::O_CLOEXEC,
                    0o644,
                )
            };
            if raw < 0 {
                return Err(io("create package companion", io::Error::last_os_error()));
            }
            let opened = unsafe { OwnedFd::from_raw_fd(raw) };
            return Ok((root.join(relative), File::from(opened)));
        }
        let mut raw = unsafe {
            libc::openat(
                directory.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if raw < 0 && io::Error::last_os_error().raw_os_error() == Some(libc::ENOENT) {
            if unsafe { libc::mkdirat(directory.as_raw_fd(), name.as_ptr(), 0o755) } < 0 {
                let error = io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::EEXIST) {
                    return Err(io("create package directory", error));
                }
            }
            raw = unsafe {
                libc::openat(
                    directory.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                )
            };
        }
        if raw < 0 {
            return Err(io("open package directory", io::Error::last_os_error()));
        }
        directory = unsafe { OwnedFd::from_raw_fd(raw) };
    }
    Err(invalid("package path must name a file"))
}

#[cfg(not(unix))]
fn create_relative(root: &Path, relative: &str) -> Result<(PathBuf, File)> {
    let canonical_root = fs::canonicalize(root).map_err(|e| io("resolve package root", e))?;
    let segments: Vec<&str> = relative.split('/').collect();
    let mut current = canonical_root.clone();
    for (index, segment) in segments.iter().enumerate() {
        current.push(segment);
        if index + 1 == segments.len() {
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(windows)]
            {
                use std::os::windows::fs::OpenOptionsExt;
                options.custom_flags(0x0020_0000); // FILE_FLAG_OPEN_REPARSE_POINT
            }
            let file = options
                .open(&current)
                .map_err(|e| io("create package companion", e))?;
            return Ok((current, file));
        }
        match fs::symlink_metadata(&current) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    return Err(invalid(format!(
                        "package directory is not a plain directory: {segment:?}"
                    )));
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                fs::create_dir(&current).map_err(|e| io("create package directory", e))?;
            }
            Err(error) => return Err(io("inspect package directory", error)),
        }
        let resolved =
            fs::canonicalize(&current).map_err(|e| io("resolve package directory", e))?;
        if !resolved.starts_with(&canonical_root) {
            return Err(invalid(format!(
                "package path escapes its root: {relative:?}"
            )));
        }
    }
    Err(invalid("package path must name a file"))
}

/// Copies every primary tensor and companion payload into one packed container.
pub fn pack(
    output: impl Write,
    format: &str,
    primary: &Container,
    companions: &[(String, Slice)],
) -> Result<()> {
    let names: Vec<String> = primary.tensor_names().map(str::to_owned).collect();
    let mut writer = Writer::new(output, format)?;
    for name in names {
        let slice = primary.tensor(&name)?;
        writer.add(&name, slice.len(), Source::Slice(slice))?;
    }
    for (name, slice) in companions {
        writer.add(name, slice.len(), Source::Slice(slice.clone()))?;
    }
    writer.finish()
}

/// Write one payload slice as a canonical companion at `relative`.
pub fn write_payload_companion(root: &Path, relative: &str, payload: &Slice) -> Result<()> {
    validate_relative_path(relative)?;
    let (_, mut file) = create_relative(root, relative)?;
    write_companion(&mut file, payload)?;
    file.sync_all()
        .map_err(|e| io("sync package companion", e))?;
    Ok(())
}

/// Writes deterministic companion files extracted from a packed container.
///
/// `files` maps a packed tensor name to its package-relative companion path.
/// Existing directories are traversed without following symlinks, and an
/// existing destination file is never overwritten.
pub fn unpack(container: &Container, root: &Path, files: &[(String, String)]) -> Result<()> {
    if files.is_empty() {
        return Err(invalid("unpack requires at least one payload"));
    }
    let mut paths = BTreeSet::new();
    for (tensor, relative) in files {
        if tensor.starts_with(ALIGNMENT_PREFIX) {
            return Err(invalid("alignment tensors cannot be extracted"));
        }
        let slice = container.tensor(tensor)?;
        validate_relative_path(relative)?;
        if !paths.insert(relative.clone()) {
            return Err(invalid(format!("duplicate unpack path {relative:?}")));
        }
        let (_, mut file) = create_relative(root, relative)?;
        write_companion(&mut file, &slice)?;
        file.sync_all()
            .map_err(|e| io("sync package companion", e))?;
    }
    Ok(())
}

// Strict header decoding: duplicate keys, unknown fields and wrong shapes fail.

struct RawHeader {
    format: String,
    tensors: BTreeMap<String, RawTensor>,
}
struct RawTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: Vec<u64>,
}

const HEADER_FIELDS: &[&str] = &["dtype", "shape", "data_offsets"];

struct HeaderVisitor;
impl<'de> Visitor<'de> for HeaderVisitor {
    type Value = RawHeader;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a safetensors header object")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<RawHeader, A::Error> {
        let mut seen = BTreeSet::new();
        let mut format = None;
        let mut tensors = BTreeMap::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate header key {key:?}")));
            }
            if key == "__metadata__" {
                format = Some(map.next_value::<RawMetadata>()?.format);
                continue;
            }
            if tensors.len() >= MAX_TENSORS {
                return Err(de::Error::custom("container has too many tensors"));
            }
            validate_tensor_name(&key).map_err(de::Error::custom)?;
            tensors.insert(key, map.next_value::<RawTensor>()?);
        }
        let format = format.ok_or_else(|| de::Error::custom("header lacks __metadata__.format"))?;
        Ok(RawHeader { format, tensors })
    }
}

impl<'de> Deserialize<'de> for RawHeader {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_map(HeaderVisitor)
    }
}

struct RawMetadata {
    format: String,
}
struct MetadataVisitor;
impl<'de> Visitor<'de> for MetadataVisitor {
    type Value = RawMetadata;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("header metadata")
    }
    fn visit_map<A: MapAccess<'de>>(
        self,
        mut map: A,
    ) -> std::result::Result<RawMetadata, A::Error> {
        let mut seen = BTreeSet::new();
        let mut format = None;
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate metadata key {key:?}")));
            }
            if key != "format" {
                return Err(de::Error::unknown_field(&key, &["format"]));
            }
            format = Some(map.next_value::<String>()?);
        }
        let format = format.ok_or_else(|| de::Error::custom("metadata lacks format"))?;
        validate_format(&format).map_err(de::Error::custom)?;
        Ok(RawMetadata { format })
    }
}
impl<'de> Deserialize<'de> for RawMetadata {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_map(MetadataVisitor)
    }
}

struct TensorVisitor;
impl<'de> Visitor<'de> for TensorVisitor {
    type Value = RawTensor;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a safetensors tensor descriptor")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<RawTensor, A::Error> {
        let mut seen = BTreeSet::new();
        let mut dtype = None;
        let mut shape = None;
        let mut offsets = None;
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate tensor key {key:?}")));
            }
            match key.as_str() {
                "dtype" => dtype = Some(map.next_value::<String>()?),
                "shape" => shape = Some(map.next_value::<Vec<u64>>()?),
                "data_offsets" => offsets = Some(map.next_value::<Vec<u64>>()?),
                other => return Err(de::Error::unknown_field(other, HEADER_FIELDS)),
            }
        }
        Ok(RawTensor {
            dtype: dtype.ok_or_else(|| de::Error::custom("tensor lacks dtype"))?,
            shape: shape.ok_or_else(|| de::Error::custom("tensor lacks shape"))?,
            data_offsets: offsets.ok_or_else(|| de::Error::custom("tensor lacks data_offsets"))?,
        })
    }
}
impl<'de> Deserialize<'de> for RawTensor {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_map(TensorVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    struct Scratch(PathBuf);
    impl Scratch {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "flare-container-{}-{}-{tag}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn join(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write_raw(path: &Path, header: &str, data: &[u8]) {
        let mut header = header.as_bytes().to_vec();
        while header.len() % ALIGNMENT as usize != ALIGNMENT as usize - 8 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(data);
        fs::write(path, bytes).unwrap();
    }

    fn container_with(scratch: &Scratch, name: &str, payload: &[u8]) -> (PathBuf, Container) {
        let path = scratch.join(name);
        let mut writer = Writer::new(File::create(&path).unwrap(), PAYLOAD_FORMAT).unwrap();
        writer
            .add(
                COMPANION_TENSOR,
                payload.len() as u64,
                Source::Bytes(payload.to_vec()),
            )
            .unwrap();
        writer.finish().unwrap();
        let container = Container::open(&path).unwrap();
        (path, container)
    }

    #[test]
    fn round_trip_preserves_payload_bytes_and_alignment() {
        let scratch = Scratch::new("round-trip");
        let payload: Vec<u8> = (0..9000u32).map(|value| value as u8).collect();
        let (_, container) = container_with(&scratch, "payload.safetensors", &payload);
        assert_eq!(container.format(), PAYLOAD_FORMAT);
        let slice = container.tensor(COMPANION_TENSOR).unwrap();
        assert_eq!(slice.len(), payload.len() as u64);
        assert_eq!(slice.offset() % ALIGNMENT, 0);
        let mut bytes = vec![0u8; payload.len()];
        slice.read_at(0, &mut bytes).unwrap();
        assert_eq!(bytes, payload);
        let mut again = Vec::new();
        slice.copy_to(&mut again).unwrap();
        assert_eq!(again, payload);
        assert_eq!(container.payload_names().count(), 1);
    }

    #[test]
    fn companion_wrapper_is_deterministic_and_digest_excludes_it() {
        let scratch = Scratch::new("deterministic");
        let payload: Vec<u8> = (0..1234u32).map(|value| (value * 7) as u8).collect();
        let (_, container) = container_with(&scratch, "payload.safetensors", &payload);
        let slice = container.tensor(COMPANION_TENSOR).unwrap();

        let mut first = Vec::new();
        write_companion(&mut first, &slice).unwrap();
        let mut second = Vec::new();
        write_companion(&mut second, &slice).unwrap();
        assert_eq!(first, second);

        let path = scratch.join("wrapper.safetensors");
        fs::write(&path, &first).unwrap();
        let wrapped = Container::open(&path).unwrap();
        let mut bytes = vec![0u8; payload.len()];
        wrapped
            .tensor(COMPANION_TENSOR)
            .unwrap()
            .read_at(0, &mut bytes)
            .unwrap();
        assert_eq!(bytes, payload);

        assert_eq!(
            slice.digest().unwrap().as_bytes(),
            digest::digest(&digest::SHA256, &payload).as_ref()
        );
    }

    #[test]
    fn duplicate_header_keys_are_rejected() {
        let scratch = Scratch::new("duplicate");
        let path = scratch.join("duplicate.safetensors");
        let header = r#"{"__metadata__":{"format":"flare-payload"},"weights.a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"weights.a":{"dtype":"U8","shape":[4],"data_offsets":[4096,4100]}}"#;
        write_raw(&path, header, &[1u8; 4096]);
        let error = Container::open(&path).unwrap_err().to_string();
        assert!(error.contains("duplicate"), "{error}");
    }

    #[test]
    fn overlapping_or_incomplete_coverage_is_rejected() {
        let scratch = Scratch::new("coverage");
        for (tag, header) in [
            (
                "overlap",
                r#"{"__metadata__":{"format":"flare-payload"},"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]},"b":{"dtype":"U8","shape":[8],"data_offsets":[4,12]}}"#,
            ),
            (
                "gap",
                r#"{"__metadata__":{"format":"flare-payload"},"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[8,12]}}"#,
            ),
            (
                "short",
                r#"{"__metadata__":{"format":"flare-payload"},"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#,
            ),
        ] {
            let path = scratch.join(&format!("{tag}.safetensors"));
            write_raw(&path, header, &[0u8; 8192]);
            assert!(Container::open(&path).is_err(), "{tag}");
        }
    }

    #[test]
    fn wrong_dtype_shape_and_offsets_are_rejected() {
        let scratch = Scratch::new("shape");
        for (tag, header) in [
            (
                "dtype",
                r#"{"__metadata__":{"format":"flare-payload"},"a":{"dtype":"F32","shape":[4],"data_offsets":[0,4]}}"#,
            ),
            (
                "shape",
                r#"{"__metadata__":{"format":"flare-payload"},"a":{"dtype":"U8","shape":[2,2],"data_offsets":[0,4]}}"#,
            ),
            (
                "offsets",
                r#"{"__metadata__":{"format":"flare-payload"},"a":{"dtype":"U8","shape":[4],"data_offsets":[0]}}"#,
            ),
        ] {
            let path = scratch.join(&format!("{tag}.safetensors"));
            write_raw(&path, header, &[0u8; 8192]);
            assert!(Container::open(&path).is_err(), "{tag}");
        }
    }

    #[test]
    fn unaligned_header_length_is_rejected() {
        let scratch = Scratch::new("alignment");
        let path = scratch.join("unaligned.safetensors");
        let header = r#"{"__metadata__":{"format":"flare-payload"},"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[1, 2, 3, 4]);
        fs::write(&path, bytes).unwrap();
        assert!(Container::open(&path).is_err());
    }

    #[test]
    fn pack_and_unpack_copy_bytes_without_a_key() {
        let scratch = Scratch::new("pack");
        let manifest = b"exact-signed-envelope-bytes".to_vec();
        let payload: Vec<u8> = (0..5000u32).map(|value| (value * 13) as u8).collect();
        let primary_path = scratch.join("model.safetensors");
        let mut primary =
            Writer::new(File::create(&primary_path).unwrap(), "flare-package").unwrap();
        primary
            .add(
                "flare.manifest",
                manifest.len() as u64,
                Source::Bytes(manifest.clone()),
            )
            .unwrap();
        primary.finish().unwrap();
        let primary = Container::open(&primary_path).unwrap();
        let (_, companion) = container_with(&scratch, "companion.safetensors", &payload);
        let payload_slice = companion.tensor(COMPANION_TENSOR).unwrap();

        let packed_path = scratch.join("packed.safetensors");
        pack(
            File::create(&packed_path).unwrap(),
            primary.format(),
            &primary,
            &[("weights.layer-000".into(), payload_slice.clone())],
        )
        .unwrap();
        let packed = Container::open(&packed_path).unwrap();
        let names: Vec<&str> = packed.payload_names().collect();
        assert_eq!(names, ["flare.manifest", "weights.layer-000"]);
        let mut manifest_out = vec![0u8; manifest.len()];
        packed
            .tensor("flare.manifest")
            .unwrap()
            .read_at(0, &mut manifest_out)
            .unwrap();
        assert_eq!(manifest_out, manifest);

        let unpacked = scratch.join("unpacked");
        fs::create_dir_all(&unpacked).unwrap();
        unpack(
            &packed,
            &unpacked,
            &[(
                "weights.layer-000".into(),
                "weights/layer-000.safetensors".into(),
            )],
        )
        .unwrap();
        let companion = Container::open(&unpacked.join("weights/layer-000.safetensors")).unwrap();
        let mut bytes = vec![0u8; payload.len()];
        companion
            .tensor(COMPANION_TENSOR)
            .unwrap()
            .read_at(0, &mut bytes)
            .unwrap();
        assert_eq!(bytes, payload);
        assert_eq!(
            packed
                .tensor("weights.layer-000")
                .unwrap()
                .digest()
                .unwrap()
                .as_bytes(),
            payload_slice.digest().unwrap().as_bytes()
        );
    }

    #[test]
    fn unpack_refuses_existing_destinations_and_symlinked_directories() {
        let scratch = Scratch::new("unpack-guard");
        let payload = b"payload-bytes".to_vec();
        let packed_path = scratch.join("packed.safetensors");
        let mut writer = Writer::new(File::create(&packed_path).unwrap(), PAYLOAD_FORMAT).unwrap();
        writer
            .add(
                "weights.layer-000",
                payload.len() as u64,
                Source::Bytes(payload),
            )
            .unwrap();
        writer.finish().unwrap();
        let packed = Container::open(&packed_path).unwrap();

        let root = scratch.join("root");
        fs::create_dir_all(&root).unwrap();
        unpack(
            &packed,
            &root,
            &[(
                "weights.layer-000".into(),
                "weights/layer-000.safetensors".into(),
            )],
        )
        .unwrap();
        assert!(
            unpack(
                &packed,
                &root,
                &[(
                    "weights.layer-000".into(),
                    "weights/layer-000.safetensors".into()
                )]
            )
            .is_err()
        );

        #[cfg(unix)]
        {
            let outside = scratch.join("outside");
            fs::create_dir_all(&outside).unwrap();
            let root = scratch.join("linked-root");
            fs::create_dir_all(&root).unwrap();
            std::os::unix::fs::symlink(&outside, root.join("weights")).unwrap();
            assert!(
                unpack(
                    &packed,
                    &root,
                    &[(
                        "weights.layer-000".into(),
                        "weights/layer-000.safetensors".into()
                    )]
                )
                .is_err()
            );
            assert!(!outside.join("layer-000.safetensors").exists());
        }
    }

    #[test]
    fn packed_tensor_names_round_trip() {
        assert_eq!(
            packed_tensor_name("deployments/sm120-tp2/rank-0/graphs.safetensors").unwrap(),
            "deployments.sm120-tp2.rank-0.graphs"
        );
        assert_eq!(
            companion_path("weights.layer-000").unwrap(),
            "weights/layer-000.safetensors"
        );
        assert_eq!(
            companion_path(&packed_tensor_name("weights/layer-000.safetensors").unwrap()).unwrap(),
            "weights/layer-000.safetensors"
        );
        assert!(packed_tensor_name("weights/../secret.safetensors").is_err());
    }

    #[test]
    fn path_validation_rejects_traversal_and_ambiguity() {
        for path in [
            "",
            "/abs/x.safetensors",
            "../x.safetensors",
            "a/../b.safetensors",
            "a//b",
            "./a",
            "a\\b",
        ] {
            assert!(validate_relative_path(path).is_err(), "{path:?}");
        }
        assert!(validate_relative_path("weights/layer-000.safetensors").is_ok());
        for name in ["", ".hidden", "name..", "a/b", "with space"] {
            assert!(validate_tensor_name(name).is_err(), "{name:?}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_is_rejected_without_reopening_a_path() {
        let scratch = Scratch::new("symlink");
        let outside = scratch.join("outside.safetensors");
        fs::write(&outside, b"x").unwrap();
        let root = scratch.join("root");
        fs::create_dir_all(&root).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("escape.safetensors")).unwrap();
        let error = open_relative(&root, "escape.safetensors")
            .unwrap_err()
            .to_string();
        assert!(error.contains("open package payload"), "{error}");

        let outside_dir = scratch.join("outside-dir");
        fs::create_dir_all(&outside_dir).unwrap();
        fs::write(outside_dir.join("payload.safetensors"), b"y").unwrap();
        std::os::unix::fs::symlink(&outside_dir, root.join("linked")).unwrap();
        assert!(open_relative(&root, "linked/payload.safetensors").is_err());
    }

    #[test]
    fn writer_rejects_duplicate_and_empty_tensors() {
        let mut writer = Writer::new(Vec::new(), PAYLOAD_FORMAT).unwrap();
        writer.add("a", 1, Source::Bytes(vec![0])).unwrap();
        assert!(writer.add("a", 1, Source::Bytes(vec![0])).is_err());
        assert!(writer.add("b", 0, Source::Bytes(Vec::new())).is_err());
        assert!(writer.add("c", 2, Source::Bytes(vec![0])).is_err());
        assert!(writer.add("align.0", 1, Source::Bytes(vec![0])).is_err());
    }

    #[test]
    fn packed_writer_inserts_aligned_zero_padding() {
        let scratch = Scratch::new("padding");
        let mut bytes = Vec::new();
        {
            let mut writer = Writer::new(&mut bytes, PAYLOAD_FORMAT).unwrap();
            writer.add("a", 100, Source::Bytes(vec![1; 100])).unwrap();
            writer.add("b", 100, Source::Bytes(vec![2; 100])).unwrap();
            writer.finish().unwrap();
        }
        let path = scratch.join("packed.safetensors");
        fs::write(&path, &bytes).unwrap();
        let container = Container::open(&path).unwrap();
        let a = container.tensor("a").unwrap();
        let b = container.tensor("b").unwrap();
        assert_eq!(a.offset() % ALIGNMENT, 0);
        assert_eq!(b.offset() % ALIGNMENT, 0);
        assert!(b.offset() >= a.offset() + a.len());
        assert!(
            container
                .tensor_names()
                .any(|name| name.starts_with(ALIGNMENT_PREFIX))
        );
    }
}
