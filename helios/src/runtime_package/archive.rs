//! Bounded protobuf indexes and resource reads inside a safetensors container.

use super::{
    format::{Package, invalid},
    protection::{Protection, Publisher, ResourceReader},
    protobuf::wire,
};
use crate::Result;
use crate::os::FileReadAt;
use prost::Message;
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read, Take},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

const MAX_HEADER: usize = 64 << 10;
const MAX_INDEX: usize = 128 << 20;
const MAX_RECORD: usize = 1 << 20;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    #[serde(rename = "__metadata__")]
    format: Format,
    #[serde(rename = "flare.payload")]
    payload: ByteTensor,
    #[serde(rename = "flare.metadata")]
    index: ByteTensor,
    #[serde(rename = "flare.layout", default)]
    layout: Option<ByteTensor>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Layout {
    version: u32,
    files: BTreeMap<String, Vec<(u64, String)>>,
}

impl Layout {
    fn validate(&self) -> Result<()> {
        if self.version != 1 || !self.files.contains_key("") || self.files.len() > 100 {
            return Err(invalid("invalid snapshot layout"));
        }
        let mut count = 0;
        let mut lengths = BTreeMap::new();
        for (name, parts) in &self.files {
            if !name.is_empty() {
                validate_filename(name)?;
            }
            let mut total = 0u64;
            for (size, blob) in parts {
                count += 1;
                if *size == 0
                    || *size > (64u64 << 30) - 4096
                    || count > 65536
                    || (!blob.is_empty()
                        && (blob.len() != 76
                            || !blob.ends_with(".safetensors")
                            || !blob.as_bytes()[..64]
                                .iter()
                                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c))))
                {
                    return Err(invalid("invalid snapshot blob"));
                }
                total = total
                    .checked_add(*size)
                    .ok_or_else(|| invalid("snapshot length overflow"))?;
                if !blob.is_empty()
                    && lengths
                        .insert(blob, size)
                        .is_some_and(|previous| previous != size)
                {
                    return Err(invalid("conflicting snapshot blob lengths"));
                }
            }
        }
        Ok(())
    }

    fn segments(&self, span: &wire::Span) -> Result<Vec<(Option<String>, u64, u64)>> {
        let parts = self
            .files
            .get(&span.filename)
            .ok_or_else(|| invalid("missing logical package file"))?;
        let end = span
            .offset
            .checked_add(span.nbytes)
            .ok_or_else(|| invalid("resource overflow"))?;
        let mut cursor = 0u64;
        let mut result = Vec::new();
        for (size, name) in parts {
            let stop = cursor + size;
            if cursor < end && span.offset < stop {
                let start = cursor.max(span.offset);
                result.push((
                    (!name.is_empty()).then_some(name.clone()),
                    start - cursor,
                    end.min(stop) - start,
                ));
            }
            cursor = stop;
        }
        if end > cursor {
            return Err(invalid("resource exceeds logical payload"));
        }
        Ok(result)
    }

    fn read(file: &File) -> Result<Option<Self>> {
        let mut prefix = [0; 8];
        file.read_exact_at(&mut prefix, 0)
            .map_err(|e| e.to_string())?;
        let size = u64::from_le_bytes(prefix);
        if size == 0 || size > MAX_HEADER as u64 {
            return Err(invalid("invalid package header"));
        }
        let mut bytes = vec![0; size as usize];
        file.read_exact_at(&mut bytes, 8)
            .map_err(|e| e.to_string())?;
        let header: Header = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        if header.format.format != "flare-snapshot" {
            if header.layout.is_some() {
                return Err(invalid("unexpected snapshot layout"));
            }
            return Ok(None);
        }
        let tensor = header
            .layout
            .ok_or_else(|| invalid("snapshot lacks storage layout"))?;
        let count = tensor.size()?;
        if header.payload.size()? != 0
            || tensor.data_offsets[0] != header.index.data_offsets[1]
            || count == 0
            || count > MAX_INDEX as u64
        {
            return Err(invalid("invalid snapshot layout tensor"));
        }
        let mut bytes = vec![0; count as usize];
        file.read_exact_at(&mut bytes, size + 8 + tensor.data_offsets[0])
            .map_err(|e| e.to_string())?;
        let layout: Self = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        layout.validate()?;
        Ok(Some(layout))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Format {
    format: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ByteTensor {
    dtype: String,
    shape: [u64; 1],
    data_offsets: [u64; 2],
}
impl ByteTensor {
    fn size(&self) -> Result<u64> {
        let [begin, end] = self.data_offsets;
        if self.dtype != "U8" || end.checked_sub(begin) != Some(self.shape[0]) {
            return Err(invalid("invalid package byte tensor"));
        }
        Ok(self.shape[0])
    }
}

/// One package file retained while a peer receives selected byte ranges.
#[derive(Clone)]
pub struct PackageFile {
    /// Empty for the primary; companions retain their signed sibling basenames.
    pub filename: String,
    pub file: Arc<File>,
    pub ranges: Vec<(u64, u64)>,
}

fn merge_ranges(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.retain(|&(_, nbytes)| nbytes > 0);
    ranges.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (offset, nbytes) in ranges {
        match merged.last_mut() {
            Some((first, extent)) if offset <= *first + *extent => {
                *extent = (*extent).max(offset + nbytes - *first);
            }
            _ => merged.push((offset, nbytes)),
        }
    }
    merged
}

pub(super) fn validate_filename(name: &str) -> Result<()> {
    let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
    if name.len() > 255
        || !name.ends_with(".safetensors")
        || name.contains("..")
        || !name
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        || matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'))
    {
        return Err(invalid("invalid package companion filename"));
    }
    Ok(())
}

pub(super) struct PayloadFile {
    pub file: Arc<File>,
    pub start: u64,
    pub length: u64,
    direct: Arc<Mutex<Option<Arc<File>>>>,
}
impl PayloadFile {
    pub(super) fn new(file: Arc<File>, start: u64, length: u64) -> Self {
        Self::new_shared(file, start, length, Arc::new(Mutex::new(None)))
    }
    /// Share one cached direct-I/O handle with other payloads in the same container. Payloads that
    /// live in one packed file must not each reopen it.
    pub(super) fn new_shared(
        file: Arc<File>,
        start: u64,
        length: u64,
        direct: Arc<Mutex<Option<Arc<File>>>>,
    ) -> Self {
        Self {
            file,
            start,
            length,
            direct,
        }
    }
    pub fn direct(&self) -> Result<Arc<File>> {
        let mut cached = self.direct.lock().unwrap();
        if cached.is_none() {
            *cached = Some(Arc::new(crate::os::direct_reopen(&self.file).map_err(
                |e| invalid(format!("open package payload for direct I/O: {e}")),
            )?));
        }
        Ok(cached.as_ref().unwrap().clone())
    }
    fn companion(file: File) -> Result<Self> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct DataHeader {
            #[serde(rename = "__metadata__")]
            format: Format,
            #[serde(rename = "flare.payload")]
            payload: ByteTensor,
        }
        let info = file.metadata().map_err(|e| e.to_string())?;
        if !info.is_file() {
            return Err(invalid("package companion must be a regular file"));
        }
        let mut prefix = [0; 8];
        file.read_exact_at(&mut prefix, 0)
            .map_err(|e| e.to_string())?;
        let size = u64::from_le_bytes(prefix);
        if size == 0 || size > MAX_HEADER as u64 || (size + 8) % 4096 != 0 {
            return Err(invalid("invalid companion header size"));
        }
        let mut bytes = vec![0; size as usize];
        file.read_exact_at(&mut bytes, 8)
            .map_err(|e| e.to_string())?;
        if bytes[0] != b'{' {
            return Err(invalid("invalid companion header"));
        }
        let header: DataHeader =
            serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        let length = header.payload.size()?;
        if header.format.format != "flare-data"
            || header.payload.data_offsets[0] != 0
            || info.len().checked_sub(size + 8) != Some(length)
        {
            return Err(invalid("invalid companion payload length or format"));
        }
        Ok(PayloadFile::new(Arc::new(file), size + 8, length))
    }
}

/// Inspect bounded metadata for sibling files before authentication, without opening any sibling.
/// Callers may fetch these names from the primary's pinned source; signatures still require verification.
pub fn package_companions(path: &Path) -> Result<Vec<String>> {
    // A distributed directory package names its companions through signed location
    // records; only a packed snapshot carries them in the archive layout.
    let primary = if path.is_dir() {
        path.join("model.safetensors")
    } else {
        path.to_owned()
    };
    if crate::package::reader::declared_format(&primary)?
        == Some(crate::package::container::PAYLOAD_FORMAT.to_owned())
    {
        let metadata = crate::package::reader::Metadata::open(&primary)?;
        let names: std::collections::BTreeSet<String> = metadata
            .locations
            .locations
            .iter()
            .filter(|location| !location.file.is_empty())
            .map(|location| location.file.clone())
            .collect();
        return Ok(names.into_iter().collect());
    }
    let (file, _, _, bytes) = Archive::envelope(path)?;
    if let Some(layout) = Layout::read(&file)? {
        return Ok(layout
            .files
            .values()
            .flatten()
            .filter_map(|(_, name)| (!name.is_empty()).then_some(name.clone()))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect());
    }
    let envelope =
        wire::ProtectedIndex::decode(bytes.as_slice()).map_err(|e| invalid(e.to_string()))?;
    if envelope.version != 6
        || envelope.groups.len() != 3
        || envelope.groups[0].kind != 1
        || envelope.ranks.is_empty()
    {
        return Err(invalid("invalid protected package envelope"));
    }
    let metadata = wire::PublisherMetadata::decode(envelope.groups[0].body.as_slice())
        .map_err(|e| invalid(e.to_string()))?;
    let mut index = metadata
        .index
        .ok_or_else(|| invalid("missing model index"))?;
    if !matches!(index.version, 4 | 5) {
        return Err(invalid("unsupported model index"));
    }
    let mut parts = wire::StoredData::decode(envelope.groups[2].body.as_slice())
        .map_err(|e| invalid(e.to_string()))?
        .parts;
    for group in envelope.ranks {
        let rank = wire::RankProtection::decode(group.body.as_slice())
            .map_err(|e| invalid(e.to_string()))?;
        index
            .ranks
            .push(rank.bundle.ok_or_else(|| invalid("missing rank bundle"))?);
        parts.extend(rank.data.unwrap_or_default().parts);
    }
    let mut names = std::collections::BTreeSet::new();
    for span in index
        .ranks
        .iter()
        .flat_map(|rank| rank.resources.values())
        .chain(
            index
                .assets
                .iter()
                .filter_map(|asset| asset.payload.as_ref()),
        )
        .chain(
            super::engram::spans(&index, None)?
                .into_iter()
                .map(|(_, span)| span),
        )
        .chain(&parts)
    {
        if !span.filename.is_empty() {
            validate_filename(&span.filename)?;
            if index.version != 5 {
                return Err(invalid("companion references require model index 5"));
            }
            names.insert(span.filename.clone());
            if names.len() > 99 {
                return Err(invalid("package references more than 99 data files"));
            }
        }
    }
    Ok(names.into_iter().collect())
}

pub(super) struct Archive {
    pub path: PathBuf,
    pub start: u64,
    /// Payload bytes after `start`; the signed index follows them to the end of the file.
    pub payload: u64,
    pub file: Arc<File>,
    pub files: BTreeMap<String, PayloadFile>,
    layout: Option<Layout>,
    _directory: File,
    /// SHA-256 of the exact signed envelope bytes read at open.
    pub(super) envelope_digest: [u8; 32],
    pub(super) protection: Protection,
}
impl Archive {
    /// SHA-256 of the exact signed envelope bytes authenticated for this copy.
    pub(super) fn envelope_digest(&self) -> [u8; 32] {
        self.envelope_digest
    }

    /// Open the file and read its signed envelope: the file, the payload start and size, and the
    /// envelope bytes, none of them verified yet.
    pub(super) fn envelope(path: &Path) -> Result<(File, u64, u64, Vec<u8>)> {
        #[cfg(unix)]
        let file = File::open(path).map_err(|e| e.to_string())?;
        #[cfg(windows)]
        let file = {
            use std::os::windows::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .read(true)
                .share_mode(windows_sys::Win32::Storage::FileSystem::FILE_SHARE_READ)
                .open(path)
                .map_err(|e| e.to_string())?
        };
        let mut prefix = [0u8; 8];
        file.read_exact_at(&mut prefix, 0)
            .map_err(|e| e.to_string())?;
        let header_size = u64::from_le_bytes(prefix);
        if header_size == 0 || header_size > MAX_HEADER as u64 {
            return Err(invalid("invalid safetensors header size"));
        }
        let start = 8 + header_size;
        let mut header = vec![0; header_size as usize];
        file.read_exact_at(&mut header, 8)
            .map_err(|e| e.to_string())?;
        if header[0] != b'{' || start % 4096 != 0 {
            return Err(invalid("invalid or unaligned safetensors header"));
        }
        let header: Header = serde_json::from_slice(&header)
            .map_err(|e| invalid(format!("invalid safetensors header: {e}")))?;
        if !matches!(header.format.format.as_str(), "flare" | "flare-snapshot") {
            return Err(invalid("not a Flare safetensors package"));
        }
        let payload = header.payload.size()?;
        let size = header.index.size()?;
        if size == 0 || size > MAX_INDEX as u64 {
            return Err(invalid("invalid model index size"));
        }
        if header.payload.data_offsets[0] != 0 || header.index.data_offsets[0] != payload {
            return Err(invalid(
                "package tensors must cover the data buffer without gaps or overlap",
            ));
        }
        let length = file.metadata().map_err(|e| e.to_string())?.len();
        let end = header
            .layout
            .as_ref()
            .map_or(header.index.data_offsets[1], |l| l.data_offsets[1]);
        if length.checked_sub(start) != Some(end) {
            return Err(invalid(
                "safetensors data length differs from model package",
            ));
        }
        let mut bytes = vec![0; size as usize];
        file.read_exact_at(&mut bytes, start + payload)
            .map_err(|e| e.to_string())?;
        let logical_payload =
            Layout::read(&file)?.map_or(payload, |l| l.files[""].iter().map(|p| p.0).sum());
        Ok((file, start, logical_payload, bytes))
    }
    pub fn open(
        path: &Path,
        publishers: &[Publisher],
        rank: Option<usize>,
    ) -> Result<(Self, wire::ModelIndex)> {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        #[cfg(unix)]
        let directory = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_PATH | libc::O_DIRECTORY)
                .open(parent)
                .map_err(|e| e.to_string())?
        };
        #[cfg(windows)]
        let directory = {
            use std::os::windows::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .read(true)
                .share_mode(windows_sys::Win32::Storage::FileSystem::FILE_SHARE_READ)
                .custom_flags(windows_sys::Win32::Storage::FileSystem::FILE_FLAG_BACKUP_SEMANTICS)
                .open(parent)
                .map_err(|e| e.to_string())?
        };
        #[cfg(unix)]
        let primary = {
            use std::os::fd::AsRawFd;
            PathBuf::from(format!("/proc/self/fd/{}", directory.as_raw_fd())).join(
                path.file_name()
                    .ok_or_else(|| invalid("missing primary filename"))?,
            )
        };
        #[cfg(windows)]
        let primary = path.to_owned();
        let (file, start, payload, bytes) = Self::envelope(&primary)?;
        let envelope_digest: [u8; 32] = super::protection::sha256(&bytes)
            .try_into()
            .map_err(|_| invalid("signed envelope digest has the wrong length"))?;
        let (protection, index) = Protection::verify(&bytes, publishers, payload, rank)?;
        let layout = Layout::read(&file)?;
        let file = Arc::new(file);
        let mut files = BTreeMap::new();
        files.insert(
            String::new(),
            PayloadFile::new(
                file.clone(),
                start,
                if layout.is_some() { 0 } else { payload },
            ),
        );
        let mut names = std::collections::BTreeSet::new();
        let spans: Vec<_> = protection
            .data_parts()
            .into_iter()
            .chain(index.ranks.iter().flat_map(|r| r.resources.values()))
            .chain(index.assets.iter().filter_map(|a| a.payload.as_ref()))
            .chain(
                super::engram::spans(&index, rank)?
                    .into_iter()
                    .map(|(_, s)| s),
            )
            .collect();
        for span in &spans {
            if let Some(layout) = &layout {
                for (name, _, _) in layout.segments(span)? {
                    if let Some(name) = name {
                        names.insert(name);
                    }
                }
            } else if !span.filename.is_empty() {
                names.insert(span.filename.clone());
            }
        }
        for name in names {
            validate_filename(&name)?;
            if files.len() > if layout.is_some() { 65536 } else { 99 } {
                return Err(invalid("package references too many data files"));
            }
            #[cfg(unix)]
            let companion = {
                use std::os::fd::{AsRawFd, FromRawFd};
                let name = std::ffi::CString::new(name.as_str()).unwrap();
                // SAFETY: directory and name remain live; the returned descriptor has one owner.
                let fd = unsafe {
                    libc::openat(
                        directory.as_raw_fd(),
                        name.as_ptr(),
                        libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NONBLOCK,
                    )
                };
                if fd < 0 {
                    return Err(invalid(format!(
                        "open companion {:?}: {}",
                        name,
                        std::io::Error::last_os_error()
                    )));
                }
                unsafe { File::from_raw_fd(fd) }
            };
            #[cfg(windows)]
            let companion = {
                use std::os::windows::fs::OpenOptionsExt;
                use windows_sys::Win32::Storage::FileSystem::FILE_SHARE_READ;
                std::fs::OpenOptions::new()
                    .read(true)
                    .share_mode(FILE_SHARE_READ)
                    .open(parent.join(&name))
                    .map_err(|e| e.to_string())?
            };
            let companion = PayloadFile::companion(companion)?;
            if let Some(layout) = &layout {
                let size = layout
                    .files
                    .values()
                    .flatten()
                    .find(|(_, blob)| blob == &name)
                    .map(|(size, _)| *size)
                    .ok_or_else(|| invalid("unlisted snapshot blob"))?;
                if companion.length != size || companion.start != 4096 {
                    return Err(invalid("snapshot blob length or alignment mismatch"));
                }
            }
            files.insert(name, companion);
        }
        let result = Self {
            #[cfg(unix)]
            path: {
                use std::os::fd::AsRawFd;
                PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()))
            },
            #[cfg(windows)]
            path: path.to_owned(),
            start,
            payload,
            file,
            files,
            layout,
            _directory: directory,
            envelope_digest,
            protection,
        };
        for part in result.protection.data_parts() {
            result.segments(part)?;
        }
        let spans = index
            .ranks
            .iter()
            .flat_map(|rank| rank.resources.values())
            .chain(
                index
                    .assets
                    .iter()
                    .filter_map(|asset| asset.payload.as_ref()),
            )
            .chain(
                super::engram::spans(&index, rank)?
                    .into_iter()
                    .map(|(_, span)| span),
            );
        let mut ends = BTreeMap::new();
        for span in spans {
            result.source(span)?;
            let end = ends.entry(&span.filename).or_insert(0);
            *end = (*end).max(span.offset + span.nbytes);
        }
        if rank.is_none()
            && ends
                .iter()
                .any(|(name, end)| result.logical_length(name) != Some(*end))
        {
            return Err(invalid("unreferenced package payload bytes"));
        }
        Ok((result, index))
    }
    fn logical_length(&self, name: &str) -> Option<u64> {
        match &self.layout {
            Some(layout) => layout
                .files
                .get(name)
                .map(|parts| parts.iter().map(|p| p.0).sum()),
            None => self.files.get(name).map(|f| f.length),
        }
    }
    fn segments(&self, span: &wire::Span) -> Result<Vec<(Option<String>, u64, u64)>> {
        let parts = if let Some(layout) = &self.layout {
            layout.segments(span)?
        } else {
            vec![(Some(span.filename.clone()), span.offset, span.nbytes)]
        };
        for (name, offset, size) in &parts {
            if let Some(name) = name {
                let file = self
                    .files
                    .get(name)
                    .ok_or_else(|| invalid("missing package blob"))?;
                if offset
                    .checked_add(*size)
                    .is_none_or(|end| end > file.length)
                {
                    return Err(invalid("resource exceeds package payload"));
                }
            }
        }
        Ok(parts)
    }
    pub fn source(&self, span: &wire::Span) -> Result<(&PayloadFile, u64)> {
        let parts = self.segments(span)?;
        if parts.len() != 1 || parts[0].0.is_none() {
            return Err(invalid("resource must occupy one contiguous payload blob"));
        }
        let (name, offset, _) = &parts[0];
        let file = &self.files[name.as_ref().unwrap()];
        Ok((file, file.start + offset))
    }
    #[cfg(windows)]
    pub fn source_name(&self, span: &wire::Span) -> Result<String> {
        let parts = self.segments(span)?;
        if parts.len() != 1 {
            return Err(invalid("resource crosses payload blobs"));
        }
        parts[0]
            .0
            .clone()
            .ok_or_else(|| invalid("resource names padding"))
    }
    pub fn source_index(&self, span: &wire::Span) -> Result<usize> {
        let parts = self.segments(span)?;
        if parts.len() != 1 || parts[0].0.is_none() {
            return Err(invalid("resource must occupy one contiguous payload blob"));
        }
        self.file_index(parts[0].0.as_ref().unwrap())
    }
    pub fn read_logical(&self, span: &wire::Span, offset: u64, output: &mut [u8]) -> Result<()> {
        if offset
            .checked_add(output.len() as u64)
            .is_none_or(|n| n > span.nbytes)
        {
            return Err(invalid("logical read exceeds resource"));
        }
        let slice = wire::Span {
            filename: span.filename.clone(),
            offset: span.offset + offset,
            nbytes: output.len() as u64,
        };
        let mut cursor = 0;
        for (name, offset, size) in self.segments(&slice)? {
            let target = &mut output[cursor..cursor + size as usize];
            if let Some(name) = name {
                let file = &self.files[&name];
                file.file
                    .read_exact_at(target, file.start + offset)
                    .map_err(|e| e.to_string())?;
            } else {
                target.fill(0);
            }
            cursor += size as usize;
        }
        Ok(())
    }
    pub fn file_index(&self, filename: &str) -> Result<usize> {
        self.files
            .keys()
            .position(|name| name == filename)
            .ok_or_else(|| invalid("unknown package file"))
    }

    /// Per-file ranges that a rank needs, including signed stored-data padding.
    pub fn rank_ranges(&self, index: &wire::ModelIndex, rank: usize) -> Result<Vec<PackageFile>> {
        let position = self.protection.position(rank)?;
        let bundle = index
            .ranks
            .get(position)
            .ok_or_else(|| invalid("rank outside package"))?;
        let stored = self.protection.stored_ranges(position);
        let mut selected: BTreeMap<String, Vec<(u64, u64)>> = BTreeMap::new();
        let length = self.file.metadata().map_err(|e| e.to_string())?.len();
        selected.insert(
            String::new(),
            if self.layout.is_some() {
                vec![(0, length)]
            } else {
                vec![
                    (0, self.start),
                    (
                        self.start + self.payload,
                        length - self.start - self.payload,
                    ),
                ]
            },
        );
        for span in bundle
            .resources
            .values()
            .chain(index.assets.iter().filter_map(|a| a.payload.as_ref()))
            .chain(
                super::engram::spans(&index, Some(rank))?
                    .into_iter()
                    .map(|(_, span)| span),
            )
            .chain(&stored)
        {
            for (name, offset, size) in self.segments(span)? {
                if let Some(name) = name {
                    let file = &self.files[&name];
                    let ranges = selected.entry(name).or_default();
                    if self.layout.is_some() {
                        // A content-addressed blob is transferred whole, so it can be reused.
                        ranges.push((0, file.start + file.length));
                    } else {
                        ranges.push((0, file.start));
                        ranges.push((file.start + offset, size));
                    }
                }
            }
        }
        Ok(selected
            .into_iter()
            .map(|(filename, ranges)| PackageFile {
                file: self.files[&filename].file.clone(),
                filename,
                ranges: merge_ranges(ranges),
            })
            .collect())
    }
    pub fn span<'a>(&self, package: &'a Package, key: &str) -> Result<&'a wire::Span> {
        package
            .resources
            .get(key)
            .ok_or_else(|| invalid(format!("missing package resource {key}")))
    }
    pub fn validate(&self, package: &Package) -> Result<()> {
        for allocation in &package.allocations {
            if allocation.is_payload_bound() {
                for (source, span) in allocation.sources.iter().zip(&allocation.data_spans) {
                    let payload = self.span(package, &source.resource)?;
                    if source.nbytes != span[1] as u64
                        || source
                            .offset
                            .checked_add(source.nbytes)
                            .is_none_or(|end| end > payload.nbytes)
                    {
                        return Err(invalid("allocation source exceeds its payload resource"));
                    }
                }
            } else if let Some(key) = &allocation.file {
                if self.span(package, key)?.nbytes != allocation.stored_bytes() as u64 {
                    return Err(invalid("allocation payload size differs from its spans"));
                }
            }
        }
        for module in &package.modules {
            if self.span(package, &module.file)?.nbytes < 4 {
                return Err(invalid("empty module image"));
            }
            for global in &module.globals {
                if self.span(package, &global.file)?.nbytes != global.nbytes as u64 {
                    return Err(invalid("module global payload size mismatch"));
                }
            }
        }
        for graph in &package.graphs {
            if graph
                .node_count
                .checked_add(graph.edge_count)
                .and_then(|n| n.checked_mul(4))
                .is_none_or(|n| {
                    n as u64 > self.span(package, &graph.resource).map_or(0, |s| s.nbytes)
                })
            {
                return Err(invalid(
                    "graph resource cannot contain its declared records",
                ));
            }
        }
        Ok(())
    }
    /// Drop clean archive pages before CUDA initialization and after payload loading.
    ///
    /// Caches are sized from the free device memory the driver reports. An integrated GPU such as
    /// GB10 allocates from system memory, where the driver does not count cached file pages as free,
    /// so distribution and verification reads can otherwise prevent context or cache allocation.
    pub fn release_page_cache(&self) {
        // SAFETY: an advisory call on an open descriptor; failure only leaves the pages cached.
        for file in self.files.values() {
            crate::os::release_page_cache(&file.file);
        }
    }
    fn reader(&self, package: &Package, key: &str) -> Result<ResourceReader> {
        let span = self.span(package, key)?;
        let (source, offset) = self.source(span)?;
        self.protection.reader(source.file.clone(), offset, span)
    }
    pub fn read(&self, package: &Package, key: &str) -> Result<Vec<u8>> {
        let mut reader = self.reader(package, key)?;
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        Ok(bytes)
    }
    pub fn records(&self, package: &Package, key: &str) -> Result<Records> {
        let reader = self.reader(package, key)?;
        let nbytes = reader.len();
        Ok(Records {
            reader: BufReader::with_capacity(16 << 10, reader).take(nbytes),
            buffer: Vec::new(),
        })
    }
}

pub(super) struct Records {
    reader: Take<BufReader<ResourceReader>>,
    buffer: Vec<u8>,
}
impl Records {
    pub fn read<T: Message + Default>(&mut self) -> Result<T> {
        let mut prefix = [0u8; 4];
        self.reader
            .read_exact(&mut prefix)
            .map_err(|e| invalid(format!("truncated graph record: {e}")))?;
        let size = u32::from_le_bytes(prefix) as usize;
        if size > MAX_RECORD || size as u64 > self.reader.limit() {
            return Err(invalid("invalid graph record size"));
        }
        self.buffer.resize(size, 0);
        self.reader
            .read_exact(&mut self.buffer)
            .map_err(|e| e.to_string())?;
        T::decode(self.buffer.as_slice()).map_err(|e| invalid(format!("invalid graph record: {e}")))
    }
    pub fn finish(self) -> Result<()> {
        if self.reader.limit() != 0 {
            return Err(invalid("unexpected trailing graph records"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    #[test]
    fn transfer_ranges_merge_adjacent_and_overlapping_extents() {
        assert_eq!(
            merge_ranges(vec![(20, 3), (0, 10), (5, 10), (15, 1), (30, 0)]),
            vec![(0, 16), (20, 3)]
        );
    }

    #[test]
    fn header_and_index_lengths_are_bounded_before_reading() {
        let path =
            std::env::temp_dir().join(format!("flare-header-{}.safetensors", std::process::id()));
        for size in [0, MAX_HEADER as u64 + 1, u64::MAX] {
            std::fs::write(&path, size.to_le_bytes()).unwrap();
            assert!(
                Archive::open(&path, &[], None)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("header size")
            );
        }
        let header = serde_json::json!({
            "__metadata__": {"format": "flare"},
            "flare.payload": {"dtype": "U8", "shape": [0], "data_offsets": [0,0]},
            "flare.metadata": {"dtype": "U8", "shape": [MAX_INDEX + 1], "data_offsets": [0,MAX_INDEX + 1]},
        });
        let mut bytes = serde_json::to_vec(&header).unwrap();
        bytes.resize(4088, b' ');
        let mut file = File::create(&path).unwrap();
        file.write_all(&4088u64.to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
        drop(file);
        assert!(
            Archive::open(&path, &[], None)
                .err()
                .unwrap()
                .to_string()
                .contains("index size")
        );
        std::fs::remove_file(path).unwrap();
    }
}

fn retain(file: &Arc<File>) -> Result<Arc<File>> {
    #[cfg(windows)]
    {
        return Ok(Arc::new(crate::os::direct_read_guard(file).map_err(
            |error| invalid(format!("retain read-only package: {error}")),
        )?));
    }
    #[cfg(unix)]
    {
        Ok(file.clone())
    }
}

impl super::payload::Files for Archive {
    fn span<'a>(&self, package: &'a Package, key: &str) -> Result<&'a wire::Span> {
        Archive::span(self, package, key)
    }
    fn source(&self, span: &wire::Span) -> Result<(&PayloadFile, u64)> {
        Archive::source(self, span)
    }
    fn read(&self, package: &Package, key: &str) -> Result<Vec<u8>> {
        Archive::read(self, package, key)
    }
    fn records<'a>(
        &'a self,
        package: &Package,
        key: &str,
    ) -> Result<super::payload::RecordReader<'a>> {
        Ok(super::payload::RecordReader::Archive(Archive::records(
            self, package, key,
        )?))
    }
    fn path(&self) -> &std::path::Path {
        &self.path
    }
    fn source_index(&self, span: &wire::Span) -> Result<usize> {
        Archive::source_index(self, span)
    }
    fn asset_path(&self, span: &wire::Span) -> Result<std::path::PathBuf> {
        let (file, offset) = self.source(span)?;
        let _ = offset;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            Ok(std::path::PathBuf::from(format!(
                "/proc/self/fd/{}",
                file.file.as_raw_fd()
            )))
        }
        #[cfg(windows)]
        {
            let name = self.source_name(span)?;
            let root = self.path.as_path();
            Ok(if name.is_empty() {
                root.to_owned()
            } else {
                root.parent()
                    .unwrap_or_else(|| std::path::Path::new("."))
                    .join(name)
            })
        }
    }
    fn release_page_cache(&self) {
        Archive::release_page_cache(self)
    }
    fn direct_files(&self) -> Box<dyn Iterator<Item = Result<Arc<File>>> + '_> {
        Box::new(self.files.values().map(|file| file.direct()))
    }
    fn retained_files(&self) -> Result<Vec<Arc<File>>> {
        self.files
            .values()
            .map(|payload| retain(&payload.file))
            .chain(std::iter::once(retain(&self.file)))
            .collect()
    }
    fn validate(&self, package: &Package) -> Result<()> {
        Archive::validate(self, package)
    }
    fn resource_reader<'a>(
        &'a self,
        package: &Package,
        key: &str,
    ) -> Result<Box<dyn std::io::Read + 'a>> {
        Ok(Box::new(self.reader(package, key)?))
    }
}
