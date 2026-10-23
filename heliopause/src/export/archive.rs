//! Safetensors metadata and pinned logical package sources.

use super::{
    ExportResult as Result,
    io::{self, Identity, Source},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

pub(crate) const MAX_INDEX: u64 = 128 << 20;
const MAX_HEADER: u64 = 64 << 10;
const MAX_PAYLOAD: u64 = (64u64 << 30) - 4096;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Layout {
    pub(crate) version: u32,
    #[serde(deserialize_with = "unique_files")]
    pub(crate) files: BTreeMap<String, Vec<(u64, String)>>,
}
fn unique_files<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<BTreeMap<String, Vec<(u64, String)>>, D::Error> {
    struct Unique;
    impl<'de> serde::de::Visitor<'de> for Unique {
        type Value = BTreeMap<String, Vec<(u64, String)>>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("unique logical package files")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut result = BTreeMap::new();
            while let Some((name, parts)) = map.next_entry::<String, Vec<(u64, String)>>()? {
                if result.insert(name, parts).is_some() {
                    return Err(serde::de::Error::custom("duplicate logical package file"));
                }
            }
            Ok(result)
        }
    }
    deserializer.deserialize_map(Unique)
}
impl Layout {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.version != 1 || !self.files.contains_key("") || self.files.len() > 100 {
            return Err("invalid snapshot layout".into());
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
                    || *size > MAX_PAYLOAD
                    || count > 65536
                    || (!blob.is_empty() && !blob_name(blob))
                {
                    return Err("invalid snapshot payload segment".into());
                }
                total = total
                    .checked_add(*size)
                    .ok_or("logical package length overflow")?;
                if !blob.is_empty()
                    && lengths
                        .insert(blob, *size)
                        .is_some_and(|previous| previous != *size)
                {
                    return Err("conflicting snapshot blob lengths".into());
                }
            }
        }
        Ok(())
    }
}
pub(crate) fn blob_name(name: &str) -> bool {
    name.len() == 76
        && name.ends_with(".safetensors")
        && name.as_bytes()[..64]
            .iter()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
}
pub(crate) fn validate_filename(name: &str) -> Result<()> {
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
        return Err("invalid package companion filename".into());
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Format {
    format: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Tensor {
    dtype: String,
    shape: [u64; 1],
    data_offsets: [u64; 2],
}
impl Tensor {
    fn length(&self) -> Result<u64> {
        if self.dtype != "U8"
            || self.data_offsets[1].checked_sub(self.data_offsets[0]) != Some(self.shape[0])
        {
            return Err("invalid package byte tensor".into());
        }
        Ok(self.shape[0])
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    #[serde(rename = "__metadata__")]
    metadata: Format,
    #[serde(rename = "flare.payload")]
    payload: Tensor,
    #[serde(rename = "flare.metadata", default)]
    index: Option<Tensor>,
    #[serde(rename = "flare.layout", default)]
    layout: Option<Tensor>,
}

pub(crate) struct Container {
    pub(crate) header_json: serde_json::Value,
    pub(crate) file: Arc<File>,
    pub(crate) base: u64,
    pub(crate) payload_length: u64,
    pub(crate) envelope: Vec<u8>,
    pub(crate) layout: Option<Layout>,
    pub(crate) format: String,
}
impl Container {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        Self::from_file(Source::open(path, 0, 0)?.file)
    }
    pub(crate) fn from_file(file: Arc<File>) -> Result<Self> {
        let before = Identity::of(&file)?;
        let mut prefix = [0; 8];
        io::read_exact_at(&file, &mut prefix, 0)?;
        let size = u64::from_le_bytes(prefix);
        if size == 0 || size > MAX_HEADER || (size + 8) % 4096 != 0 {
            return Err("invalid package header size".into());
        }
        let base = size + 8;
        let mut bytes = vec![0; size as usize];
        io::read_exact_at(&file, &mut bytes, 8)?;
        let header: Header = serde_json::from_slice(&bytes)?;
        let header_json = serde_json::from_slice(&bytes)?;
        let payload_length = header.payload.length()?;
        if header.payload.data_offsets[0] != 0 {
            return Err("invalid package payload range".into());
        }
        let mut end = payload_length;
        let mut envelope = Vec::new();
        let mut layout = None;
        match header.metadata.format.as_str() {
            "flare-data" => {
                if header.index.is_some() || header.layout.is_some() {
                    return Err("unexpected companion metadata".into());
                }
            }
            "flare" | "flare-staging" | "flare-snapshot" => {
                let index = header.index.ok_or("missing model metadata")?;
                let length = index.length()?;
                if length == 0 || length > MAX_INDEX || index.data_offsets[0] != payload_length {
                    return Err("invalid model index size or range".into());
                }
                end = index.data_offsets[1];
                envelope.resize(length as usize, 0);
                io::read_exact_at(&file, &mut envelope, base + payload_length)?;
                if header.metadata.format == "flare-snapshot" {
                    let tensor = header.layout.ok_or("missing snapshot layout")?;
                    let count = tensor.length()?;
                    if payload_length != 0
                        || tensor.data_offsets[0] != end
                        || count == 0
                        || count > MAX_INDEX
                    {
                        return Err("invalid snapshot layout tensor".into());
                    }
                    let mut encoded = vec![0; count as usize];
                    io::read_exact_at(&file, &mut encoded, base + end)?;
                    let decoded: Layout = serde_json::from_slice(&encoded)?;
                    decoded.validate()?;
                    layout = Some(decoded);
                    end = tensor.data_offsets[1];
                } else if header.layout.is_some() {
                    return Err("unexpected snapshot layout".into());
                }
            }
            _ => return Err("not a Flare package".into()),
        }
        if file.metadata()?.len() != base.checked_add(end).ok_or("container length overflow")? {
            return Err("safetensors data length differs from model package".into());
        }
        if before != Identity::of(&file)? {
            return Err("package metadata changed while reading".into());
        }
        Ok(Self {
            header_json,
            file,
            base,
            payload_length,
            envelope,
            layout,
            format: header.metadata.format,
        })
    }
}

#[derive(Clone)]
pub(crate) enum SourcePiece {
    Data(Source),
    Zero(u64),
}
pub(crate) struct PackageSources {
    pub(crate) container: Container,
    directory: PathBuf,
    physical: BTreeMap<String, Source>,
    _directory: File,
}
impl PackageSources {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let directory = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let directory_handle = open_directory(directory)?;
        let stable = pinned_directory(&directory_handle, directory);
        let container =
            Container::open(&stable.join(path.file_name().ok_or("missing primary filename")?))?;
        let mut physical = BTreeMap::new();
        physical.insert(
            String::new(),
            Source::from_file(
                container.file.clone(),
                container.base,
                container.payload_length,
            )?,
        );
        Ok(Self {
            container,
            directory: stable,
            physical,
            _directory: directory_handle,
        })
    }
    pub(crate) fn from_layout(directory: &Path, layout: Layout) -> Result<Self> {
        layout.validate()?;
        let directory_handle = open_directory(directory)?;
        let stable = pinned_directory(&directory_handle, directory);
        // The directory handle is retained independently of the unpublished primary.
        let file = Arc::new(directory_handle.try_clone()?);
        let container = Container {
            header_json: serde_json::Value::Null,
            file,
            base: 0,
            payload_length: 0,
            envelope: Vec::new(),
            layout: Some(layout),
            format: "flare-snapshot".to_owned(),
        };
        Ok(Self {
            container,
            directory: stable,
            physical: BTreeMap::new(),
            _directory: directory_handle,
        })
    }
    fn physical(&mut self, name: &str) -> Result<Source> {
        if let Some(source) = self.physical.get(name) {
            return Ok(source.clone());
        }
        validate_filename(name)?;
        if self.physical.len()
            >= if self.container.layout.is_some() {
                65537
            } else {
                100
            }
        {
            return Err("too many physical package files".into());
        }
        let file = Container::open(&self.directory.join(name))?;
        if file.format != "flare-data" {
            return Err("invalid package companion format".into());
        }
        let source = Source::from_file(file.file, file.base, file.payload_length)?;
        self.physical.insert(name.to_owned(), source.clone());
        Ok(source)
    }
    pub(crate) fn logical_length(&mut self, name: &str) -> Result<u64> {
        if let Some(layout) = &self.container.layout {
            return Ok(layout
                .files
                .get(name)
                .ok_or("missing logical package file")?
                .iter()
                .map(|p| p.0)
                .sum());
        }
        Ok(self.physical(name)?.length)
    }
    pub(crate) fn sources(
        &mut self,
        name: &str,
        offset: u64,
        length: u64,
    ) -> Result<Vec<SourcePiece>> {
        let end = offset.checked_add(length).ok_or("logical range overflow")?;
        if end > self.logical_length(name)? {
            return Err("resource exceeds logical payload".into());
        }
        let Some(layout) = &self.container.layout else {
            return Ok(vec![SourcePiece::Data(
                self.physical(name)?.slice(offset, length)?,
            )]);
        };
        let parts = layout
            .files
            .get(name)
            .ok_or("missing logical package file")?
            .clone();
        let mut cursor = 0;
        let mut pieces = Vec::new();
        for (size, blob) in parts {
            let stop = cursor + size;
            if cursor < end && offset < stop {
                let first = offset.max(cursor);
                let count = end.min(stop) - first;
                pieces.push(if blob.is_empty() {
                    SourcePiece::Zero(count)
                } else {
                    let source = self.physical(&blob)?;
                    if source.offset != 4096 || source.length != size {
                        return Err("snapshot blob length or alignment mismatch".into());
                    }
                    SourcePiece::Data(source.slice(first - cursor, count)?)
                });
            }
            cursor = stop;
        }
        Ok(pieces)
    }
    pub(crate) fn read_at(&mut self, name: &str, offset: u64, output: &mut [u8]) -> Result<()> {
        let mut cursor = 0;
        for piece in self.sources(name, offset, output.len() as u64)? {
            let count = match &piece {
                SourcePiece::Data(source) => source.length,
                SourcePiece::Zero(count) => *count,
            } as usize;
            match piece {
                SourcePiece::Data(source) => {
                    source.read_at(0, &mut output[cursor..cursor + count])?
                }
                SourcePiece::Zero(_) => output[cursor..cursor + count].fill(0),
            }
            cursor += count;
        }
        if cursor != output.len() {
            return Err("truncated logical package range".into());
        }
        Ok(())
    }
}

fn encode_header(raw: Vec<u8>) -> Result<Vec<u8>> {
    let size = (raw.len() + 8).next_multiple_of(4096) - 8;
    if size > MAX_HEADER as usize {
        return Err("package header exceeds its size limit".into());
    }
    let mut header = Vec::with_capacity(size + 8);
    header.extend_from_slice(&(size as u64).to_le_bytes());
    header.extend_from_slice(&raw);
    header.resize(size + 8, b' ');
    Ok(header)
}
pub(crate) fn container_header(length: u64, index_size: u64, format: &str) -> Result<Vec<u8>> {
    let mut raw = format!(
        "{{\"__metadata__\":{{\"format\":{}}},\"flare.payload\":{{\"dtype\":\"U8\",\"shape\":[{}],\"data_offsets\":[0,{}]}}",
        serde_json::to_string(format)?,
        length,
        length
    );
    if format != "flare-data" {
        raw.push_str(&format!(
            ",\"flare.metadata\":{{\"dtype\":\"U8\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            index_size,
            length,
            length
                .checked_add(index_size)
                .ok_or("container length overflow")?
        ));
    }
    raw.push('}');
    encode_header(raw.into_bytes())
}
pub(crate) fn temporary(directory: &Path, prefix: &str) -> Result<(File, PathBuf)> {
    for _ in 0..100 {
        let path = directory.join(format!(
            ".{prefix}-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => return Ok((file, path)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Err("cannot create unique export temporary file".into())
}
pub(crate) fn install_file(temporary: &Path, destination: &Path) -> Result<()> {
    std::fs::hard_link(temporary, destination)?;
    Ok(())
}
pub(crate) fn install_blob(temporary: &Path, destination: &Path) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::ffi::OsStrExt;
        let source = std::ffi::CString::new(temporary.as_os_str().as_bytes())?;
        let target = std::ffi::CString::new(destination.as_os_str().as_bytes())?;
        if unsafe {
            libc::renameat2(
                libc::AT_FDCWD,
                source.as_ptr(),
                libc::AT_FDCWD,
                target.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        } != 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Storage::FileSystem::{MOVEFILE_WRITE_THROUGH, MoveFileExW};
        let source: Vec<u16> = temporary.as_os_str().encode_wide().chain(Some(0)).collect();
        let target: Vec<u16> = destination
            .as_os_str()
            .encode_wide()
            .chain(Some(0))
            .collect();
        if unsafe { MoveFileExW(source.as_ptr(), target.as_ptr(), MOVEFILE_WRITE_THROUGH) } == 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    Ok(())
}
pub(crate) fn remove_temporary(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        std::fs::remove_file(path)?;
    }
    #[cfg(windows)]
    {
        use std::os::windows::{fs::OpenOptionsExt, io::AsRawHandle};
        use windows_sys::Win32::Storage::FileSystem::*;
        let file = OpenOptions::new()
            .access_mode(DELETE)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
            .open(path)?;
        let info = FILE_DISPOSITION_INFO_EX {
            Flags: FILE_DISPOSITION_FLAG_DELETE
                | FILE_DISPOSITION_FLAG_POSIX_SEMANTICS
                | FILE_DISPOSITION_FLAG_IGNORE_READONLY_ATTRIBUTE,
        };
        if unsafe {
            SetFileInformationByHandle(
                file.as_raw_handle(),
                FileDispositionInfoEx,
                (&info as *const FILE_DISPOSITION_INFO_EX).cast(),
                std::mem::size_of_val(&info) as u32,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    Ok(())
}
pub(crate) fn install_snapshot(destination: &Path, envelope: &[u8], layout: &Layout) -> Result<()> {
    layout.validate()?;
    let encoded = serde_json::to_vec(layout)?;
    if envelope.is_empty()
        || envelope.len() as u64 > MAX_INDEX
        || encoded.is_empty()
        || encoded.len() as u64 > MAX_INDEX
    {
        return Err("snapshot metadata exceeds its limit".into());
    }
    let raw = format!(
        "{{\"__metadata__\":{{\"format\":\"flare-snapshot\"}},\"flare.payload\":{{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[0,0]}},\"flare.metadata\":{{\"dtype\":\"U8\",\"shape\":[{}],\"data_offsets\":[0,{}]}},\"flare.layout\":{{\"dtype\":\"U8\",\"shape\":[{}],\"data_offsets\":[{},{}]}}}}",
        envelope.len(),
        envelope.len(),
        encoded.len(),
        envelope.len(),
        envelope.len() + encoded.len()
    );
    let header = encode_header(raw.into_bytes())?;
    let directory = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(directory)?;
    sync_directory(directory)?;
    let (file, path) = temporary(directory, "snapshot")?;
    let mut installed = false;
    let result = (|| {
        io::write_all_at(&file, &header, 0)?;
        io::write_all_at(&file, envelope, header.len() as u64)?;
        io::write_all_at(&file, &encoded, (header.len() + envelope.len()) as u64)?;
        file.sync_all()?;
        install_file(&path, destination)?;
        installed = true;
        sync_directory(directory)
    })();
    let removed = remove_temporary(&path);
    let result = result.and(removed);
    if result.is_err() && installed {
        if let Ok(linked) = io::open_read(destination) {
            if Identity::of(&file)?.same_file(&Identity::of(&linked)?) {
                std::fs::remove_file(destination)?;
            }
        }
    }
    result
}
fn open_directory(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Storage::FileSystem::*;
        options
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE);
    }
    Ok(options.open(path)?)
}
fn pinned_directory(file: &File, path: &Path) -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        let _ = path;
        PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()))
    }
    #[cfg(windows)]
    {
        let _ = file;
        path.to_owned()
    }
}
pub(crate) fn sync_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        open_directory(path)?.sync_all()?;
    }
    #[cfg(windows)]
    {
        use std::{
            ffi::c_void,
            os::windows::{fs::OpenOptionsExt, io::AsRawHandle},
        };
        use windows_sys::Win32::{Foundation::HANDLE, Storage::FileSystem::*};
        #[repr(C)]
        struct IoStatus {
            status: isize,
            information: usize,
        }
        #[link(name = "ntdll")]
        unsafe extern "system" {
            fn NtFlushBuffersFileEx(
                file: HANDLE,
                flags: u32,
                parameters: *mut c_void,
                size: u32,
                status: *mut IoStatus,
            ) -> i32;
            fn RtlNtStatusToDosError(status: i32) -> u32;
        }
        let directory = OpenOptions::new()
            .read(true)
            .write(true)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(path)?;
        let mut status = IoStatus {
            status: 0,
            information: 0,
        };
        let result = unsafe {
            NtFlushBuffersFileEx(
                directory.as_raw_handle(),
                0,
                std::ptr::null_mut(),
                0,
                &mut status,
            )
        };
        if result < 0 {
            return Err(std::io::Error::from_raw_os_error(
                unsafe { RtlNtStatusToDosError(result) } as i32,
            )
            .into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blob_header_preserves_python_content_identity() {
        use sha2::{Digest, Sha256};
        let header = container_header(3, 0, "flare-data").unwrap();
        assert_eq!(header.len(), 4096);
        assert_eq!(&header[..8], &4088u64.to_le_bytes());
        let mut hash = Sha256::new();
        hash.update(header);
        hash.update(b"abc");
        assert_eq!(
            super::super::blob::hex(&hash.finalize()),
            "8f56e06cc2a240781ab8020dcd057b2d0f80df7599500615e6f43cc3b0a92885"
        );
    }
    #[test]
    fn duplicate_files_and_conflicting_blob_sizes_are_rejected() {
        assert!(serde_json::from_str::<Layout>(r#"{"version":1,"files":{"":[],"":[]}}"#).is_err());
        let name = format!("{}.safetensors", "a".repeat(64));
        let layout = Layout {
            version: 1,
            files: BTreeMap::from([(String::new(), vec![(1, name.clone()), (2, name)])]),
        };
        assert!(layout.validate().is_err());
    }
    #[test]
    fn snapshot_resolves_data_and_padding_without_overwrite() {
        let directory = std::env::temp_dir().join(format!(
            "flare-layout-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let blob = format!("{}.safetensors", "a".repeat(64));
        let mut bytes = container_header(3, 0, "flare-data").unwrap();
        bytes.extend_from_slice(b"abc");
        std::fs::write(directory.join(&blob), bytes).unwrap();
        let layout = Layout {
            version: 1,
            files: BTreeMap::from([(
                String::new(),
                vec![(2, String::new()), (3, blob), (1, String::new())],
            )]),
        };
        let path = directory.join("model.safetensors");
        install_snapshot(&path, b"signed envelope", &layout).unwrap();
        assert!(install_snapshot(&path, b"replacement", &layout).is_err());
        let mut sources = PackageSources::open(&path).unwrap();
        let mut output = [0xff; 6];
        sources.read_at("", 0, &mut output).unwrap();
        assert_eq!(&output, b"\0\0abc\0");
        assert!(sources.read_at("", 5, &mut [0; 2]).is_err());
        assert_eq!(&sources.container.envelope, b"signed envelope");
        drop(sources);
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn staging_and_companion_sources_keep_payload_bounds() {
        let directory = std::env::temp_dir().join(format!(
            "flare-staging-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let primary = directory.join("model.safetensors");
        let mut data = container_header(3, 1, "flare-staging").unwrap();
        data.extend_from_slice(b"abcx");
        std::fs::write(&primary, data).unwrap();
        let mut companion = container_header(2, 0, "flare-data").unwrap();
        companion.extend_from_slice(b"yz");
        std::fs::write(directory.join("data.safetensors"), companion).unwrap();
        let mut sources = PackageSources::open(&primary).unwrap();
        let mut output = [0; 2];
        sources.read_at("data.safetensors", 0, &mut output).unwrap();
        assert_eq!(&output, b"yz");
        assert!(sources.sources("", 2, 2).is_err());
        assert_eq!(sources.container.format, "flare-staging");
        drop(sources);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
