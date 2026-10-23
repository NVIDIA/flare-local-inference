//! Shared native cache of executables and rank package files, under `~/.cache/flare`.
//!
//! Executables live in `bin/` under their SHA-256 names and run only from there. Package files
//! live in `packages/<directory>/<filename>`: one directory per package and rank, holding each
//! file at its logical length with only the transferred byte ranges written, so a rank's copy is
//! a sparse file. Content-addressed payloads arrive whole and are hardlinked from `blobs/`,
//! so package directories share their immutable bytes. Entries are pinned by transferring
//! sessions and running groups; package eviction preserves the reusable blob store.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};

use crate::fabric::Failure;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn default_root() -> PathBuf {
    crate::environment::home_directory()
        .expect("home directory")
        .join(".cache/flare")
}

/// File locks protect rank files from concurrent writes and agent eviction.
/// Lock files live outside the removable package directory, so their inodes stay stable.
pub struct Lease(File);

impl Drop for Lease {
    fn drop(&mut self) {
        // A concurrent process launch can briefly inherit this open file description before
        // exec closes it. Release the lock when its owning rank finishes, independently of that.
        let _ = self.lock(libc::LOCK_UN);
    }
}

impl Lease {
    pub fn shared(directory: &Path) -> Result<Self, Failure> {
        let file = Self::open(directory)?;
        let lease = Self(file);
        lease.lock(libc::LOCK_SH)?;
        Ok(lease)
    }

    fn open(directory: &Path) -> Result<File, Failure> {
        let locks = directory
            .parent()
            .ok_or("cache directory lacks a parent")?
            .join(".locks");
        fs::create_dir_all(&locks)?;
        let name = directory
            .file_name()
            .ok_or("cache directory lacks a name")?;
        Ok(fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(locks.join(name))?)
    }

    fn lock(&self, mode: i32) -> Result<(), Failure> {
        use std::os::fd::AsRawFd;
        if unsafe { libc::flock(self.0.as_raw_fd(), mode) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(())
    }

    pub fn exclusive(&self) -> Result<(), Failure> {
        self.lock(libc::LOCK_EX)
    }
    pub fn downgrade(&self) -> Result<(), Failure> {
        self.lock(libc::LOCK_SH)
    }

    fn evict(directory: &Path) -> Result<Option<Self>, Failure> {
        use std::os::fd::AsRawFd;
        let lease = Self(Self::open(directory)?);
        if unsafe { libc::flock(lease.0.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::WouldBlock {
                return Ok(None);
            }
            return Err(error.into());
        }
        Ok(Some(lease))
    }
}

/// The SHA-256 name under which an executable is cached and verified.
pub fn binary_name(bytes: &[u8]) -> String {
    binary_name_from_digest(&Sha256::digest(bytes))
}

fn binary_name_from_digest(digest: &[u8]) -> String {
    format!("flare-{}", hex(&digest[..12]))
}

pub fn kind_directory(kind: &str) -> Result<&'static str, Failure> {
    match kind {
        "binary" => Ok("bin"),
        "package" => Ok("packages"),
        _ => Err(format!("unknown cache kind {kind:?}").into()),
    }
}

fn component_valid(part: &str) -> bool {
    !part.is_empty()
        && !part.starts_with('.')
        && !part.contains("..")
        && part
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
}

/// A cache entry path: `bin/NAME` or `packages/DIRECTORY/FILENAME`, with no traversal.
pub fn entry(cache: &Path, directory: &str, name: &str) -> Result<PathBuf, Failure> {
    let components: Vec<_> = name.split('/').collect();
    let expected = if directory == "packages" { 2 } else { 1 };
    if components.len() != expected || !components.iter().all(|part| component_valid(part)) {
        return Err(format!("invalid cache entry name {name:?}").into());
    }
    Ok(cache.join(directory).join(name))
}

/// The unit eviction removes: a package directory, or an executable file.
pub fn unit(cache: &Path, path: &Path) -> PathBuf {
    let packages = cache.join("packages");
    if path.parent().and_then(Path::parent) == Some(packages.as_path()) {
        path.parent().unwrap().to_owned()
    } else {
        path.to_owned()
    }
}

/// Pinned units with their pin counts.
pub type Pins = Arc<Mutex<HashMap<PathBuf, usize>>>;

/// One holder's pins, released when dropped.
pub struct Pinned {
    pins: Pins,
    paths: Vec<PathBuf>,
}

impl Pinned {
    pub fn new(pins: &Pins) -> Self {
        Self {
            pins: pins.clone(),
            paths: Vec::new(),
        }
    }

    pub fn pin(&mut self, unit: PathBuf) {
        *self.pins.lock().unwrap().entry(unit.clone()).or_default() += 1;
        self.paths.push(unit);
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        let mut pins = self.pins.lock().unwrap();
        for path in &self.paths {
            if let Some(count) = pins.get_mut(path) {
                *count -= 1;
                if *count == 0 {
                    pins.remove(path);
                }
            }
        }
    }
}

/// Remove a kind directory's units that nothing pins, leaving transfers in progress.
pub fn evict(directory: &Path, pins: &Pins) -> Result<Vec<PathBuf>, Failure> {
    let pins = pins.lock().unwrap();
    let mut removed = Vec::new();
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        let hidden = path
            .file_name()
            .is_none_or(|name| name.to_string_lossy().starts_with('.'));
        if !hidden && !pins.contains_key(&path) {
            let Some(_lease) = Lease::evict(&path)? else {
                continue;
            };
            if path.is_dir() {
                fs::remove_dir_all(&path)?;
            } else {
                fs::remove_file(&path)?;
            }
            removed.push(path);
        }
    }
    Ok(removed)
}

/// The entry's path when the cache holds it at `size` bytes.
pub fn have(cache: &Path, kind: &str, name: &str, size: u64) -> Result<Option<PathBuf>, Failure> {
    let path = entry(cache, kind_directory(kind)?, name)?;
    if kind == "package" && !path.exists() {
        if let Some(blob) = blob_name(name) {
            let stored = cache.join("blobs").join(blob);
            if fs::symlink_metadata(&stored).is_ok_and(|m| m.is_file() && m.len() == size) {
                fs::create_dir_all(path.parent().unwrap())?;
                match fs::hard_link(stored, &path) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(e) => return Err(e.into()),
                }
            }
        }
    }
    Ok(fs::symlink_metadata(&path)
        .is_ok_and(|m| m.is_file() && m.len() == size)
        .then_some(path))
}

fn blob_name(name: &str) -> Option<&str> {
    let file = name.rsplit('/').next()?;
    let hash = file.strip_suffix(".safetensors")?;
    (hash.len() == 64
        && hash
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)))
    .then_some(file)
}

/// Validate byte ranges: increasing, disjoint, non-empty, within `size`.
pub fn validate_ranges(size: u64, ranges: &[(u64, u64)]) -> Result<(), Failure> {
    let mut end = 0;
    for &(offset, nbytes) in ranges {
        if offset < end || nbytes == 0 || offset.checked_add(nbytes).is_none_or(|e| e > size) {
            return Err("ranges must increase, stay disjoint and fit the file".into());
        }
        end = offset + nbytes;
    }
    Ok(())
}

pub fn ranges_from_json(
    size: u64,
    listed: Option<&serde_json::Value>,
) -> Result<Vec<(u64, u64)>, Failure> {
    let Some(listed) = listed.filter(|r| !r.is_null()) else {
        return Ok(vec![(0, size)]);
    };
    let mut ranges = Vec::new();
    for range in listed.as_array().ok_or("ranges must be a list")? {
        let (Some(offset), Some(nbytes)) = (range[0].as_u64(), range[1].as_u64()) else {
            return Err("each range is [offset, bytes]".into());
        };
        ranges.push((offset, nbytes));
    }
    validate_ranges(size, &ranges)?;
    Ok(ranges)
}

/// Total bytes the ranges cover.
pub fn range_bytes(ranges: &[(u64, u64)]) -> u64 {
    ranges.iter().map(|&(_, n)| n).sum()
}

/// Store `ranges` of an entry of `size` bytes, read in order from `source`, as `kind`/`name`.
/// Bytes outside the ranges stay holes. An executable must hash to its name.
pub fn store(
    cache: &Path,
    kind: &str,
    name: &str,
    size: u64,
    ranges: &[(u64, u64)],
    source: &mut dyn Read,
) -> Result<PathBuf, Failure> {
    store_entry(cache, kind, name, size, ranges, |partial| {
        store_partial(partial, kind, name, size, ranges, source)
    })
}

/// Divide the requested bytes evenly without filling sparse holes or overlapping writes.
fn split_ranges(ranges: &[(u64, u64)], streams: usize) -> Vec<Vec<(u64, u64)>> {
    let total = range_bytes(ranges);
    let width = total.div_ceil(streams as u64).max(1);
    let mut parts = Vec::new();
    let mut part = Vec::new();
    let mut remaining = width;
    for &(mut offset, mut count) in ranges {
        while count > 0 {
            let take = count.min(remaining);
            part.push((offset, take));
            offset += take;
            count -= take;
            remaining -= take;
            if remaining == 0 {
                parts.push(std::mem::take(&mut part));
                remaining = width;
            }
        }
    }
    if !part.is_empty() {
        parts.push(part);
    }
    parts
}

/// Fetch disjoint ranges on independent connections, then publish the complete cache entry.
/// Each worker holds at most 1 MiB of file data; all workers drain before cleanup or publication.
pub fn store_parallel<R: Read>(
    cache: &Path,
    name: &str,
    size: u64,
    ranges: &[(u64, u64)],
    streams: usize,
    source: impl Fn(usize, &[(u64, u64)]) -> Result<R, Failure> + Sync,
) -> Result<PathBuf, Failure> {
    use std::os::unix::fs::FileExt;
    use std::sync::atomic::{AtomicBool, Ordering};

    if !(1..=64).contains(&streams) {
        return Err("transfer streams must be in 1..=64".into());
    }
    store_entry(cache, "package", name, size, ranges, |partial| {
        let output = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(partial)?;
        output.set_len(size)?;
        let parts = split_ranges(ranges, streams);
        let failed = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let workers = parts
                .iter()
                .enumerate()
                .map(|(index, part)| {
                    let source = &source;
                    let output = &output;
                    let failed = &failed;
                    scope.spawn(move || {
                        let result = (|| -> Result<(), Failure> {
                            let mut reader = source(index, part)?;
                            let mut buffer = vec![0; 1 << 20];
                            for &(mut offset, mut count) in part {
                                while count > 0 {
                                    if failed.load(Ordering::Relaxed) {
                                        return Ok(()); // Another worker's error prevents publication.
                                    }
                                    let take = count.min(buffer.len() as u64) as usize;
                                    reader.read_exact(&mut buffer[..take])?;
                                    output.write_all_at(&buffer[..take], offset)?;
                                    offset += take as u64;
                                    count -= take as u64;
                                }
                            }
                            Ok(())
                        })();
                        if result.is_err() {
                            failed.store(true, Ordering::Relaxed);
                        }
                        result
                    })
                })
                .collect::<Vec<_>>();
            let mut result = Ok(());
            for worker in workers {
                let joined = worker
                    .join()
                    .unwrap_or_else(|_| Err("transfer worker panicked".into()));
                if joined.is_err() {
                    failed.store(true, Ordering::Relaxed);
                    result = joined;
                }
            }
            result
        })?;
        output.sync_all()?;
        if let Some(blob) = blob_name(name) {
            // Parallel writes cannot update one ordered digest. Hash the assembled file before
            // it can enter the shared immutable blob store, including on a cache collision.
            let mut digest = Sha256::new();
            std::io::copy(&mut &output, &mut digest)?;
            if format!("{}.safetensors", hex(&digest.finalize())) != blob {
                return Err(format!("{name}: payload digest mismatch").into());
            }
            output.set_permissions(fs::Permissions::from_mode(0o444))?;
        }
        Ok(())
    })
}

fn store_entry(
    cache: &Path,
    kind: &str,
    name: &str,
    size: u64,
    ranges: &[(u64, u64)],
    write: impl FnOnce(&Path) -> Result<(), Failure>,
) -> Result<PathBuf, Failure> {
    let path = entry(cache, kind_directory(kind)?, name)?;
    validate_ranges(size, ranges)?;
    if kind == "package" && blob_name(name).is_some() && ranges != [(0, size)] {
        return Err("content-addressed payloads must be transferred whole".into());
    }
    if kind == "binary" && ranges != [(0, size)] {
        return Err("only packages arrive as byte ranges".into());
    }
    fs::create_dir_all(path.parent().unwrap())?;
    let partial = path.with_file_name(format!(
        ".{}.partial",
        path.file_name().unwrap().to_string_lossy()
    ));
    let result = write(&partial);
    if result.is_err() {
        let _ = fs::remove_file(&partial);
        return result.map(|_| path);
    }
    if let Some(blob) = (kind == "package").then(|| blob_name(name)).flatten() {
        let directory = cache.join("blobs");
        fs::create_dir_all(&directory)?;
        let stored = directory.join(blob);
        match fs::hard_link(&partial, &stored) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                use std::os::unix::fs::OpenOptionsExt;
                let mut source = fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                    .open(&stored)?;
                let metadata = source.metadata()?;
                if !metadata.is_file() || metadata.len() != size {
                    return Err("invalid cached payload blob".into());
                }
                let mut digest = Sha256::new();
                std::io::copy(&mut source, &mut digest)?;
                if format!("{}.safetensors", hex(&digest.finalize())) != blob {
                    return Err("cached payload digest mismatch".into());
                }
            }
            Err(e) => return Err(e.into()),
        }
        fs::remove_file(&partial)?;
        fs::hard_link(stored, &partial)?;
    }
    fs::rename(&partial, &path)?;
    Ok(path)
}

fn store_partial(
    partial: &Path,
    kind: &str,
    name: &str,
    size: u64,
    ranges: &[(u64, u64)],
    source: &mut dyn Read,
) -> Result<(), Failure> {
    let mut output = File::create(partial)?;
    output.set_len(size)?;
    let mut digest = Sha256::new();
    let mut block = vec![0u8; 16 << 20];
    for &(offset, nbytes) in ranges {
        output.seek(SeekFrom::Start(offset))?;
        let mut remaining = nbytes;
        while remaining > 0 {
            let want = remaining.min(block.len() as u64) as usize;
            let count = source.read(&mut block[..want])?;
            if count == 0 {
                return Err(format!("{name}: transfer ended {remaining} bytes early").into());
            }
            if kind == "binary" || blob_name(name).is_some() {
                digest.update(&block[..count]);
            }
            output.write_all(&block[..count])?;
            remaining -= count as u64;
        }
    }
    output.sync_all()?;
    if kind == "binary" {
        let found = binary_name_from_digest(&digest.finalize());
        if found != name {
            return Err(format!("{name}: received bytes hash to {found}").into());
        }
        fs::set_permissions(partial, fs::Permissions::from_mode(0o700))?;
    } else if let Some(blob) = blob_name(name) {
        let found = format!("{}.safetensors", hex(&digest.finalize()));
        if found != blob {
            return Err(format!("{name}: payload digest mismatch").into());
        }
        fs::set_permissions(partial, fs::Permissions::from_mode(0o444))?;
    }
    Ok(())
}

/// Read `ranges` of `file` in order, as a transfer sends them.
pub struct RangeReader<'a> {
    file: &'a File,
    ranges: std::slice::Iter<'a, (u64, u64)>,
    position: u64,
    remaining: u64,
}

impl<'a> RangeReader<'a> {
    pub fn new(file: &'a File, ranges: &'a [(u64, u64)]) -> Self {
        Self {
            file,
            ranges: ranges.iter(),
            position: 0,
            remaining: 0,
        }
    }
}

impl Read for RangeReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        use std::os::unix::fs::FileExt;
        while self.remaining == 0 {
            match self.ranges.next() {
                Some(&(offset, nbytes)) => {
                    self.position = offset;
                    self.remaining = nbytes;
                }
                None => return Ok(0),
            }
        }
        let want = buffer.len().min(self.remaining as usize);
        let count = self.file.read_at(&mut buffer[..want], self.position)?;
        if count == 0 {
            return Err(std::io::Error::other("file ended inside a requested range"));
        }
        self.position += count as u64;
        self.remaining -= count as u64;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_native_reader_lease_prevents_agent_eviction() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = temporary.path().join("packages");
        let package = directory.join("export-rank1");
        fs::create_dir_all(&package).unwrap();
        fs::write(package.join("model.safetensors"), b"held").unwrap();
        let lease = Lease::shared(&package).unwrap();
        let pins = Pins::default();
        assert!(evict(&directory, &pins).unwrap().is_empty());
        assert!(package.is_dir());
        drop(lease);
        assert_eq!(evict(&directory, &pins).unwrap(), vec![package.clone()]);
        assert!(!package.exists());
    }

    #[test]
    fn entries_are_bounded_components() {
        let cache = Path::new("/c");
        assert!(entry(cache, "bin", "flare-00").is_ok());
        assert!(entry(cache, "packages", "export-1-rank1/model.safetensors").is_ok());
        for name in ["", "../x", "a/b", ".hidden"] {
            assert!(entry(cache, "bin", name).is_err(), "{name}");
        }
        for name in [
            "a",
            "a/../escape",
            "a/b/c",
            "a/",
            "a/hidden\\name",
            "a/.hidden",
        ] {
            assert!(entry(cache, "packages", name).is_err(), "{name}");
        }
        assert_eq!(
            unit(cache, Path::new("/c/packages/d/f")),
            PathBuf::from("/c/packages/d")
        );
        assert_eq!(
            unit(cache, Path::new("/c/bin/f")),
            PathBuf::from("/c/bin/f")
        );
    }

    #[test]
    fn ranges_store_sparsely_and_executables_must_hash_to_their_names() {
        use std::os::unix::fs::MetadataExt;
        let temp = tempfile::tempdir().unwrap();
        let cache = temp.path();
        let bytes: Vec<u8> = (0..(8u32 << 20)).map(|i| (i % 251) as u8).collect();
        let ranges = [(0, 4096), (5 << 20, 1 << 20)];
        validate_ranges(bytes.len() as u64, &ranges).unwrap();
        assert!(validate_ranges(100, &[(0, 50), (40, 10)]).is_err());
        assert!(validate_ranges(100, &[(0, 0)]).is_err());
        let source = temp.path().join("source");
        fs::write(&source, &bytes).unwrap();
        let file = File::open(&source).unwrap();
        let mut reader = RangeReader::new(&file, &ranges);
        let path = store(
            cache,
            "package",
            "e-8-rank1/m.safetensors",
            bytes.len() as u64,
            &ranges,
            &mut reader,
        )
        .unwrap();
        let copy = fs::read(&path).unwrap();
        assert_eq!(copy.len(), bytes.len());
        for &(offset, nbytes) in &ranges {
            let span = offset as usize..(offset + nbytes) as usize;
            assert_eq!(copy[span.clone()], bytes[span]);
        }
        assert!(copy[8192..1 << 20].iter().all(|&b| b == 0));
        assert!(fs::metadata(&path).unwrap().blocks() * 512 <= 4 << 20);
        assert_eq!(
            have(
                cache,
                "package",
                "e-8-rank1/m.safetensors",
                bytes.len() as u64
            )
            .unwrap(),
            Some(path.clone())
        );
        assert_eq!(
            have(cache, "package", "e-8-rank1/m.safetensors", 1).unwrap(),
            None
        );

        let script = b"#!/bin/sh\necho hi\n";
        let name = binary_name(script);
        let stored = store(
            cache,
            "binary",
            &name,
            script.len() as u64,
            &[(0, script.len() as u64)],
            &mut &script[..],
        )
        .unwrap();
        assert_eq!(
            fs::metadata(&stored).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert!(
            store(
                cache,
                "binary",
                "flare-000000000000000000000000",
                script.len() as u64,
                &[(0, script.len() as u64)],
                &mut &script[..]
            )
            .is_err()
        );
        assert!(
            !cache
                .join("bin/.flare-000000000000000000000000.partial")
                .exists()
        );

        // Eviction removes unpinned units of a kind and keeps pinned ones.
        let pins = Pins::default();
        let mut pinned = Pinned::new(&pins);
        pinned.pin(unit(cache, &path));
        let removed = evict(&cache.join("packages"), &pins).unwrap();
        assert!(removed.is_empty() && path.exists());
        drop(pinned);
        let removed = evict(&cache.join("packages"), &pins).unwrap();
        assert_eq!(removed, vec![cache.join("packages/e-8-rank1")]);
    }

    #[test]
    fn parallel_ranges_preserve_sparse_bytes_and_small_tails() {
        use std::os::unix::fs::MetadataExt;
        use std::sync::Barrier;
        let temp = tempfile::tempdir().unwrap();
        let bytes = (0..9_000_019).map(|i| (i % 251) as u8).collect::<Vec<_>>();
        let ranges = [(7, 101), (3_000_000, 5_999_999)];
        for streams in [1, 4, 8, 16, 32, 64] {
            let ready = Barrier::new(streams);
            let path = store_parallel(
                temp.path(),
                &format!("rank{streams}/m.safetensors"),
                bytes.len() as u64,
                &ranges,
                streams,
                |_, part| {
                    ready.wait(); // Every requested stream must be live concurrently.
                    let data = part
                        .iter()
                        .flat_map(|&(offset, count)| {
                            bytes[offset as usize..(offset + count) as usize]
                                .iter()
                                .copied()
                        })
                        .collect::<Vec<_>>();
                    Ok(std::io::Cursor::new(data))
                },
            )
            .unwrap();
            let copied = fs::read(&path).unwrap();
            for &(offset, count) in &ranges {
                let span = offset as usize..(offset + count) as usize;
                assert_eq!(copied[span.clone()], bytes[span]);
            }
            assert!(copied[108..3_000_000].iter().all(|&b| b == 0));
            assert!(fs::metadata(path).unwrap().blocks() * 512 < bytes.len() as u64);
        }
        let parts = split_ranges(&[(10, 3), (100, 2)], 8);
        assert_eq!(
            parts,
            vec![
                vec![(10, 1)],
                vec![(11, 1)],
                vec![(12, 1)],
                vec![(100, 1)],
                vec![(101, 1)]
            ]
        );
        assert!(split_ranges(&[], 8).is_empty());
        for streams in [0, 65] {
            assert!(
                store_parallel(
                    temp.path(),
                    "bad/m.safetensors",
                    1,
                    &[(0, 1)],
                    streams,
                    |_, _| Ok(std::io::empty())
                )
                .is_err()
            );
        }
    }

    #[test]
    fn parallel_failure_drains_readers_and_removes_partial_files() {
        use std::sync::{
            Barrier,
            atomic::{AtomicUsize, Ordering},
        };
        struct Reader<'a> {
            active: &'a AtomicUsize,
            fail: bool,
        }
        impl Read for Reader<'_> {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                if self.fail {
                    return Err(std::io::Error::other("injected stream failure"));
                }
                out.fill(1);
                Ok(out.len())
            }
        }
        impl Drop for Reader<'_> {
            fn drop(&mut self) {
                self.active.fetch_sub(1, Ordering::SeqCst);
            }
        }
        let temp = tempfile::tempdir().unwrap();
        let active = AtomicUsize::new(0);
        let ready = Barrier::new(8);
        let error = store_parallel(
            temp.path(),
            "rank/m.safetensors",
            8 << 20,
            &[(0, 8 << 20)],
            8,
            |index, _| {
                active.fetch_add(1, Ordering::SeqCst);
                ready.wait();
                Ok(Reader {
                    active: &active,
                    fail: index == 3,
                })
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("injected stream failure"));
        assert_eq!(active.load(Ordering::SeqCst), 0);
        assert!(!temp.path().join("packages/rank/m.safetensors").exists());
        assert!(
            !temp
                .path()
                .join("packages/rank/.m.safetensors.partial")
                .exists()
        );
    }

    #[test]
    fn parallel_blobs_verify_the_assembled_bytes_before_publication() {
        let temp = tempfile::tempdir().unwrap();
        let bytes = b"a payload split over eight streams with a tail";
        let blob = format!("{}.safetensors", super::hex(&Sha256::digest(bytes)));
        let name = format!("rank/{blob}");
        let size = bytes.len() as u64;
        for corrupt in [true, false] {
            let result =
                store_parallel(temp.path(), &name, size, &[(0, size)], 8, |index, part| {
                    let mut data = part
                        .iter()
                        .flat_map(|&(offset, count)| {
                            bytes[offset as usize..(offset + count) as usize]
                                .iter()
                                .copied()
                        })
                        .collect::<Vec<_>>();
                    if corrupt && index == 3 {
                        data[0] ^= 1;
                    }
                    Ok(std::io::Cursor::new(data))
                });
            if corrupt {
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("payload digest mismatch")
                );
                assert!(!temp.path().join("blobs").join(&blob).exists());
                assert!(!temp.path().join("packages").join(&name).exists());
            } else {
                let path = result.unwrap();
                assert_eq!(fs::read(&path).unwrap(), bytes);
                assert_eq!(fs::metadata(path).unwrap().permissions().mode() & 0o222, 0);
            }
        }
        assert!(
            store_parallel(temp.path(), &name, size, &[(0, 2)], 8, |_, _| Ok(
                std::io::empty()
            ))
            .is_err()
        );
    }

    #[test]
    fn snapshots_share_verified_blobs_and_eviction_preserves_them() {
        use std::os::unix::fs::MetadataExt;
        let temp = tempfile::tempdir().unwrap();
        let cache = temp.path();
        let bytes = b"immutable model payload";
        let blob = format!("{}.safetensors", super::hex(&Sha256::digest(bytes)));
        let first_name = format!("first/{blob}");
        let second_name = format!("second/{blob}");
        let size = bytes.len() as u64;
        let first = store(
            cache,
            "package",
            &first_name,
            size,
            &[(0, size)],
            &mut &bytes[..],
        )
        .unwrap();
        let second = have(cache, "package", &second_name, size).unwrap().unwrap();
        assert_eq!(
            fs::metadata(&first).unwrap().ino(),
            fs::metadata(&second).unwrap().ino()
        );
        assert_eq!(fs::metadata(&first).unwrap().mode() & 0o222, 0);
        evict(&cache.join("packages"), &Pins::default()).unwrap();
        assert!(!first.exists() && !second.exists());
        assert!(cache.join("blobs").join(&blob).exists());
        assert!(have(cache, "package", &first_name, size).unwrap().is_some());
        assert!(
            store(
                cache,
                "package",
                &second_name,
                size,
                &[(0, 4)],
                &mut &bytes[..4]
            )
            .is_err()
        );
        let bad = vec![b'x'; bytes.len()];
        assert!(
            store(
                cache,
                "package",
                &second_name,
                size,
                &[(0, size)],
                &mut &bad[..]
            )
            .is_err()
        );
        assert!(!second.exists());
    }
}
