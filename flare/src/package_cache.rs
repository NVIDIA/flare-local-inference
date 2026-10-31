/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

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

pub use crate::cache_support::model_root;
#[cfg(test)]
pub(crate) use crate::cache_support::{model_root_from, requested_tensor_parallel, select_deployments};

/// Install the selected deployment's ready closure into the configured model cache.
///
/// Verification, payload hashing, primary rewriting, publication, and the
/// eviction-safety lease all live in `flare_helios::package::cache`.
#[cfg(test)]
pub fn install_selected(
    verified: flare_helios::package::protection::VerifiedPackage,
    metadata: flare_helios::package::reader::Metadata,
    root: &Path,
    model_key: &str,
    entry: &str,
) -> Result<flare_helios::package::cache::Entry, Failure> {
    let store = flare_helios::package::cache::Root::open(root)?;
    let source = flare_helios::package::cache::PackageSource::new(metadata, verified)?;
    Ok(store.install_package(source, model_key, entry)?)
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
    #[cfg(test)]
    pub fn shared(directory: &Path) -> Result<Self, Failure> {
        let file = Self::open(directory)?;
        let lease = Self(file);
        lease.lock(libc::LOCK_SH)?;
        Ok(lease)
    }

    /// Take the write lease directly. Upgrading a shared lease can deadlock against another
    /// holder of the same directory, so writers open an exclusive lock instead.
    pub fn write(directory: &Path) -> Result<Self, Failure> {
        let lease = Self(Self::open(directory)?);
        lease.lock(libc::LOCK_EX)?;
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
    // Package companions may nest canonical paths; binaries stay a single validated component.
    if (directory == "packages" && components.len() < 2)
        || (directory != "packages" && components.len() != 1)
        || !components.iter().all(|part| component_valid(part))
    {
        return Err(format!("invalid cache entry name {name:?}").into());
    }
    Ok(cache.join(directory).join(name))
}

/// The unit eviction removes: a package directory, or an executable file.
pub fn unit(cache: &Path, path: &Path) -> PathBuf {
    let packages = cache.join("packages");
    if let Some(stripped) = path.strip_prefix(&packages).ok() {
        let components: Vec<_> = stripped.components().collect();
        if components.len() >= 2 {
            let mut unit = packages;
            unit.push(components[0]);
            return unit;
        }
    }
    let _ = packages;
    path.to_owned()
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

/// Direct children owned by `flare_helios::package::cache::Root`.
fn is_model_cache_directory(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some("blobs" | "models" | "refs" | "sources" | "origins" | "staging" | "locks")
    )
}

/// Remove unpinned cache units, leaving transfers in progress and other package-cache data alone.
pub fn evict(directory: &Path, kind: &str, pins: &Pins) -> Result<Vec<PathBuf>, Failure> {
    let package_cache = kind_directory(kind)? == "packages";
    let pins = pins.lock().unwrap();
    let mut removed = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let hidden = path
            .file_name()
            .is_none_or(|name| name.to_string_lossy().starts_with('.'));
        if package_cache && is_model_cache_directory(&path) {
            continue;
        }
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
        assert!(evict(&directory, "package", &pins).unwrap().is_empty());
        assert!(package.is_dir());
        drop(lease);
        assert_eq!(
            evict(&directory, "package", &pins).unwrap(),
            vec![package.clone()]
        );
        assert!(!package.exists());
    }

    #[test]
    fn agent_eviction_preserves_the_prepared_model_cache() {
        let temporary = tempfile::tempdir().unwrap();
        let packages = temporary.path().join("packages");
        let model = packages.join("models/model-key/entry-id/model.safetensors");
        let blob = packages.join("blobs/0123456789abcdef.safetensors");
        let reference = packages.join("refs/model-key");
        fs::create_dir_all(model.parent().unwrap()).unwrap();
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        fs::create_dir_all(reference.parent().unwrap()).unwrap();
        fs::write(&model, b"prepared model").unwrap();
        fs::write(&blob, b"shared payload").unwrap();
        fs::write(&reference, b"entry-id\n").unwrap();
        let other_model_cache_files = [
            "sources/selector",
            "origins/version",
            "staging/transaction",
            "locks/model-key",
        ];
        for relative in other_model_cache_files {
            let path = packages.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, b"prepared model metadata").unwrap();
        }

        let peer_package = packages.join("export-8-rank1");
        let complete_package = packages.join("export-4");
        fs::create_dir_all(&peer_package).unwrap();
        fs::write(peer_package.join("model.safetensors"), b"peer package").unwrap();
        fs::create_dir_all(&complete_package).unwrap();
        fs::write(
            complete_package.join("model.safetensors"),
            b"complete package",
        )
        .unwrap();

        let removed = evict(&packages, "package", &Pins::default()).unwrap();

        assert!(removed.contains(&peer_package));
        assert!(removed.contains(&complete_package));
        assert!(!peer_package.exists());
        assert!(!complete_package.exists());
        assert!(model.is_file());
        assert!(blob.is_file());
        assert!(reference.is_file());
        for relative in other_model_cache_files {
            assert!(packages.join(relative).is_file());
        }
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
            "export-1/weights/layer-000.safetensors",
            "export-1/deployments/sm120-tp1/rank-0/graphs.safetensors",
        ] {
            assert!(entry(cache, "packages", name).is_ok(), "{name}");
        }
        for name in ["a", "a/../escape", "a/", "a/hidden\\name", "a/.hidden"] {
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
        let removed = evict(&cache.join("packages"), "package", &pins).unwrap();
        assert!(removed.is_empty() && path.exists());
        drop(pinned);
        let removed = evict(&cache.join("packages"), "package", &pins).unwrap();
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
        }
        // Physical sparseness is checked on a separate workload with a large
        // untouched interior hole. The original transfer above covers the full
        // parallel/range workload but tmpfs can allocate its large written span
        // at huge-page granularity, making that path an invalid st_blocks probe.
        let sparse_size = 1u64 << 30;
        let sparse_ranges = [(0u64, 4096u64), (sparse_size - 4096, 4096)];
        let sparse_path = store_parallel(
            temp.path(),
            "sparse/m.safetensors",
            sparse_size,
            &sparse_ranges,
            1,
            |_, part| {
                Ok(std::io::Cursor::new(vec![
                    0u8;
                    part.iter()
                        .map(|(_, n)| *n as usize)
                        .sum()
                ]))
            },
        )
        .unwrap();
        let sparse_blocks = fs::metadata(&sparse_path).unwrap().blocks() * 512;
        assert!(
            sparse_blocks < sparse_size,
            "sparse allocation {sparse_blocks} must be less than {sparse_size}"
        );
        let sparse_bytes = fs::read(&sparse_path).unwrap();
        assert_eq!(&sparse_bytes[..4096], &vec![0; 4096]);
        assert_eq!(&sparse_bytes[sparse_size as usize - 4096..], &vec![0; 4096]);
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
        evict(&cache.join("packages"), "package", &Pins::default()).unwrap();
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

    #[test]
    fn installs_selected_deployment_payloads_into_model_cache() {
        use flare_helios::package::{
            PayloadId, cache as model_cache, export, protection, reader, schema::wire,
        };
        use flare_helios::runtime_package::Publisher;
        use ring::signature::{Ed25519KeyPair, KeyPair};

        let scratch = std::env::temp_dir().join(format!("flare-install-{}", std::process::id()));
        let _ = fs::remove_dir_all(&scratch);
        let source = scratch.join("source");
        let root = scratch.join("cache");
        fs::create_dir_all(&source).unwrap();

        let bytes = b"layer zero weights".repeat(10);
        let id = PayloadId::of(&bytes);
        let raw = scratch.join("layer-000.bin");
        fs::write(&raw, &bytes).unwrap();

        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
        let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Test".into(),
            namespace: "test".into(),
            public_key,
            development: false,
        };
        let package_id = [7u8; 32];

        let rank = wire::RankBody {
            rank: 0,
            deployment_id: "sm120-tp1".into(),
            context: Some(Default::default()),
            allocations: vec![wire::Allocation {
                name: "device-0".into(),
                nbytes: bytes.len() as u64,
                alignment: 4096,
                ..Default::default()
            }],
            graphs: vec![wire::Graph {
                name: "decode".into(),
                resource: 0,
                node_count: 1,
                edge_count: 0,
                ..Default::default()
            }],
            initializers: vec![wire::Initializer {
                allocation: "device-0".into(),
                destination_offset: 0,
                source: Some(wire::PayloadUse {
                    slice: Some(wire::PayloadSlice {
                        payload_id: id.as_bytes().to_vec(),
                        offset: 0,
                        nbytes: bytes.len() as u64,
                    }),
                    role: wire::PayloadRole::Weights as i32,
                }),
            }],
            ..Default::default()
        };
        let graph_bytes = scratch.join("graph.bin");
        fs::write(&graph_bytes, b"graph-bytes").unwrap();
        let graph_nbytes = 11u64;
        let runtime = flare_helios::runtime_package::wire::ServingContract {
            model_id: "test-model".into(),
            identity: "test-model-sm120".into(),
            max_sequence_tokens: 8,
            prefill_chunk_size: 1,
            pages: 1,
            state_rows: 1,
            page_tokens: 1,
            model_contract: Some(flare_helios::runtime_package::wire::ModelContract {
                vocabulary: 8,
                graphs: vec![flare_helios::runtime_package::wire::GraphContract {
                    graph: "decode".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let deployment = wire::DeploymentBody {
            deployment_id: "sm120-tp1".into(),
            architecture: vec![12, 0],
            tensor_parallel: 1,
            contract: Some(runtime),
            ..Default::default()
        };
        let request = export::PublishRequest {
            package: wire::PackageBody {
                model_identity: "test-model".into(),
                hardware_identity: "sm120".into(),
                frontend: Some(flare_helios::runtime_package::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    flare_helios::runtime_package::wire::DefaultSamplingParameters::default(),
                ),
                publisher: Some(wire::PublisherMetadata {
                    namespace: "test".into(),
                    official_source: "https://huggingface.co/test/test-model".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            payloads: vec![],
            deployments: vec![export::PublishDeployment {
                payloads: vec![],
                body: deployment,
                ranks: vec![export::PublishRank {
                    body: rank,
                    payloads: vec![export::PublishPayload {
                        role: wire::PayloadRole::Weights,
                        relative: "weights/layer-000.safetensors".into(),
                        tensor: Some("flare.payload".into()),
                        source: flare_helios::package::container::Slice::new(
                            std::sync::Arc::new(std::fs::File::open(&raw).unwrap()),
                            0,
                            (bytes.len() as u64) as u64,
                        )
                        .unwrap(),
                        id,
                    }],
                    resources: vec![export::PublishResource {
                        kind: wire::ExecutableKind::Graph,
                        path: graph_bytes,
                        nbytes: (graph_nbytes) as u64,
                    }],
                }],
            }],
            package_id,
            public_key,
            accepted: "test".into(),
            no_weights: false,
        };
        let published =
            export::publish(&request, &key, export::Destination::Directory(&source)).unwrap();
        let verified = protection::verify(
            &published.manifest,
            &[publisher],
            protection::Selection::Deployment {
                deployment: 0,
                rank: None,
            },
        )
        .unwrap();
        let metadata = reader::Metadata::open(&source).unwrap();
        let entry = install_selected(verified, metadata, &root, "test-model", "tp1").unwrap();
        assert!(entry.directory().join("model.safetensors").is_file());
        assert!(
            entry
                .directory()
                .join("weights/layer-000.safetensors")
                .is_file()
        );
        assert!(entry.payload("weights/layer-000.safetensors").is_ok());

        let store = model_cache::Root::open(&root).unwrap();
        let resolved = store.resolve("test-model").unwrap().unwrap();
        assert_eq!(resolved.id(), "tp1");
        assert_eq!(
            resolved
                .payload("weights/layer-000.safetensors")
                .unwrap()
                .len(),
            bytes.len() as u64
        );
        let object = store.payload(id).unwrap().unwrap();
        assert_eq!(
            object.path(),
            store.path().join("blobs").join(id.file_name())
        );
        drop(resolved);
        let _ = fs::remove_dir_all(&scratch);
    }

    #[test]
    fn packed_import_installs_semantic_companions_and_ignores_stray_frontend_tensors() {
        use flare_helios::package::{
            PayloadId, container, conversion, export, protection, reader,
            schema::wire,
        };
        use flare_helios::runtime_package::Publisher;
        use ring::signature::{Ed25519KeyPair, KeyPair};

        let scratch = std::env::temp_dir().join(format!("flare-packed-{}", std::process::id()));
        let _ = fs::remove_dir_all(&scratch);
        let source = scratch.join("source");
        let root = scratch.join("cache");
        fs::create_dir_all(&source).unwrap();
        let bytes = b"packed layer weights".repeat(9);
        let id = PayloadId::of(&bytes);
        let raw = scratch.join("layer.bin");
        fs::write(&raw, &bytes).unwrap();
        let graph = scratch.join("graph.bin");
        fs::write(&graph, b"graph-bytes").unwrap();

        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
        let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Test".into(),
            namespace: "test".into(),
            public_key,
            development: false,
        };
        let runtime = flare_helios::runtime_package::wire::ServingContract {
            model_id: "packed".into(),
            identity: "packed-sm120".into(),
            max_sequence_tokens: 8,
            prefill_chunk_size: 1,
            pages: 1,
            state_rows: 1,
            page_tokens: 1,
            model_contract: Some(flare_helios::runtime_package::wire::ModelContract {
                vocabulary: 8,
                graphs: vec![flare_helios::runtime_package::wire::GraphContract {
                    graph: "decode".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let rank = wire::RankBody {
            rank: 0,
            deployment_id: "sm120-tp1".into(),
            context: Some(Default::default()),
            allocations: vec![wire::Allocation {
                name: "device-0".into(),
                nbytes: bytes.len() as u64,
                alignment: 4096,
                ..Default::default()
            }],
            initializers: vec![wire::Initializer {
                allocation: "device-0".into(),
                destination_offset: 0,
                source: Some(wire::PayloadUse {
                    slice: Some(wire::PayloadSlice {
                        payload_id: id.as_bytes().to_vec(),
                        offset: 0,
                        nbytes: bytes.len() as u64,
                    }),
                    role: wire::PayloadRole::Weights as i32,
                }),
            }],
            graphs: vec![wire::Graph {
                name: "decode".into(),
                resource: 0,
                node_count: 1,
                ..Default::default()
            }],
            ..Default::default()
        };
        let deployment = wire::DeploymentBody {
            deployment_id: "sm120-tp1".into(),
            architecture: vec![12, 0],
            tensor_parallel: 1,
            contract: Some(runtime),
            ..Default::default()
        };
        let request = export::PublishRequest {
            package: wire::PackageBody {
                model_identity: "packed".into(),
                hardware_identity: "sm120".into(),
                frontend: Some(flare_helios::runtime_package::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    flare_helios::runtime_package::wire::DefaultSamplingParameters::default(),
                ),
                publisher: Some(wire::PublisherMetadata {
                    namespace: "test".into(),
                    official_source: "https://huggingface.co/test/packed".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            payloads: vec![],
            deployments: vec![export::PublishDeployment {
                payloads: vec![],
                body: deployment,
                ranks: vec![export::PublishRank {
                    body: rank,
                    payloads: vec![export::PublishPayload {
                        role: wire::PayloadRole::Weights,
                        relative: "weights/layer-000.safetensors".into(),
                        tensor: Some(container::COMPANION_TENSOR.into()),
                        source: flare_helios::package::container::Slice::new(
                            std::sync::Arc::new(std::fs::File::open(&raw).unwrap()),
                            0,
                            (bytes.len() as u64) as u64,
                        )
                        .unwrap(),
                        id,
                    }],
                    resources: vec![export::PublishResource {
                        kind: wire::ExecutableKind::Graph,
                        path: graph,
                        nbytes: (11) as u64,
                    }],
                }],
            }],
            package_id: [9u8; 32],
            public_key,
            accepted: "test".into(),
            no_weights: false,
        };
        let published =
            export::publish(&request, &key, export::Destination::Directory(&source)).unwrap();

        // An unrelated tensor in the untrusted primary header grants nothing.
        let primary_path = source.join("model.safetensors");
        let primary = container::Container::open(&primary_path).unwrap();
        let mut output = Vec::new();
        let mut writer = container::Writer::new(&mut output, container::PAYLOAD_FORMAT).unwrap();
        for name in primary.tensor_names() {
            if name.starts_with("align.") {
                continue;
            }
            let slice = primary.tensor(name).unwrap();
            writer
                .add(name, slice.len(), container::Source::Slice(slice))
                .unwrap();
        }
        writer
            .add(
                "frontend.fake",
                8,
                container::Source::Bytes(b"not-real".to_vec()),
            )
            .unwrap();
        writer.finish().unwrap();
        drop(primary);
        fs::write(&primary_path, &output).unwrap();

        let packed = scratch.join("packed.safetensors");
        conversion::pack(&source, &packed).unwrap();
        let metadata = reader::Metadata::open(&packed).unwrap();
        let verified = protection::verify(
            &metadata.envelope,
            &[publisher],
            protection::Selection::Deployment {
                deployment: 0,
                rank: None,
            },
        )
        .unwrap();
        assert!(verified.matches_envelope(&metadata.envelope));
        assert_eq!(published.payloads.len(), 2);

        let entry = install_selected(verified, metadata, &root, "packed", "tp1").unwrap();
        assert!(
            entry
                .directory()
                .join("weights/layer-000.safetensors")
                .is_file()
        );
        let installed =
            container::Container::open(&entry.directory().join("model.safetensors")).unwrap();
        assert!(installed.tensor_names().all(|name| name != "frontend.fake"));
        assert!(
            entry
                .payload("weights/layer-000.safetensors")
                .unwrap()
                .len()
                == bytes.len() as u64
        );
        let _ = fs::remove_dir_all(&scratch);
    }

    #[test]
    fn source_complete_packages_materialize_selected_targets_before_ready() {
        use flare_helios::package::{
            PayloadId, cache as model_cache, container, conversion, derivation, export, protection,
            reader, schema::wire,
        };
        use flare_helios::runtime_package::Publisher;
        use ring::signature::{Ed25519KeyPair, KeyPair};

        let scratch = std::env::temp_dir().join(format!("flare-derived-{}", std::process::id()));
        let _ = fs::remove_dir_all(&scratch);
        let source = scratch.join("source");
        fs::create_dir_all(&source).unwrap();

        let weights = b"derivable weight payload bytes".to_vec();
        let target = weights[..weights.len() - 6].to_vec();
        let weight_id = PayloadId::of(&weights);
        let target_id = PayloadId::of(&target);
        let weight_path = scratch.join("weights.bin");
        fs::write(&weight_path, &weights).unwrap();
        let target_path = scratch.join("target.bin");
        fs::write(&target_path, &target).unwrap();

        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
        let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Test".into(),
            namespace: "test".into(),
            public_key,
            development: false,
        };
        let runtime = flare_helios::runtime_package::wire::ServingContract {
            model_id: "derived".into(),
            identity: "derived-sm120".into(),
            max_sequence_tokens: 8,
            prefill_chunk_size: 1,
            pages: 1,
            state_rows: 1,
            page_tokens: 1,
            model_contract: Some(flare_helios::runtime_package::wire::ModelContract {
                vocabulary: 8,
                graphs: vec![flare_helios::runtime_package::wire::GraphContract {
                    graph: "decode".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let initializers = [(weight_id, weights.len()), (target_id, target.len())]
            .iter()
            .enumerate()
            .map(|(index, (id, nbytes))| wire::Initializer {
                allocation: format!("device-{index}"),
                destination_offset: 0,
                source: Some(wire::PayloadUse {
                    slice: Some(wire::PayloadSlice {
                        payload_id: id.as_bytes().to_vec(),
                        offset: 0,
                        nbytes: *nbytes as u64,
                    }),
                    role: wire::PayloadRole::Weights as i32,
                }),
            })
            .collect();
        let rank = wire::RankBody {
            rank: 0,
            deployment_id: "sm120-tp1".into(),
            context: Some(Default::default()),
            allocations: vec![
                wire::Allocation {
                    name: "device-0".into(),
                    nbytes: weights.len() as u64,
                    alignment: 4096,
                    ..Default::default()
                },
                wire::Allocation {
                    name: "device-1".into(),
                    nbytes: target.len() as u64,
                    alignment: 4096,
                    ..Default::default()
                },
            ],
            initializers,
            graphs: vec![wire::Graph {
                name: "decode".into(),
                resource: 0,
                node_count: 1,
                ..Default::default()
            }],
            ..Default::default()
        };
        let graph = scratch.join("graph.bin");
        fs::write(&graph, b"graph-bytes").unwrap();
        let request = export::PublishRequest {
            package: wire::PackageBody {
                model_identity: "derived".into(),
                hardware_identity: "sm120".into(),
                frontend: Some(flare_helios::runtime_package::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    flare_helios::runtime_package::wire::DefaultSamplingParameters::default(),
                ),
                publisher: Some(wire::PublisherMetadata {
                    namespace: "test".into(),
                    official_source: "https://huggingface.co/test/derived".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            payloads: vec![],
            deployments: vec![export::PublishDeployment {
                payloads: vec![],
                body: wire::DeploymentBody {
                    deployment_id: "sm120-tp1".into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 1,
                    contract: Some(runtime),
                    ..Default::default()
                },
                ranks: vec![export::PublishRank {
                    body: rank,
                    payloads: vec![
                        export::PublishPayload {
                            role: wire::PayloadRole::Weights,
                            relative: "weights/layer-000.safetensors".into(),
                            tensor: Some(container::COMPANION_TENSOR.into()),
                            source: flare_helios::package::container::Slice::new(
                                std::sync::Arc::new(std::fs::File::open(&weight_path).unwrap()),
                                0,
                                (weights.len() as u64) as u64,
                            )
                            .unwrap(),
                            id: weight_id,
                        },
                        export::PublishPayload {
                            role: wire::PayloadRole::Weights,
                            relative: "weights/state.safetensors".into(),
                            tensor: Some(container::COMPANION_TENSOR.into()),
                            source: flare_helios::package::container::Slice::new(
                                std::sync::Arc::new(std::fs::File::open(&target_path).unwrap()),
                                0,
                                (target.len() as u64) as u64,
                            )
                            .unwrap(),
                            id: target_id,
                        },
                    ],
                    resources: vec![export::PublishResource {
                        kind: wire::ExecutableKind::Graph,
                        path: graph,
                        nbytes: (11) as u64,
                    }],
                }],
            }],
            package_id: [11u8; 32],
            public_key,
            accepted: "test".into(),
            no_weights: false,
        };
        export::publish(&request, &key, export::Destination::Directory(&source)).unwrap();

        // The production combiner path proves the recipe against captured bytes and
        // then removes the target's location and companion.
        let recipe = derivation::Derivation {
            target: target_id,
            length: target.len() as u64,
            operations: vec![derivation::Operation::Copy(derivation::Copy {
                source: weight_id,
                source_offset: 0,
                destination_offset: 0,
                contiguous: target.len() as u64,
                outer: vec![],
            })],
        };
        let limits = derivation::Limits::default();
        let wrong = derivation::Derivation {
            operations: vec![derivation::Operation::Copy(derivation::Copy {
                source: weight_id,
                source_offset: 1,
                destination_offset: 0,
                contiguous: target.len() as u64,
                outer: vec![],
            })],
            ..recipe.clone()
        };
        assert!(
            conversion::emit_derivations(&source, std::slice::from_ref(&wrong), &limits).is_err()
        );
        assert_eq!(
            conversion::emit_derivations(&source, std::slice::from_ref(&recipe), &limits).unwrap(),
            1
        );
        assert!(!source.join("weights/state.safetensors").exists());

        let root = scratch.join("cache");
        let metadata = reader::Metadata::open(&source).unwrap();
        let verified = protection::verify(
            &metadata.envelope,
            &[publisher],
            protection::Selection::Deployment {
                deployment: 0,
                rank: None,
            },
        )
        .unwrap();
        let entry = install_selected(verified, metadata, &root, "derived", "tp1").unwrap();
        assert!(
            entry
                .directory()
                .join("weights/tuned/sm120-tp1/derived-000.safetensors")
                .is_file()
        );
        assert_eq!(
            entry
                .payload("weights/tuned/sm120-tp1/derived-000.safetensors")
                .unwrap()
                .len(),
            target.len() as u64
        );
        let store = model_cache::Root::open(&root).unwrap();
        assert!(store.payload(target_id).unwrap().is_some());
        drop(entry);
        assert_eq!(store.entry("derived", "tp1").unwrap().id(), "tp1");
        let _ = fs::remove_dir_all(&scratch);
    }

    #[test]
    fn union_acquisition_uses_recipes_from_both_selected_deployments() {
        use flare_helios::package::{
            PayloadId, cache as model_cache, container, conversion, derivation, export, protection,
            reader, schema::wire,
        };
        use flare_helios::runtime_package::Publisher;
        use ring::signature::{Ed25519KeyPair, KeyPair};

        let scratch = std::env::temp_dir().join(format!("flare-union-{}", std::process::id()));
        let _ = fs::remove_dir_all(&scratch);
        let source = scratch.join("source");
        fs::create_dir_all(&source).unwrap();

        let w1 = b"first deployment source bytes".to_vec();
        let w2 = b"second deployment source bytes".to_vec();
        let t1 = w2[..w2.len() - 4].to_vec();
        let t2 = w2[1..w2.len() - 3].to_vec();
        let (w1_id, w2_id) = (PayloadId::of(&w1), PayloadId::of(&w2));
        let (t1_id, t2_id) = (PayloadId::of(&t1), PayloadId::of(&t2));
        assert_ne!(t1_id, t2_id);
        let files = [
            ("w1", &w1, w1_id),
            ("w2", &w2, w2_id),
            ("t1", &t1, t1_id),
            ("t2", &t2, t2_id),
        ];
        let mut paths = Vec::new();
        for (name, bytes, _) in &files {
            let path = scratch.join(format!("{name}.bin"));
            fs::write(&path, bytes).unwrap();
            paths.push(path);
        }

        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
        let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Test".into(),
            namespace: "test".into(),
            public_key,
            development: false,
        };
        let runtime = |id: &str| flare_helios::runtime_package::wire::ServingContract {
            model_id: "union".into(),
            identity: format!("union-{id}"),
            max_sequence_tokens: 8,
            prefill_chunk_size: 1,
            pages: 1,
            state_rows: 1,
            page_tokens: 1,
            model_contract: Some(flare_helios::runtime_package::wire::ModelContract {
                vocabulary: 8,
                graphs: vec![flare_helios::runtime_package::wire::GraphContract {
                    graph: "decode".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let rank_body =
            |deployment: &str, rank: u32, pair: (PayloadId, u64), extra: (PayloadId, u64)| {
                let mut initializers = Vec::new();
                for (index, (id, nbytes)) in [pair, extra].iter().enumerate() {
                    initializers.push(wire::Initializer {
                        allocation: format!("device-{index}"),
                        destination_offset: 0,
                        source: Some(wire::PayloadUse {
                            slice: Some(wire::PayloadSlice {
                                payload_id: id.as_bytes().to_vec(),
                                offset: 0,
                                nbytes: *nbytes,
                            }),
                            role: wire::PayloadRole::Weights as i32,
                        }),
                    });
                }
                wire::RankBody {
                    rank,
                    deployment_id: deployment.into(),
                    context: Some(Default::default()),
                    allocations: (0..2)
                        .map(|index| wire::Allocation {
                            name: format!("device-{index}"),
                            nbytes: if index == 0 { pair.1 } else { extra.1 },
                            alignment: 4096,
                            ..Default::default()
                        })
                        .collect(),
                    initializers,
                    graphs: vec![wire::Graph {
                        name: "decode".into(),
                        resource: 0,
                        node_count: 1,
                        ..Default::default()
                    }],
                    ..Default::default()
                }
            };
        let graph1 = scratch.join("graph1.bin");
        fs::write(&graph1, b"graph-one!!").unwrap();
        let graph2 = scratch.join("graph2.bin");
        fs::write(&graph2, b"graph-two!!").unwrap();
        let payload =
            |relative: &str, path: &Path, nbytes: usize, id: PayloadId| export::PublishPayload {
                role: wire::PayloadRole::Weights,
                relative: relative.into(),
                tensor: Some(container::COMPANION_TENSOR.into()),
                source: flare_helios::package::container::Slice::new(
                    std::sync::Arc::new(std::fs::File::open(path).unwrap()),
                    0,
                    nbytes as u64,
                )
                .unwrap(),
                id,
            };
        let request = export::PublishRequest {
            package: wire::PackageBody {
                model_identity: "union".into(),
                hardware_identity: "sm120".into(),
                frontend: Some(flare_helios::runtime_package::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    flare_helios::runtime_package::wire::DefaultSamplingParameters::default(),
                ),
                publisher: Some(wire::PublisherMetadata {
                    namespace: "test".into(),
                    official_source: "https://huggingface.co/test/union".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            payloads: vec![],
            deployments: vec![
                export::PublishDeployment {
                    payloads: vec![],
                    body: wire::DeploymentBody {
                        deployment_id: "sm120-tp1".into(),
                        architecture: vec![12, 0],
                        tensor_parallel: 1,
                        contract: Some(runtime("tp1")),
                        ..Default::default()
                    },
                    ranks: vec![export::PublishRank {
                        body: rank_body(
                            "sm120-tp1",
                            0,
                            (w1_id, w1.len() as u64),
                            (t1_id, t1.len() as u64),
                        ),
                        payloads: vec![
                            payload("weights/tp1/w1.safetensors", &paths[0], w1.len(), w1_id),
                            payload("weights/tp1/t1.safetensors", &paths[2], t1.len(), t1_id),
                        ],
                        resources: vec![export::PublishResource {
                            kind: wire::ExecutableKind::Graph,
                            path: graph1,
                            nbytes: (11) as u64,
                        }],
                    }],
                },
                export::PublishDeployment {
                    payloads: vec![],
                    body: wire::DeploymentBody {
                        deployment_id: "sm120-tp2".into(),
                        architecture: vec![12, 0],
                        tensor_parallel: 2,
                        contract: Some(runtime("tp2")),
                        ..Default::default()
                    },
                    ranks: vec![
                        export::PublishRank {
                            body: rank_body(
                                "sm120-tp2",
                                0,
                                (w2_id, w2.len() as u64),
                                (t2_id, t2.len() as u64),
                            ),
                            payloads: vec![
                                payload("weights/tp2/w2.safetensors", &paths[1], w2.len(), w2_id),
                                payload("weights/tp2/t2.safetensors", &paths[3], t2.len(), t2_id),
                            ],
                            resources: vec![export::PublishResource {
                                kind: wire::ExecutableKind::Graph,
                                path: graph2.clone(),
                                nbytes: 11,
                            }],
                        },
                        export::PublishRank {
                            body: rank_body(
                                "sm120-tp2",
                                1,
                                (w2_id, w2.len() as u64),
                                (t2_id, t2.len() as u64),
                            ),
                            payloads: vec![],
                            resources: vec![export::PublishResource {
                                kind: wire::ExecutableKind::Graph,
                                path: graph2,
                                nbytes: (11) as u64,
                            }],
                        },
                    ],
                },
            ],
            package_id: [13u8; 32],
            public_key,
            accepted: "test".into(),
            no_weights: false,
        };
        export::publish(&request, &key, export::Destination::Directory(&source)).unwrap();

        let limits = derivation::Limits::default();
        let recipe = |target: PayloadId, source: PayloadId, length: u64| derivation::Derivation {
            target,
            length,
            operations: vec![derivation::Operation::Copy(derivation::Copy {
                source,
                source_offset: 0,
                destination_offset: 0,
                contiguous: length,
                outer: vec![],
            })],
        };
        // The TP1 target is derived from the TP2 source, so its recipe needs
        // authority from both selected deployments.
        let mut second = recipe(t2_id, w2_id, t2.len() as u64);
        if let derivation::Operation::Copy(copy) = &mut second.operations[0] {
            copy.source_offset = 1;
        }
        let recipes = vec![recipe(t1_id, w2_id, t1.len() as u64), second];
        assert_eq!(
            conversion::emit_derivations(&source, &recipes, &limits).unwrap(),
            2
        );

        let root = scratch.join("cache");
        let metadata = reader::Metadata::open(&source).unwrap();
        let sizes = crate::package_cache::requested_tensor_parallel(&[1], 1).unwrap();
        assert!(
            crate::package_cache::select_deployments(&metadata, &[publisher.clone()], &sizes)
                .is_err()
        );
        let _selections = crate::package_cache::select_deployments(
            &metadata,
            &[publisher.clone()],
            &[1, 2].iter().copied().collect::<Vec<_>>(),
        )
        .unwrap();
        let store = model_cache::Root::open(&root).unwrap();
        // The real union installs both selected deployments and both targets.
        let metadata = reader::Metadata::open(&source).unwrap();
        let all = crate::package_cache::requested_tensor_parallel(&[1, 2], 2).unwrap();
        let selections =
            crate::package_cache::select_deployments(&metadata, &[publisher.clone()], &all)
                .unwrap();
        let verified =
            protection::verify(&metadata.envelope, &[publisher.clone()], selections[0]).unwrap();
        let union = model_cache::PackageSource::new(metadata, verified)
            .unwrap()
            .union(&[publisher], &selections)
            .unwrap();
        let entry = store.install_package(union, "union", "tp1+tp2").unwrap();
        let directory = entry.directory();
        assert!(directory.join("weights/tp1/w1.safetensors").is_file());
        assert!(directory.join("weights/tp2/w2.safetensors").is_file());
        assert!(
            std::fs::read_dir(directory.join("weights/tuned/sm120-tp1"))
                .map(|mut dir| dir.next().is_some())
                .unwrap_or(false)
        );
        let _ = fs::remove_dir_all(&scratch);
    }

    #[test]
    fn model_keys_are_injective_and_keep_distinct_repositories_apart() {
        use flare_helios::package::cache::model_key;
        assert_eq!(model_key("org/model").unwrap(), "org_2Fmodel");
        assert_ne!(model_key("a/b").unwrap(), model_key("a--b").unwrap());
        assert_ne!(model_key("a%2Fb").unwrap(), model_key("a/b").unwrap());
        assert!(model_key("").is_err());
    }

    #[test]
    fn tensor_parallel_requests_default_deduplicate_and_reject_zero() {
        assert_eq!(requested_tensor_parallel(&[], 2).unwrap(), vec![2]);
        assert_eq!(requested_tensor_parallel(&[], 0).unwrap(), vec![1]);
        assert_eq!(
            requested_tensor_parallel(&[2, 1, 2], 8).unwrap(),
            vec![1, 2]
        );
        assert!(requested_tensor_parallel(&[0], 1).is_err());
    }

    #[test]
    fn model_root_prefers_explicit_then_configured_then_default() {
        let home = std::env::temp_dir().join(format!("flare-model-root-{}", std::process::id()));
        let _ = fs::remove_dir_all(&home);
        fs::create_dir_all(home.join(".config/flare")).unwrap();
        let config = home.join(".config/flare/config.toml");
        fs::write(
            &config,
            "[server]\n# package_cache = \"/commented\"\n[paths] # paths section\nhub_cache = '~/.cache/hf'\npackage_cache = \"/data/flare-blobs #1\"   # trailing comment\n",
        )
        .unwrap();

        assert_eq!(
            model_root_from(&home, Some(&config), None).unwrap(),
            PathBuf::from("/data/flare-blobs #1")
        );
        assert_eq!(
            model_root_from(&home, Some(&config), Some(Path::new("~/custom cache"))).unwrap(),
            home.join("custom cache")
        );
        assert_eq!(
            model_root_from(&home, None, None).unwrap(),
            home.join(".cache/flare/packages")
        );

        fs::write(&config, "[paths]\npackage_cache = \"\"\n").unwrap();
        assert_eq!(
            model_root_from(&home, Some(&config), None).unwrap(),
            home.join(".cache/flare/packages")
        );
        fs::write(&config, "[server]\nmodel = \"x\"\n").unwrap();
        fs::write(&config, "[paths]\npackage_cache = \"~/cached\"\n").unwrap();
        assert_eq!(
            model_root_from(&home, Some(&config), None).unwrap(),
            home.join("cached")
        );
        fs::write(&config, "[paths]\npackage_cache = 5\n").unwrap();
        assert!(model_root_from(&home, Some(&config), None).is_err());
        assert!(model_root_from(&home, Some(&config), Some(Path::new("/override"))).is_ok());
        let _ = fs::remove_dir_all(&home);
    }
}

#[cfg(test)]
mod entry_path_tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn nested_packages_validate_and_evict_at_the_package_directory() {
        let cache = Path::new("/cache");
        let nested = entry(
            cache,
            "packages",
            "export-1/deployments/sm120-tp1/rank-0/graphs.safetensors",
        )
        .unwrap();
        assert_eq!(
            unit(cache, &nested),
            Path::new("/cache/packages/export-1").to_path_buf()
        );
        assert!(entry(cache, "packages", "export-1/model.safetensors").is_ok());
        assert!(entry(cache, "packages", "underlying-payload/..").is_err());
        assert!(entry(cache, "packages", "../escape/file").is_err());
        assert!(entry(cache, "bin", "nested/tool").is_err());
        assert!(entry(cache, "bin", "tool").is_ok());
    }
}
