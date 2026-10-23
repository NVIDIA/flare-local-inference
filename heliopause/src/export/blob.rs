//! Immutable content-addressed payload publication with hash-before-write reuse.

use super::{
    ExportResult as Result,
    archive::{self, container_header},
    io::{self, Identity, IoBudget, Source},
};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, VecDeque},
    fs::File,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

pub(crate) const MAX_PUBLISHED_BLOB_BYTES: u64 = 64_000_000_000;
pub(crate) enum PreparedBlob {
    Reused(String),
    Write {
        size: u64,
        name: String,
        expected: [u8; 32],
        source_blob: Option<Source>,
    },
}

struct Verified {
    file: Option<Arc<File>>,
    identity: Identity,
}
struct VerifiedCache {
    entries: BTreeMap<String, Verified>,
    pinned: VecDeque<String>,
}
pub(crate) struct BlobStore {
    pub(crate) root: PathBuf,
    pub(crate) budget: Arc<IoBudget>,
    pub(crate) written: AtomicU64,
    pub(crate) reused: AtomicU64,
    verified: Mutex<VerifiedCache>,
}
impl BlobStore {
    pub(crate) fn new(cache: &Path, budget: Arc<IoBudget>) -> Result<Self> {
        let root = cache.join("blobs");
        std::fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            budget,
            written: AtomicU64::new(0),
            reused: AtomicU64::new(0),
            verified: Mutex::new(VerifiedCache {
                entries: BTreeMap::new(),
                pinned: VecDeque::new(),
            }),
        })
    }
    fn remember(&self, name: String, file: Arc<File>, identity: Identity) {
        let mut verified = self.verified.lock().unwrap();
        verified.pinned.retain(|old| old != &name);
        if verified.pinned.len() == 2 {
            let old = verified.pinned.pop_front().unwrap();
            if let Some(entry) = verified.entries.get_mut(&old) {
                entry.file = None;
            }
        }
        verified.pinned.push_back(name.clone());
        verified.entries.insert(
            name,
            Verified {
                file: Some(file),
                identity,
            },
        );
    }
    pub(crate) fn put(
        &self,
        size: u64,
        required: bool,
        source_blob: Option<&Source>,
        mut fill: impl FnMut(u64, &mut [u8]) -> Result<()>,
    ) -> Result<String> {
        let prepared = self.prepare(size, required, source_blob, &mut fill)?;
        self.commit(prepared, fill)
    }
    pub(crate) fn prepare(
        &self,
        size: u64,
        required: bool,
        source_blob: Option<&Source>,
        mut fill: impl FnMut(u64, &mut [u8]) -> Result<()>,
    ) -> Result<PreparedBlob> {
        if size == 0 || size > MAX_PUBLISHED_BLOB_BYTES - 4096 {
            return Err("package resource exceeds the 64 GB payload shard limit".into());
        }
        let header = container_header(size, 0, "flare-data")?;
        let expected = self.budget.hash_generated(size, &header, &mut fill)?;
        if let Some(source) = source_blob {
            source.check_unchanged()?;
        }
        let name = format!("{}.safetensors", hex(&expected));
        let target = self.root.join(&name);
        match io::open_read(&target) {
            Ok(file) => {
                let file = Arc::new(file);
                let cached = Identity::of(&file)?;
                let mut same = false;
                if let Some(source) = source_blob {
                    if source.offset == header.len() as u64
                        && source.length == size
                        && source.file.metadata()?.len() == header.len() as u64 + size
                        && Identity::of(&source.file)? == cached
                    {
                        let mut actual = vec![0; header.len()];
                        io::read_exact_at(&source.file, &mut actual, 0)?;
                        same = actual == header;
                        source.check_unchanged()?;
                    }
                }
                if !same {
                    self.verify_pinned(&name, file.clone(), expected, size + header.len() as u64)?;
                } else {
                    self.remember(name.clone(), file, cached);
                }
                self.reused.fetch_add(size, Ordering::Relaxed);
                return Ok(PreparedBlob::Reused(name));
            }
            Err(error)
                if error
                    .downcast_ref::<std::io::Error>()
                    .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound) => {}
            Err(error) => return Err(error),
        }
        if required {
            return Err(format!(
                "--no-weights requires cached payload {name}; export with weights first"
            )
            .into());
        }
        Ok(PreparedBlob::Write {
            size,
            name,
            expected,
            source_blob: source_blob.cloned(),
        })
    }
    pub(crate) fn commit(
        &self,
        prepared: PreparedBlob,
        mut fill: impl FnMut(u64, &mut [u8]) -> Result<()>,
    ) -> Result<String> {
        let PreparedBlob::Write {
            size,
            name,
            expected,
            source_blob,
        } = prepared
        else {
            let PreparedBlob::Reused(name) = prepared else {
                unreachable!()
            };
            return Ok(name);
        };
        let source_blob = source_blob.as_ref();
        if let Some(source) = source_blob {
            source.check_unchanged()?;
        }
        let header = container_header(size, 0, "flare-data")?;
        let target = self.root.join(&name);
        let (file, path) = archive::temporary(&self.root, "blob")?;
        let result: Result<String> = (|| {
            io::write_all_at(&file, &header, 0)?;
            let check =
                self.budget
                    .write_hashed(&file, header.len() as u64, size, &header, &mut fill)?;
            if check != expected {
                return Err("payload changed during export".into());
            }
            if let Some(source) = source_blob {
                source.check_unchanged()?;
            }
            readonly(&file)?;
            file.sync_all()?;
            let finalized = Identity::of(&file)?;
            drop(file);
            match archive::install_blob(&path, &target) {
                Ok(()) => {
                    let pinned = Arc::new(io::open_read(&target)?);
                    let identity = Identity::of(&pinned)?;
                    if !identity.same_content(&finalized) {
                        return Err("blob changed while installing cache entry".into());
                    }
                    self.remember(name.clone(), pinned, identity);
                    self.written.fetch_add(size, Ordering::Relaxed);
                }
                Err(e)
                    if e.downcast_ref::<std::io::Error>()
                        .is_some_and(|e| e.kind() == std::io::ErrorKind::AlreadyExists) =>
                {
                    let pinned = Arc::new(io::open_read(&target)?);
                    self.verify_pinned(&name, pinned, expected, size + header.len() as u64)?;
                    self.reused.fetch_add(size, Ordering::Relaxed);
                }
                Err(e) => return Err(e),
            }
            Ok(name)
        })();
        let removed = archive::remove_temporary(&path);
        let name = result?;
        match removed {
            Ok(()) => {}
            Err(e)
                if e.downcast_ref::<std::io::Error>()
                    .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound) => {}
            Err(e) => return Err(e),
        }
        Ok(name)
    }
    fn verify_pinned(
        &self,
        name: &str,
        file: Arc<File>,
        expected: [u8; 32],
        size: u64,
    ) -> Result<Identity> {
        let identity = Identity::of(&file)?;
        {
            let verified = self.verified.lock().unwrap();
            if let Some(previous) = verified.entries.get(name) {
                if previous.identity == identity
                    && match &previous.file {
                        Some(file) => Identity::of(file)? == identity,
                        None => true,
                    }
                {
                    return Ok(identity);
                }
            }
        }
        let identity = verify_handle(file.clone(), expected, size, &self.budget)?;
        self.remember(name.to_owned(), file, identity.clone());
        Ok(identity)
    }
    pub(crate) fn link(&self, name: &str, directory: &Path) -> Result<()> {
        if !archive::blob_name(name) {
            return Err("invalid payload blob name".into());
        }
        let expected = decode_digest(&name[..64])?;
        let source = self.root.join(name);
        let file = Arc::new(io::open_read(&source)?);
        let size = file.metadata()?.len();
        let before = self.verify_pinned(name, file.clone(), expected, size)?;
        let destination = directory.join(name);
        match std::fs::hard_link(&source, &destination) {
            Ok(()) => {
                let linked = io::open_read(&destination)?;
                if !same_inode(&file, &linked)? {
                    return Err("blob cache changed while installing snapshot link".into());
                }
                let mut after = Identity::of(&file)?;
                // Hard-link creation changes ctime, so rehash if any content identity field changed.
                if !same_content_identity(&before, &after) {
                    after = verify_handle(file.clone(), expected, size, &self.budget)?;
                }
                self.remember(name.to_owned(), file, after);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let linked = Arc::new(io::open_read(&destination)?);
                self.verify_pinned(name, linked, expected, size)?;
            }
            Err(e) if e.raw_os_error() == Some(18) => {
                return Err(
                    "package snapshot and blob cache must be on the same filesystem".into(),
                );
            }
            Err(e) => return Err(e.into()),
        }
        Ok(())
    }
    pub(crate) fn sync(&self) -> Result<()> {
        archive::sync_directory(&self.root)
    }
}

pub(crate) fn verify_file(
    path: &Path,
    expected: [u8; 32],
    size: u64,
    budget: &IoBudget,
) -> Result<Arc<File>> {
    let file = Arc::new(io::open_read(path)?);
    verify_handle(file.clone(), expected, size, budget)?;
    Ok(file)
}
fn verify_handle(
    file: Arc<File>,
    expected: [u8; 32],
    size: u64,
    budget: &IoBudget,
) -> Result<Identity> {
    verify_handle_checked(file, expected, size, budget, |_, _| Ok(()))
}
fn verify_handle_checked(
    file: Arc<File>,
    expected: [u8; 32],
    size: u64,
    budget: &IoBudget,
    mut after_scan: impl FnMut(usize, &File) -> Result<()>,
) -> Result<Identity> {
    for attempt in 0..3 {
        let before = Identity::of(&file)?;
        if file.metadata()?.len() != size {
            return Err("invalid cached blob length".into());
        }
        let source = Source::from_file(file.clone(), 0, size)?;
        let mut digest = Sha256::new();
        let scan = budget.scan(&source, |bytes| {
            digest.update(bytes);
            Ok(())
        });
        after_scan(attempt, &file)?;
        let after = Identity::of(&file)?;
        if before != after {
            continue;
        }
        scan?;
        if <[u8; 32]>::from(digest.finalize()) != expected {
            return Err("cached blob content does not match its hash".into());
        }
        return Ok(after);
    }
    Err("cached blob changed during verification".into())
}
fn readonly(file: &File) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o444))?;
    }
    #[cfg(windows)]
    {
        let mut permissions = file.metadata()?.permissions();
        permissions.set_readonly(true);
        file.set_permissions(permissions)?;
    }
    Ok(())
}
fn same_inode(first: &File, second: &File) -> Result<bool> {
    Ok(Identity::of(first)?.same_file(&Identity::of(second)?))
}
fn same_content_identity(first: &Identity, second: &Identity) -> bool {
    first.same_content(second)
}
pub(crate) fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push(DIGITS[(byte >> 4) as usize] as char);
        text.push(DIGITS[(byte & 15) as usize] as char);
    }
    text
}
fn decode_digest(text: &str) -> Result<[u8; 32]> {
    let mut digest = [0; 32];
    if text.len() != 64 {
        return Err("invalid blob digest".into());
    }
    for (i, pair) in text.as_bytes().chunks_exact(2).enumerate() {
        let digit = |b: u8| -> Result<u8> {
            match b {
                b'0'..=b'9' => Ok(b - b'0'),
                b'a'..=b'f' => Ok(b - b'a' + 10),
                _ => Err("invalid blob digest".into()),
            }
        };
        digest[i] = digit(pair[0])? * 16 + digit(pair[1])?;
    }
    Ok(digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    fn directory() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "flare-blob-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        path
    }
    #[test]
    fn cache_hits_require_one_pass_and_no_payload_writes() {
        let directory = directory();
        let store = BlobStore::new(&directory, IoBudget::for_ranks(1).unwrap()).unwrap();
        let mut contents = container_header(3, 0, "flare-data").unwrap();
        contents.extend_from_slice(b"abc");
        let expected: [u8; 32] = Sha256::digest(&contents).into();
        let name = format!("{}.safetensors", hex(&expected));
        std::fs::write(store.root.join(&name), contents).unwrap();
        let mut calls = 0;
        assert_eq!(
            store
                .put(3, true, None, |offset, bytes| {
                    calls += 1;
                    bytes.copy_from_slice(&b"abc"[offset as usize..offset as usize + bytes.len()]);
                    Ok(())
                })
                .unwrap(),
            name
        );
        assert_eq!(calls, 1);
        assert_eq!(store.written.load(Ordering::Relaxed), 0);
        assert_eq!(store.reused.load(Ordering::Relaxed), 3);
        assert_eq!(std::fs::read_dir(&store.root).unwrap().count(), 1);
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn newly_written_blob_links_without_a_verification_read() {
        let directory = directory();
        let output = directory.join("output");
        std::fs::create_dir(&output).unwrap();
        let budget = IoBudget::for_ranks(1).unwrap();
        let store = BlobStore::new(&directory, budget.clone()).unwrap();
        let size = super::super::io::BLOCK as u64 + 37;
        let mut calls = 0;
        let name = store
            .put(size, false, None, |offset, bytes| {
                calls += 1;
                for (i, byte) in bytes.iter_mut().enumerate() {
                    *byte = ((offset + i as u64) % 251) as u8;
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(calls, 4);
        assert_eq!(budget.scan_bytes.load(Ordering::Relaxed), 0);
        store.link(&name, &output).unwrap();
        assert_eq!(budget.scan_bytes.load(Ordering::Relaxed), 0);
        let cached = store.verified.lock().unwrap();
        assert_eq!(
            cached.entries[&name].identity,
            Identity::of(cached.entries[&name].file.as_ref().unwrap()).unwrap()
        );
        drop(cached);
        assert_eq!(std::fs::read_dir(&store.root).unwrap().count(), 1);
        drop(store);
        #[cfg(windows)]
        {
            let mut permissions = std::fs::metadata(output.join(&name)).unwrap().permissions();
            permissions.set_readonly(false);
            std::fs::set_permissions(output.join(&name), permissions).unwrap();
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn missing_required_and_corrupt_cache_never_publish() {
        let directory = directory();
        let store = BlobStore::new(&directory, IoBudget::for_ranks(1).unwrap()).unwrap();
        assert!(
            store
                .put(3, true, None, |_, bytes| {
                    bytes.copy_from_slice(b"abc");
                    Ok(())
                })
                .unwrap_err()
                .to_string()
                .contains("--no-weights")
        );
        assert_eq!(std::fs::read_dir(&store.root).unwrap().count(), 0);
        let mut contents = container_header(3, 0, "flare-data").unwrap();
        contents.extend_from_slice(b"abc");
        let name = format!("{}.safetensors", hex(&Sha256::digest(&contents)));
        contents[4096] = b'x';
        std::fs::write(store.root.join(name), contents).unwrap();
        assert!(
            store
                .put(3, false, None, |_, bytes| {
                    bytes.copy_from_slice(b"abc");
                    Ok(())
                })
                .unwrap_err()
                .to_string()
                .contains("hash")
        );
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn changed_second_pass_cannot_install_blob() {
        let directory = directory();
        let store = BlobStore::new(&directory, IoBudget::for_ranks(1).unwrap()).unwrap();
        let mut calls = 0;
        let error = store
            .put(3, false, None, |_, bytes| {
                calls += 1;
                bytes.fill(if calls == 1 { 1 } else { 2 });
                Ok(())
            })
            .unwrap_err();
        assert!(error.to_string().contains("changed"));
        assert_eq!(std::fs::read_dir(&store.root).unwrap().count(), 0);
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn link_collision_preserves_existing_destination() {
        let directory = directory();
        let output = directory.join("output");
        std::fs::create_dir(&output).unwrap();
        let store = BlobStore::new(&directory, IoBudget::for_ranks(1).unwrap()).unwrap();
        let name = store
            .put(3, false, None, |_, bytes| {
                bytes.copy_from_slice(b"abc");
                Ok(())
            })
            .unwrap();
        std::fs::write(output.join(&name), b"existing").unwrap();
        assert!(store.link(&name, &output).is_err());
        assert_eq!(std::fs::read(output.join(&name)).unwrap(), b"existing");
        drop(store);
        #[cfg(windows)]
        {
            let mut permissions = std::fs::metadata(directory.join("blobs").join(&name))
                .unwrap()
                .permissions();
            permissions.set_readonly(false);
            std::fs::set_permissions(directory.join("blobs").join(&name), permissions).unwrap();
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    #[cfg(unix)]
    fn verification_retries_hardlink_metadata_changes_and_bounds_churn() {
        let directory = directory();
        let path = directory.join("payload");
        std::fs::write(&path, b"abc").unwrap();
        let expected = Sha256::digest(b"abc").into();
        let budget = IoBudget::for_ranks(1).unwrap();
        let file = Arc::new(io::open_read(&path).unwrap());
        let mut attempts = 0;
        verify_handle_checked(file.clone(), expected, 3, &budget, |attempt, _| {
            attempts += 1;
            if attempt == 0 {
                std::fs::hard_link(&path, directory.join("peer-link"))?;
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(attempts, 2);
        let mut attempts = 0;
        assert!(
            verify_handle_checked(file.clone(), expected, 3, &budget, |attempt, _| {
                attempts += 1;
                std::fs::hard_link(&path, directory.join(format!("churn-{attempt}")))?;
                Ok(())
            })
            .unwrap_err()
            .to_string()
            .contains("changed")
        );
        assert_eq!(attempts, 3);
        drop(file);
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    #[cfg(unix)]
    fn verification_retries_then_rejects_changed_contents() {
        let directory = directory();
        let path = directory.join("payload");
        std::fs::write(&path, b"abc").unwrap();
        let budget = IoBudget::for_ranks(1).unwrap();
        let file = Arc::new(io::open_read(&path).unwrap());
        let error = verify_handle_checked(
            file.clone(),
            Sha256::digest(b"abc").into(),
            3,
            &budget,
            |attempt, _| {
                if attempt == 0 {
                    std::fs::write(&path, b"xyz")?;
                }
                Ok(())
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("hash"));
        drop(file);
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    #[cfg(unix)]
    fn verified_inode_mutation_is_rejected_before_link() {
        let directory = directory();
        let output = directory.join("output");
        std::fs::create_dir(&output).unwrap();
        let store = BlobStore::new(&directory, IoBudget::for_ranks(1).unwrap()).unwrap();
        let name = store
            .put(3, false, None, |_, bytes| {
                bytes.copy_from_slice(b"abc");
                Ok(())
            })
            .unwrap();
        let path = store.root.join(&name);
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let file = File::options().write(true).open(&path).unwrap();
        io::write_all_at(&file, b"xyz", 4096).unwrap();
        file.sync_all().unwrap();
        assert!(
            store
                .link(&name, &output)
                .unwrap_err()
                .to_string()
                .contains("hash")
        );
        assert!(!output.join(name).exists());
        drop(file);
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn verified_source_does_not_hide_a_different_corrupt_cache_inode() {
        let directory = directory();
        let store = BlobStore::new(&directory, IoBudget::for_ranks(1).unwrap()).unwrap();
        let mut valid = container_header(3, 0, "flare-data").unwrap();
        valid.extend_from_slice(b"abc");
        let name = format!("{}.safetensors", hex(&Sha256::digest(&valid)));
        let source_path = directory.join("source");
        std::fs::write(&source_path, &valid).unwrap();
        valid[4096] = b'x';
        std::fs::write(store.root.join(name), valid).unwrap();
        let source = Source::open(&source_path, 4096, 3).unwrap();
        assert!(
            store
                .put(3, true, Some(&source), |offset, bytes| source
                    .read_at(offset, bytes))
                .unwrap_err()
                .to_string()
                .contains("hash")
        );
        drop(source);
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn prepared_blob_defers_writes_and_verifies_the_commit_bytes() {
        let directory = directory();
        let store = BlobStore::new(&directory, IoBudget::for_ranks(1).unwrap()).unwrap();
        let prepared = store
            .prepare(3, false, None, |_, bytes| {
                bytes.fill(7);
                Ok(())
            })
            .unwrap();
        assert_eq!(std::fs::read_dir(&store.root).unwrap().count(), 0);
        assert!(
            store
                .commit(prepared, |_, bytes| {
                    bytes.fill(8);
                    Ok(())
                })
                .unwrap_err()
                .to_string()
                .contains("changed")
        );
        assert_eq!(std::fs::read_dir(&store.root).unwrap().count(), 0);
        let prepared = store
            .prepare(3, false, None, |_, bytes| {
                bytes.fill(7);
                Ok(())
            })
            .unwrap();
        let name = store
            .commit(prepared, |_, bytes| {
                bytes.fill(7);
                Ok(())
            })
            .unwrap();
        let ready = store
            .prepare(3, true, None, |_, bytes| {
                bytes.fill(7);
                Ok(())
            })
            .unwrap();
        assert_eq!(
            store
                .commit(ready, |_, _| panic!(
                    "cache hits must not read another pass"
                ))
                .unwrap(),
            name
        );
        drop(store);
        #[cfg(windows)]
        {
            let path = directory.join("blobs").join(name);
            let mut permissions = std::fs::metadata(&path).unwrap().permissions();
            permissions.set_readonly(false);
            std::fs::set_permissions(path, permissions).unwrap();
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    #[cfg(unix)]
    fn evicted_verified_handles_retain_identity_checks_without_descriptors() {
        let directory = directory();
        let budget = IoBudget::for_ranks(1).unwrap();
        let store = BlobStore::new(&directory, budget.clone()).unwrap();
        let first = store
            .put(3, false, None, |_, bytes| {
                bytes.fill(1);
                Ok(())
            })
            .unwrap();
        let retired = Arc::downgrade(
            store.verified.lock().unwrap().entries[&first]
                .file
                .as_ref()
                .unwrap(),
        );
        for value in 2..=8 {
            store
                .put(3, false, None, |_, bytes| {
                    bytes.fill(value);
                    Ok(())
                })
                .unwrap();
        }
        assert!(retired.upgrade().is_none());
        {
            let cache = store.verified.lock().unwrap();
            assert_eq!(cache.pinned.len(), 2);
            assert_eq!(
                cache
                    .entries
                    .values()
                    .filter(|entry| entry.file.is_some())
                    .count(),
                2
            );
            assert!(cache.entries[&first].file.is_none());
        }
        store
            .put(3, true, None, |_, bytes| {
                bytes.fill(1);
                Ok(())
            })
            .unwrap();
        assert_eq!(budget.scan_bytes.load(Ordering::Relaxed), 0);
        let path = store.root.join(first);
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let file = File::options().write(true).open(path).unwrap();
        io::write_all_at(&file, b"bad", 4096).unwrap();
        file.sync_all().unwrap();
        drop(file);
        assert!(
            store
                .put(3, true, None, |_, bytes| {
                    bytes.fill(1);
                    Ok(())
                })
                .unwrap_err()
                .to_string()
                .contains("hash")
        );
        assert!(budget.scan_bytes.load(Ordering::Relaxed) > 0);
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
