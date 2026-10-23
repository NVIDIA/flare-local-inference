//! Authenticate and relocate packages without CUDA initialization.
#[cfg(test)]
use super::archive::Container;
use super::{ExportOptions, ExportResult as Result, FilePart, FileSource, invalid, wire};
use super::{
    archive::{self, Layout, PackageSources},
    blob::BlobStore,
    io::IoBudget,
    package::{self, PublishStats},
    protection::{self, Publisher, PublisherRequest},
};
use helios_core::runtime_package::Publisher as Profile;
#[cfg(test)]
use helios_core::runtime_package::VerifiedPackage;
use prost::Message;
use rayon::prelude::*;
use ring::{aead, digest};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    path::Path,
    sync::{Arc, Mutex, atomic::Ordering},
};

#[derive(Clone)]
struct SharedSources(Arc<Mutex<PackageSources>>);
impl SharedSources {
    fn new(files: PackageSources) -> Self {
        Self(Arc::new(Mutex::new(files)))
    }
    fn length(&self, name: &str) -> Result<u64> {
        self.0
            .lock()
            .map_err(|_| invalid("package source lock poisoned"))?
            .logical_length(name)
    }
    fn sources(&self, name: &str, offset: u64, nbytes: u64) -> Result<Vec<archive::SourcePiece>> {
        self.0
            .lock()
            .map_err(|_| invalid("package source lock poisoned"))?
            .sources(name, offset, nbytes)
    }
    fn read_at(&self, budget: &IoBudget, name: &str, offset: u64, output: &mut [u8]) -> Result<()> {
        let mut cursor = 0;
        for piece in self.sources(name, offset, output.len() as u64)? {
            match piece {
                archive::SourcePiece::Data(source) => {
                    let n = source.length as usize;
                    budget.read_at(&source, 0, &mut output[cursor..cursor + n])?;
                    cursor += n;
                }
                archive::SourcePiece::Zero(n) => {
                    output[cursor..cursor + n as usize].fill(0);
                    cursor += n as usize;
                }
            }
        }
        Ok(())
    }
}

pub(crate) struct Authenticated {
    sources: SharedSources,
    pub index: wire::ModelIndex,
    pub envelope: Vec<u8>,
    pub package_id: Vec<u8>,
    pub publisher: Profile,
    pub namespace: String,
    pub license: Option<wire::License>,
    pub resources: BTreeMap<(u32, Option<usize>, u64), wire::ProtectedResource>,
    pub stored: BTreeMap<Option<usize>, wire::StoredData>,
}
pub(crate) fn authenticate(
    path: &Path,
    profiles: &[Profile],
    development_only: bool,
) -> Result<Authenticated> {
    let files = PackageSources::open(path)?;
    let container = &files.container;
    let payload_length = container
        .layout
        .as_ref()
        .map(|layout| layout.files[""].iter().map(|(n, _)| *n).sum())
        .unwrap_or(container.payload_length);
    let (index, publisher) = helios_core::runtime_package::verify_package_metadata(
        &container.envelope,
        profiles,
        payload_length,
    )?;
    let envelope = wire::ProtectedIndex::decode(container.envelope.as_slice())?;
    let metadata = wire::PublisherMetadata::decode(envelope.groups[0].body.as_slice())?;
    if development_only
        && (!publisher.development
            || publisher.namespace != "local-development"
            || metadata.official_source != "local-development")
    {
        return Err(invalid("only local-development packages can be promoted"));
    }
    let shared = wire::ResourceCatalog::decode(envelope.groups[1].body.as_slice())?;
    let mut resources = BTreeMap::new();
    for resource in shared.resources {
        resources.insert((3, None, resource.offset), resource);
    }
    let mut stored = BTreeMap::from([(
        None,
        wire::StoredData::decode(envelope.groups[2].body.as_slice())?,
    )]);
    for (rank, group) in envelope.ranks.iter().enumerate() {
        let signed = wire::RankProtection::decode(group.body.as_slice())?;
        stored.insert(
            Some(rank),
            signed
                .data
                .ok_or_else(|| invalid("missing rank stored-data description"))?,
        );
        for (kind, catalog) in [(2, signed.executables), (3, signed.support)] {
            for resource in catalog.unwrap_or_default().resources {
                resources.insert((kind, Some(rank), resource.offset), resource);
            }
        }
    }
    package::roles(&index)?;
    Ok(Authenticated {
        index,
        envelope: container.envelope.clone(),
        package_id: envelope.package_id,
        publisher,
        namespace: metadata.namespace,
        license: metadata.license,
        resources,
        stored,
        sources: SharedSources::new(files),
    })
}
fn hash_range(
    files: &SharedSources,
    budget: &IoBudget,
    name: &str,
    offset: u64,
    length: u64,
    context: &mut digest::Context,
    buffer: &mut [u8],
) -> Result<()> {
    let mut position = 0;
    while position < length {
        let count = (length - position).min(buffer.len() as u64) as usize;
        files.read_at(budget, name, offset + position, &mut buffer[..count])?;
        context.update(&buffer[..count]);
        position += count as u64;
    }
    Ok(())
}
fn zero_range(
    files: &SharedSources,
    budget: &IoBudget,
    name: &str,
    offset: u64,
    length: u64,
    buffer: &mut [u8],
) -> Result<()> {
    let mut position = 0;
    while position < length {
        let count = (length - position).min(buffer.len() as u64) as usize;
        files.read_at(budget, name, offset + position, &mut buffer[..count])?;
        if buffer[..count].iter().any(|b| *b != 0) {
            return Err(invalid("package padding is not zero"));
        }
        position += count as u64;
    }
    Ok(())
}
fn verify_with(
    auth: &Authenticated,
    files: &SharedSources,
    budget: &IoBudget,
    check_length: bool,
) -> Result<()> {
    auth.stored
        .par_iter()
        .try_for_each(|(_, data)| -> Result<()> {
            let mut buffer = budget.buffer()?;
            let mut context = digest::Context::new(&digest::SHA256);
            for part in protection::stored_parts(data)? {
                if part.nbytes == 0
                    || part
                        .offset
                        .checked_add(part.nbytes)
                        .is_none_or(|end| end > files.length(&part.filename).unwrap_or(0))
                {
                    return Err(invalid("signed stored range exceeds payload"));
                }
                hash_range(
                    files,
                    budget,
                    &part.filename,
                    part.offset,
                    part.nbytes,
                    &mut context,
                    buffer.bytes_mut(),
                )?;
            }
            if context.finish().as_ref() != data.sha256 {
                return Err(invalid("stored-data digest mismatch"));
            }
            Ok(())
        })?;
    let mut roles = package::roles(&auth.index)?;
    roles.sort_by_key(|r| (r.span.filename.clone(), r.span.offset));
    roles
        .par_iter()
        .filter(|r| r.kind != 4)
        .try_for_each(|role| -> Result<()> {
            let mut buffer = budget.buffer()?;
            let resource = &auth.resources[&(role.kind, role.owner, role.span.offset)];
            for (i, expected) in resource.sha256.iter().enumerate() {
                let pos = if role.kind == 2 {
                    i as u64 * (protection::CHUNK + 16)
                } else {
                    0
                };
                let count = if role.kind == 2 {
                    (protection::CHUNK + 16).min(role.span.nbytes - pos)
                } else {
                    role.span.nbytes
                };
                let mut context = digest::Context::new(&digest::SHA256);
                hash_range(
                    files,
                    budget,
                    &role.span.filename,
                    role.span.offset + pos,
                    count,
                    &mut context,
                    buffer.bytes_mut(),
                )?;
                if context.finish().as_ref() != expected {
                    return Err(invalid("publisher resource content verification failed"));
                }
            }
            Ok(())
        })?;
    let mut buffer = budget.buffer()?;
    let mut cursors = BTreeMap::new();
    for role in roles {
        let cursor = cursors.entry(role.span.filename.clone()).or_insert(0);
        zero_range(
            files,
            budget,
            &role.span.filename,
            *cursor,
            role.span.offset - *cursor,
            buffer.bytes_mut(),
        )?;
        *cursor = role.span.offset + role.span.nbytes;
    }
    if check_length {
        for (name, cursor) in cursors {
            if cursor != files.length(&name)? {
                return Err(invalid("unreferenced source payload bytes"));
            }
        }
    }
    Ok(())
}
pub(crate) fn verify_source(_path: &Path, auth: &Authenticated, budget: &IoBudget) -> Result<()> {
    verify_with(auth, &auth.sources, budget, true)
}
pub(crate) fn verify_layout(
    directory: &Path,
    layout: &Layout,
    auth: &Authenticated,
    budget: &IoBudget,
) -> Result<()> {
    verify_with(
        auth,
        &SharedSources::new(PackageSources::from_layout(directory, layout.clone())?),
        budget,
        true,
    )
}

pub(crate) fn convert(
    source: &Path,
    destination: &Path,
    profiles: &[Profile],
    cache: Option<&Path>,
    shard_size: u64,
) -> Result<PublishStats> {
    if destination.try_exists()? {
        return Err(invalid("destination already exists"));
    }
    let progress = helios_core::startup::StartupProgress::stderr_rows();
    let phase = progress.phase("Authenticate package and verify source bytes", None);
    let auth = authenticate(source, profiles, false)?;
    let planned = package::groups(&auth.index, shard_size)?;
    let budget = IoBudget::for_ranks(1)?;
    verify_source(source, &auth, &budget)?;
    phase.finish(format!("{} ranks", auth.index.ranks.len()));
    let phase = progress.phase("Hash and relocate payload blobs", None);
    let parent = destination.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let cache = cache
        .map(Path::to_path_buf)
        .unwrap_or_else(|| parent.join(".flare-cache"));
    let store = BlobStore::new(&cache, budget.clone())?;
    let names: Result<Vec<_>> = planned
        .par_iter()
        .map(|group| {
            let name = store.put(group.nbytes, false, None, |offset, buffer| {
                auth.sources
                    .read_at(&budget, &group.filename, group.offset + offset, buffer)
            })?;
            store.link(&name, parent)?;
            Ok(name)
        })
        .collect();
    let mut files = BTreeMap::from([(String::new(), vec![])]);
    for (group, name) in planned.iter().zip(names?) {
        let parts = files.entry(group.filename.clone()).or_insert_with(Vec::new);
        let cursor: u64 = parts.iter().map(|(n, _)| *n).sum();
        if cursor < group.offset {
            parts.push((group.offset - cursor, String::new()));
        }
        parts.push((group.nbytes, name));
    }
    let layout = Layout { version: 1, files };
    phase.finish(format!(
        "{} bytes written; {} bytes reused",
        store.written.load(Ordering::Relaxed),
        store.reused.load(Ordering::Relaxed)
    ));
    let phase = progress.phase("Verify relocated package and publish metadata", None);
    verify_layout(parent, &layout, &auth, &budget)?;
    store.sync()?;
    archive::install_snapshot(destination, &auth.envelope, &layout)?;
    phase.finish(format!("{} envelope bytes preserved", auth.envelope.len()));
    Ok(PublishStats {
        written: store.written.load(Ordering::Relaxed),
        reused: store.reused.load(Ordering::Relaxed),
    })
}

pub(crate) fn resign(
    source: &Path,
    destination: &Path,
    publisher: &mut Publisher,
    profile: Profile,
    options: &ExportOptions,
    shard_size: u64,
) -> Result<PublishStats> {
    if publisher.development {
        return Err(invalid("re-signing requires a production publisher"));
    }
    if destination.try_exists()? {
        return Err(invalid("destination already exists"));
    }
    let auth = authenticate(source, &[profile], true)?;
    let budget = IoBudget::for_ranks(1)?;
    verify_source(source, &auth, &budget)?;
    if publisher.license.is_none() {
        publisher.license = auth.license.clone();
    }
    let parent = destination.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let stage = tempfile::Builder::new()
        .prefix(".flare-resign-")
        .tempdir_in(parent)?;
    let mut index = auth.index.clone();
    let mut roles = package::roles(&index)?;
    roles.sort_by_key(|r| {
        (
            r.kind != 4,
            if r.kind == 4 {
                r.owner.map(|o| o + 1).unwrap_or(0)
            } else {
                0
            },
            if r.kind == 4 {
                protection::stored_parts(&auth.stored[&r.owner])
                    .unwrap()
                    .iter()
                    .position(|p| {
                        p.filename == r.span.filename
                            && p.offset <= r.span.offset
                            && r.span.offset < p.offset + p.nbytes
                    })
                    .unwrap()
            } else {
                0
            },
            r.span.offset,
        )
    });
    let mut plain_files = BTreeMap::new();
    for role in roles.iter().filter(|r| r.kind != 4) {
        if let std::collections::btree_map::Entry::Vacant(entry) = plain_files.entry(role.owner) {
            let path = stage.path().join(format!(
                "plaintext-{}",
                role.owner
                    .map(|r| r.to_string())
                    .unwrap_or_else(|| "shared".into())
            ));
            let file = std::sync::Arc::new(
                OpenOptions::new()
                    .write(true)
                    .read(true)
                    .create_new(true)
                    .open(&path)?,
            );
            entry.insert((path, file));
        }
    }
    let processed: Result<Vec<_>> = roles
        .par_iter()
        .enumerate()
        .filter(|(_, r)| r.kind != 4)
        .map(|(i, role)| {
            let mut buffer = budget.buffer()?;
            let resource = &auth.resources[&(role.kind, role.owner, role.span.offset)];
            let (plain, output) = &plain_files[&role.owner];
            let cipher = if role.kind == 2 {
                Some(protection::resource_key(
                    &auth.publisher.public_key,
                    &auth.package_id,
                    role.span.offset,
                    &auth.namespace,
                )?)
            } else {
                None
            };
            let mut remaining = resource.plain_bytes;
            let mut counter = 0;
            let mut position = 0;
            let mut hash = digest::Context::new(&digest::SHA256);
            while remaining > 0 {
                let count = remaining.min(protection::CHUNK);
                let block = &mut buffer.bytes_mut()
                    [..(count + if cipher.is_some() { 16 } else { 0 }) as usize];
                auth.sources.read_at(
                    &budget,
                    &role.span.filename,
                    role.span.offset + position,
                    block,
                )?;
                position += block.len() as u64;
                if let Some(cipher) = &cipher {
                    if protection::sha256(block) != resource.sha256[counter as usize] {
                        return Err(invalid("development executable digest mismatch"));
                    }
                    let decrypted = cipher
                        .open_in_place(
                            protection::nonce(counter),
                            aead::Aad::from(protection::chunk_aad(
                                &auth.package_id,
                                role.span.offset,
                                resource.plain_bytes,
                                counter,
                            )),
                            block,
                        )
                        .map_err(|_| invalid("development executable authentication failed"))?;
                    super::io::write_all_at(
                        output,
                        decrypted,
                        role.span.offset + resource.plain_bytes - remaining,
                    )?;
                } else {
                    hash.update(&block);
                    super::io::write_all_at(
                        output,
                        &block,
                        role.span.offset + resource.plain_bytes - remaining,
                    )?;
                }
                counter += 1;
                remaining -= count;
            }
            if cipher.is_none() && hash.finish().as_ref() != resource.sha256[0] {
                return Err(invalid("development supporting-resource digest mismatch"));
            }
            Ok((
                i,
                FileSource {
                    path: plain.clone(),
                    offset: role.span.offset,
                    nbytes: resource.plain_bytes,
                },
            ))
        })
        .collect();
    let plain: BTreeMap<_, _> = processed?.into_iter().collect();
    for (_, file) in plain_files.values() {
        file.sync_all()?;
    }
    drop(plain_files);
    let mut payload = Vec::new();
    let mut cursor = 0;
    for (i, role) in roles.iter().enumerate() {
        cursor = package::align(cursor)?;
        let span = package::span_mut(&mut index, &role.key);
        span.offset = cursor;
        span.filename.clear();
        if role.kind == 4 {
            for piece in
                auth.sources
                    .sources(&role.span.filename, role.span.offset, role.span.nbytes)?
            {
                match piece {
                    archive::SourcePiece::Data(data) => {
                        let source_path = data.pinned_path()?;
                        payload.push(FilePart {
                            offset: cursor,
                            source: FileSource {
                                path: source_path,
                                offset: data.offset,
                                nbytes: data.length,
                            },
                        });
                        cursor += data.length;
                    }
                    archive::SourcePiece::Zero(_) => {
                        return Err(invalid("stored resource overlaps layout padding"));
                    }
                }
            }
        } else {
            let source = plain[&i].clone();
            span.nbytes = source.nbytes;
            cursor += source.nbytes;
            payload.push(FilePart {
                offset: span.offset,
                source,
            });
        }
    }
    package::publish_checked(
        destination,
        index,
        payload,
        Some(publisher),
        options,
        shard_size,
        |protected, layout| {
            verify_promoted_stored(&auth, &protected.index, parent, layout, &budget)
        },
    )
}

use pyo3::{exceptions::PyValueError, prelude::*};
#[pyfunction]
fn export_convert(
    py: Python<'_>,
    source: String,
    destination: String,
    profiles: String,
    cache: Option<String>,
    shard_size: u64,
) -> PyResult<String> {
    let profiles = decode_profiles(&profiles).map_err(|e| PyValueError::new_err(e.to_string()))?;
    py.detach(|| {
        convert(
            Path::new(&source),
            Path::new(&destination),
            &profiles,
            cache.as_deref().map(Path::new),
            shard_size,
        )
        .and_then(|value| Ok(serde_json::to_string(&value)?))
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}
#[pyfunction]
fn export_resign(
    py: Python<'_>,
    source: String,
    destination: String,
    publisher: String,
    profile: String,
    options: String,
    shard_size: u64,
) -> PyResult<String> {
    let request: PublisherRequest =
        serde_json::from_str(&publisher).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let profile = decode_profiles(&format!("[{profile}]"))
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .remove(0);
    let options: ExportOptions =
        serde_json::from_str(&options).map_err(|e| PyValueError::new_err(e.to_string()))?;
    py.detach(|| {
        let mut publisher = request.publisher()?;
        let result = resign(
            Path::new(&source),
            Path::new(&destination),
            &mut publisher,
            profile,
            &options,
            shard_size,
        )?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(serde_json::to_string(&result)?)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(export_convert, module)?)?;
    module.add_function(wrap_pyfunction!(export_resign, module)?)?;
    Ok(())
}

fn verify_promoted_stored(
    auth: &Authenticated,
    index: &wire::ModelIndex,
    directory: &Path,
    layout: &Layout,
    budget: &IoBudget,
) -> Result<()> {
    let original = package::roles(&auth.index)?;
    let relocated = package::roles(index)?;
    let files = SharedSources::new(PackageSources::from_layout(directory, layout.clone())?);
    auth.stored
        .par_iter()
        .try_for_each(|(owner, data)| -> Result<()> {
            let mut buffer = budget.buffer()?;
            let mut digest = digest::Context::new(&digest::SHA256);
            for part in protection::stored_parts(data)? {
                let mut resources: Vec<_> = original
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| {
                        r.kind == 4
                            && r.owner == *owner
                            && r.span.filename == part.filename
                            && r.span.offset >= part.offset
                            && r.span.offset + r.span.nbytes <= part.offset + part.nbytes
                    })
                    .collect();
                resources.sort_by_key(|(_, r)| r.span.offset);
                let mut cursor = part.offset;
                for (i, role) in resources {
                    absorb_zeros(&mut digest, role.span.offset - cursor, buffer.bytes_mut());
                    let output = &relocated[i].span;
                    hash_range(
                        &files,
                        budget,
                        &output.filename,
                        output.offset,
                        output.nbytes,
                        &mut digest,
                        buffer.bytes_mut(),
                    )?;
                    cursor = role.span.offset + role.span.nbytes;
                }
                absorb_zeros(
                    &mut digest,
                    part.offset + part.nbytes - cursor,
                    buffer.bytes_mut(),
                );
            }
            if digest.finish().as_ref() != data.sha256 {
                return Err(invalid("stored-data digest mismatch during promotion"));
            }
            Ok(())
        })
}
fn absorb_zeros(digest: &mut digest::Context, mut count: u64, buffer: &mut [u8]) {
    buffer.fill(0);
    while count > 0 {
        let n = count.min(buffer.len() as u64) as usize;
        digest.update(&buffer[..n]);
        count -= n as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::{FilePart, FileSource};

    fn publisher(development: bool, seed: u8) -> Publisher {
        let namespace = if development {
            "local-development"
        } else {
            "flare-tests"
        };
        Publisher::from_seed(
            &[seed; 32],
            namespace.into(),
            if development {
                namespace.into()
            } else {
                "https://huggingface.co/flare-tests/fixture".into()
            },
            development,
            None,
        )
        .unwrap()
    }
    fn profile(publisher: &Publisher) -> Profile {
        Profile {
            name: "Native publisher fixture".into(),
            namespace: publisher.namespace.clone(),
            public_key: publisher.public_key().try_into().unwrap(),
            development: publisher.development,
        }
    }
    fn fixture(directory: &Path) -> (wire::ModelIndex, Vec<FilePart>) {
        let mut index = wire::ModelIndex {
            version: 4,
            ..Default::default()
        };
        let mut payload = vec![];
        let mut cursor = 0;
        for rank in 0..2 {
            let mut bundle = wire::BundleIndex::default();
            for (name, bytes) in [
                ("weights", vec![rank as u8 + 1; 8193]),
                ("cubin", vec![9; protection::CHUNK as usize + 3]),
                ("global", vec![6; 13]),
            ] {
                cursor = package::align(cursor).unwrap();
                let path = directory.join(format!("rank-{rank}-{name}"));
                fs::write(&path, &bytes).unwrap();
                bundle.resources.insert(
                    name.into(),
                    wire::Span {
                        offset: cursor,
                        nbytes: bytes.len() as u64,
                        filename: String::new(),
                    },
                );
                payload.push(FilePart {
                    offset: cursor,
                    source: FileSource {
                        path,
                        offset: 0,
                        nbytes: bytes.len() as u64,
                    },
                });
                cursor += bytes.len() as u64;
            }
            bundle.allocations.push(wire::Allocation {
                name: "weights".into(),
                file: Some("weights".into()),
                nbytes: 8193,
                ..Default::default()
            });
            bundle.modules.push(wire::Module {
                name: "module".into(),
                file: "cubin".into(),
                globals: vec![wire::Global {
                    symbol: "constant".into(),
                    file: "global".into(),
                    nbytes: 13,
                    relocations: vec![],
                }],
            });
            index.ranks.push(bundle);
        }
        (index, payload)
    }
    #[test]
    fn native_publication_is_readable_and_reuses_stored_blobs_without_writes() {
        let directory = tempfile::tempdir().unwrap();
        let (index, payload) = fixture(directory.path());
        let options = ExportOptions {
            cache: Some(directory.path().join("cache")),
            no_weights: false,
        };
        let signer = publisher(false, 42);
        let first = directory.path().join("first.safetensors");
        package::publish(
            &first,
            index.clone(),
            payload.clone(),
            Some(&signer),
            &options,
            64 << 30,
        )
        .unwrap();
        let first_verified = VerifiedPackage::open(&first, &[profile(&signer)], None).unwrap();
        first_verified.verify_stored_data().unwrap();
        let expected = helios_core::runtime_package::test_support::inspect(
            &first,
            "flare-tests",
            true,
            &[profile(&signer)],
            None,
        )
        .unwrap();
        assert_eq!(expected.len(), 4);
        let second = directory.path().join("second.safetensors");
        let strict = ExportOptions {
            no_weights: true,
            ..options.clone()
        };
        let stats =
            package::publish(&second, index, payload, Some(&signer), &strict, 64 << 30).unwrap();
        assert!(stats.reused >= 2 * 8193);
        assert_eq!(
            expected,
            helios_core::runtime_package::test_support::inspect(
                &second,
                "flare-tests",
                true,
                &[profile(&signer)],
                None
            )
            .unwrap()
        );
        let second_index = package::read_index(&second).unwrap().0;
        let expected_written: u64 = package::groups(&second_index, 64 << 30)
            .unwrap()
            .iter()
            .filter(|g| g.roles[0].kind != 4)
            .map(|g| g.nbytes)
            .sum();
        assert_eq!(stats.written, expected_written);
    }
    #[test]
    fn publication_preserves_owner_hashes_across_multiple_cold_and_reused_shards() {
        let directory = tempfile::tempdir().unwrap();
        let (mut index, mut payload) = fixture(directory.path());
        let weights = directory.path().join("additional-weights");
        fs::write(&weights, vec![37; 1 << 20]).unwrap();
        let mut cursor = payload
            .iter()
            .map(|p| p.offset + p.source.nbytes)
            .max()
            .unwrap();
        for bundle in &mut index.ranks {
            for i in 0..9 {
                cursor = package::align(cursor).unwrap();
                let name = format!("additional-{i}");
                bundle.resources.insert(
                    name.clone(),
                    wire::Span {
                        offset: cursor,
                        nbytes: 1 << 20,
                        filename: String::new(),
                    },
                );
                bundle.allocations.push(wire::Allocation {
                    name: name.clone(),
                    file: Some(name),
                    nbytes: 1 << 20,
                    ..Default::default()
                });
                payload.push(FilePart {
                    offset: cursor,
                    source: FileSource {
                        path: weights.clone(),
                        offset: 0,
                        nbytes: 1 << 20,
                    },
                });
                cursor += 1 << 20;
            }
        }
        let options = ExportOptions {
            cache: Some(directory.path().join("cache")),
            no_weights: false,
        };
        let signer = publisher(false, 42);
        for no_weights in [false, true] {
            let path = directory.path().join(format!("{no_weights}.safetensors"));
            package::publish(
                &path,
                index.clone(),
                payload.clone(),
                Some(&signer),
                &ExportOptions {
                    no_weights,
                    ..options.clone()
                },
                protection::CHUNK + 16384,
            )
            .unwrap();
            let package = VerifiedPackage::open(&path, &[profile(&signer)], None).unwrap();
            package.verify_stored_data().unwrap();
            let inspected = helios_core::runtime_package::test_support::inspect(
                &path,
                "flare-tests",
                true,
                &[profile(&signer)],
                None,
            )
            .unwrap();
            assert_eq!(inspected.len(), 4);
            assert!(
                package::groups(
                    &package::read_index(&path).unwrap().0,
                    protection::CHUNK + 16384
                )
                .unwrap()
                .iter()
                .filter(|g| g.roles[0].kind == 4)
                .count()
                    >= 6
            );
        }
    }
    #[cfg(unix)]
    #[test]
    fn publication_bounds_handles_for_more_than_4096_distinct_sources() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "export::conversion::tests::distinct_source_publication_child",
                "--ignored",
            ])
            .status()
            .unwrap();
        assert!(status.success());
    }
    #[test]
    fn publication_rejects_changed_and_replaced_sources_after_preparation() {
        for replace in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let (index, payload) = fixture(directory.path());
            let original = payload[0].source.path.clone();
            let signer = publisher(false, 42);
            let budget = IoBudget::for_ranks(1).unwrap();
            let progress = helios_core::startup::StartupProgress::stderr_rows();
            let phase = progress.phase("Prepare fixture resources", None);
            let protected = protection::prepare(
                index,
                payload,
                &signer,
                directory.path(),
                64 << 30,
                "fixture",
                budget.clone(),
                &phase,
            )
            .unwrap();
            if replace {
                fs::rename(&original, directory.path().join("replaced-source")).unwrap();
            }
            fs::write(&original, vec![123; 8193]).unwrap();
            let store = BlobStore::new(&directory.path().join("cache"), budget).unwrap();
            let error =
                package::publish_groups(&store, &protected, false, 64 << 30, &phase).unwrap_err();
            assert!(error.to_string().contains("changed"), "{error}");
        }
    }
    #[cfg(unix)]
    #[test]
    #[ignore]
    fn distinct_source_publication_child() {
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        assert_eq!(
            unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) },
            0
        );
        limit.rlim_cur = limit.rlim_cur.min(4096);
        assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) }, 0);
        let directory = tempfile::tempdir().unwrap();
        let mut index = wire::ModelIndex {
            version: 4,
            ranks: vec![wire::BundleIndex::default()],
            ..Default::default()
        };
        let mut payload = Vec::new();
        for i in 0..4200u64 {
            let name = format!("weight-{i}");
            let path = directory.path().join(&name);
            fs::write(&path, i.to_le_bytes()).unwrap();
            index.ranks[0].resources.insert(
                name.clone(),
                wire::Span {
                    offset: i * 4096,
                    nbytes: 8,
                    filename: String::new(),
                },
            );
            index.ranks[0].allocations.push(wire::Allocation {
                name: name.clone(),
                file: Some(name),
                nbytes: 8,
                ..Default::default()
            });
            payload.push(FilePart {
                offset: i * 4096,
                source: FileSource {
                    path,
                    offset: 0,
                    nbytes: 8,
                },
            });
        }
        let path = directory.path().join("catalog.safetensors");
        let signer = publisher(false, 42);
        package::publish(
            &path,
            index,
            payload,
            Some(&signer),
            &ExportOptions {
                cache: Some(directory.path().join("cache")),
                no_weights: false,
            },
            64 << 30,
        )
        .unwrap();
        VerifiedPackage::open(&path, &[profile(&signer)], None)
            .unwrap()
            .verify_stored_data()
            .unwrap();
    }
    #[test]
    fn conversion_preserves_envelope_and_promotion_preserves_plaintext() {
        let directory = tempfile::tempdir().unwrap();
        let (index, payload) = fixture(directory.path());
        let options = ExportOptions {
            cache: Some(directory.path().join("cache")),
            no_weights: false,
        };
        let development = publisher(true, 17);
        let source = directory.path().join("development.safetensors");
        package::publish(
            &source,
            index,
            payload,
            Some(&development),
            &options,
            64 << 30,
        )
        .unwrap();
        let target = directory.path().join("converted.safetensors");
        convert(
            &source,
            &target,
            &[profile(&development)],
            options.cache.as_deref(),
            protection::CHUNK + 16384,
        )
        .unwrap();
        assert_eq!(
            Container::open(&source).unwrap().envelope,
            Container::open(&target).unwrap().envelope
        );
        let original = helios_core::runtime_package::test_support::inspect(
            &target,
            "local-development",
            true,
            &[profile(&development)],
            None,
        )
        .unwrap();
        let mut production = publisher(false, 42);
        let promoted = directory.path().join("production.safetensors");
        resign(
            &target,
            &promoted,
            &mut production,
            profile(&development),
            &options,
            64 << 30,
        )
        .unwrap();
        let actual = helios_core::runtime_package::test_support::inspect(
            &promoted,
            "flare-tests",
            true,
            &[profile(&production)],
            None,
        )
        .unwrap();
        assert_eq!(
            original.iter().map(|(_, h)| h).collect::<Vec<_>>(),
            actual.iter().map(|(_, h)| h).collect::<Vec<_>>()
        );
    }
    #[test]
    fn absent_required_weights_and_mutated_resources_never_publish_primary() {
        let directory = tempfile::tempdir().unwrap();
        let (index, payload) = fixture(directory.path());
        let options = ExportOptions {
            cache: Some(directory.path().join("cache")),
            no_weights: true,
        };
        let signer = publisher(false, 42);
        let destination = directory.path().join("missing.safetensors");
        let error = package::publish(
            &destination,
            index.clone(),
            payload.clone(),
            Some(&signer),
            &options,
            64 << 30,
        )
        .unwrap_err();
        assert!(error.to_string().contains("no-weights"));
        assert!(!destination.exists());
        let source = directory.path().join("source.safetensors");
        package::publish(
            &source,
            index,
            payload,
            Some(&signer),
            &ExportOptions {
                no_weights: false,
                ..options.clone()
            },
            64 << 30,
        )
        .unwrap();
        let container = Container::open(&source).unwrap();
        let layout = container.layout.unwrap();
        let blob = directory.path().join(&layout.files[""][0].1);
        let mut permissions = fs::metadata(&blob).unwrap().permissions();
        permissions.set_readonly(false);
        fs::set_permissions(&blob, permissions).unwrap();
        let file = OpenOptions::new().write(true).open(&blob).unwrap();
        crate::export::io::write_all_at(&file, b"wrong", 4096).unwrap();
        let output = directory.path().join("invalid.safetensors");
        let error = convert(
            &source,
            &output,
            &[profile(&signer)],
            options.cache.as_deref(),
            64 << 30,
        )
        .unwrap_err();
        assert!(error.to_string().contains("digest"));
        assert!(!output.exists());
    }
}

fn decode_profiles(encoded: &str) -> Result<Vec<Profile>> {
    let mut values: Vec<serde_json::Value> = serde_json::from_str(encoded)?;
    for value in &mut values {
        if let Some(object) = value.as_object_mut() {
            if !object.contains_key("name") {
                let namespace = object
                    .get("namespace")
                    .cloned()
                    .unwrap_or_else(|| "Package publisher".into());
                object.insert("name".into(), namespace);
            }
        }
    }
    Ok(serde_json::from_value(values.into())?)
}
