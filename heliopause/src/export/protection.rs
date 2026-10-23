//! Publisher signatures and acceptance-bound executable encryption.
use super::io::{Identity, IoBudget, Source};
use super::package::{self, Role};
use super::{ExportResult as Result, FilePart, FileSource, invalid, wire};
use prost::Message;
use rayon::prelude::*;
use ring::{
    aead, digest, hkdf,
    rand::{SecureRandom, SystemRandom},
    signature::{self, KeyPair},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

pub(crate) const CHUNK: u64 = 4 << 20;
pub(crate) fn sha256(bytes: &[u8]) -> Vec<u8> {
    digest::digest(&digest::SHA256, bytes).as_ref().to_vec()
}
pub(crate) fn fields(values: &[&[u8]]) -> Vec<u8> {
    let mut result = Vec::new();
    for value in values {
        result.extend_from_slice(&(value.len() as u64).to_le_bytes());
        result.extend_from_slice(value);
    }
    result
}
pub(crate) fn signature_message(package: &[u8], signer: &[u8], kind: u32, body: &[u8]) -> Vec<u8> {
    fields(&[
        b"flare-package-signature-v4",
        &4u32.to_le_bytes(),
        package,
        signer,
        &kind.to_le_bytes(),
        body,
    ])
}
pub(crate) fn resource_key(
    public: &[u8],
    package: &[u8],
    offset: u64,
    namespace: &str,
) -> Result<aead::LessSafeKey> {
    let info = fields(&[
        b"flare-executable-v3",
        &offset.to_le_bytes(),
        namespace.as_bytes(),
    ]);
    let key = hkdf::Salt::new(hkdf::HKDF_SHA256, package).extract(public);
    let parts = [info.as_slice()];
    let mut bytes = [0; 32];
    key.expand(&parts, hkdf::HKDF_SHA256)
        .map_err(|_| invalid("derive executable key"))?
        .fill(&mut bytes)
        .map_err(|_| invalid("derive executable key"))?;
    Ok(aead::LessSafeKey::new(
        aead::UnboundKey::new(&aead::AES_256_GCM, &bytes)
            .map_err(|_| invalid("invalid executable key"))?,
    ))
}
pub(crate) fn chunk_aad(package: &[u8], offset: u64, size: u64, chunk: u64) -> Vec<u8> {
    fields(&[
        b"flare-executable-chunk-v3",
        package,
        &offset.to_le_bytes(),
        &size.to_le_bytes(),
        &chunk.to_le_bytes(),
    ])
}
pub(crate) fn nonce(counter: u64) -> aead::Nonce {
    let mut bytes = [0; 12];
    bytes[4..].copy_from_slice(&counter.to_be_bytes());
    aead::Nonce::assume_unique_for_key(bytes)
}
pub(crate) fn validate_namespace(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 96
        || !value.as_bytes()[0].is_ascii_alphanumeric()
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
    {
        return Err(invalid(
            "publisher namespace must be a Hugging Face owner name",
        ));
    }
    Ok(())
}
pub(crate) fn validate_source(namespace: &str, source: &str) -> Result<()> {
    validate_namespace(namespace)?;
    let repository = source
        .strip_prefix(&format!("https://huggingface.co/{namespace}/"))
        .ok_or_else(|| {
            invalid("official_source must be https://huggingface.co/<publisher>/<repository>")
        })?;
    if repository.is_empty()
        || repository.len() > 96
        || !repository.as_bytes()[0].is_ascii_alphanumeric()
        || !repository
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    {
        return Err(invalid(
            "official_source must be https://huggingface.co/<publisher>/<repository>",
        ));
    }
    Ok(())
}
pub(crate) fn validate_license(value: &wire::License) -> Result<()> {
    if value.name.is_empty()
        || value.name.chars().count() > 128
        || value.name.chars().any(char::is_control)
    {
        return Err(invalid(
            "license name must be 1 to 128 printable characters",
        ));
    }
    if !value.link.is_empty()
        && (!value.link.starts_with("https://")
            || value.link.len() <= 8
            || value.link.len() > 512
            || !value.link.bytes().all(|b| b.is_ascii_graphic()))
    {
        return Err(invalid(
            "license link must be an https URL of printable ASCII, at most 512 characters",
        ));
    }
    Ok(())
}

pub(crate) struct Publisher {
    key: signature::Ed25519KeyPair,
    pub namespace: String,
    pub official_source: String,
    pub development: bool,
    pub license: Option<wire::License>,
}
impl Publisher {
    pub(crate) fn from_seed(
        seed: &[u8],
        namespace: String,
        official_source: String,
        development: bool,
        license: Option<wire::License>,
    ) -> Result<Self> {
        if development {
            if namespace != "local-development" || official_source != "local-development" {
                return Err(invalid("invalid local development publisher"));
            }
        } else {
            validate_source(&namespace, &official_source)?;
        }
        if let Some(value) = &license {
            validate_license(value)?;
        }
        let key = signature::Ed25519KeyPair::from_seed_unchecked(seed)
            .map_err(|_| invalid("publishing requires an Ed25519 private key"))?;
        Ok(Self {
            key,
            namespace,
            official_source,
            development,
            license,
        })
    }
    pub(crate) fn load(
        path: &Path,
        password: Option<&[u8]>,
        namespace: String,
        official_source: String,
        license: Option<wire::License>,
    ) -> Result<Self> {
        let mut file = fs::File::open(path)?;
        let metadata = file.metadata()?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if metadata.uid() != unsafe { libc::getuid() } || metadata.mode() & 0o077 != 0 {
                return Err(invalid(format!(
                    "signing key must have owner-only permissions: {}",
                    path.display()
                )));
            }
        }
        #[cfg(windows)]
        validate_private_file(path)?;
        if !metadata.is_file() {
            return Err(invalid("signing key must be an owner-only regular file"));
        }
        if metadata.len() > 16384 {
            return Err(invalid("signing key exceeds 16 KiB"));
        }
        let mut encoded = Vec::new();
        std::io::Read::by_ref(&mut file)
            .take(16385)
            .read_to_end(&mut encoded)?;
        if encoded.len() > 16384 {
            return Err(invalid("signing key exceeds 16 KiB"));
        }
        let mut seed = load_key_seed(&encoded, password)?;
        let result = Self::from_seed(
            &seed,
            namespace.clone(),
            official_source.clone(),
            namespace == "local-development" && official_source == "local-development",
            license,
        );
        seed.fill(0);
        encoded.fill(0);
        result
    }
    pub(crate) fn public_key(&self) -> &[u8] {
        self.key.public_key().as_ref()
    }
    pub(crate) fn fingerprint(&self) -> Vec<u8> {
        sha256(self.public_key())
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct PublisherRequest {
    pub namespace: String,
    pub official_source: String,
    #[serde(default)]
    pub development: bool,
    pub license: Option<wire::License>,
    pub key_path: Option<std::path::PathBuf>,
    #[serde(default)]
    pub password: Option<Vec<u8>>,
    #[serde(default)]
    pub seed: Option<Vec<u8>>,
    #[serde(default)]
    pub expected_public_key: Option<Vec<u8>>,
}
impl PublisherRequest {
    pub(crate) fn publisher(&self) -> Result<Publisher> {
        if let Some(path) = &self.key_path {
            if self.development {
                if self.namespace != "local-development"
                    || self.official_source != "local-development"
                {
                    return Err(invalid("invalid local development publisher"));
                }
            } else {
                validate_source(&self.namespace, &self.official_source)?;
            }
            {
                let mut result = Publisher::load(
                    path,
                    self.password.as_deref(),
                    self.namespace.clone(),
                    self.official_source.clone(),
                    self.license.clone(),
                )?;
                if self
                    .expected_public_key
                    .as_deref()
                    .is_none_or(|expected| expected != result.public_key())
                {
                    return Err(invalid(
                        "signing key changed since publisher initialization",
                    ));
                }
                result.development = self.development;
                if self.development
                    && (result.namespace != "local-development"
                        || result.official_source != "local-development")
                {
                    return Err(invalid("invalid local development publisher"));
                }
                Ok(result)
            }
        } else {
            Publisher::from_seed(
                self.seed
                    .as_deref()
                    .ok_or_else(|| invalid("publisher private key is missing"))?,
                self.namespace.clone(),
                self.official_source.clone(),
                self.development,
                self.license.clone(),
            )
        }
    }
}

#[derive(Clone)]
pub(crate) struct SourceSnapshot {
    pub path: PathBuf,
    pub identity: Identity,
}
impl SourceSnapshot {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let source = Source::open(path, 0, 0)?;
        Ok(Self {
            path: path.into(),
            identity: source.identity(),
        })
    }
    pub(crate) fn source(&self, offset: u64, length: u64) -> Result<Source> {
        Source::open_checked(&self.path, offset, length, &self.identity)
    }
    pub(crate) fn check_unchanged(&self) -> Result<()> {
        self.source(0, 0)?.check_unchanged()
    }
}

pub(crate) struct Protected {
    pub source_snapshots: Vec<SourceSnapshot>,
    pub index: wire::ModelIndex,
    pub payloads: BTreeMap<String, Vec<FilePart>>,
    pub ranges: BTreeMap<Option<usize>, Vec<wire::Span>>,
    pub code: BTreeMap<usize, wire::ResourceCatalog>,
    pub support: BTreeMap<Option<usize>, wire::ResourceCatalog>,
    pub package_id: Vec<u8>,
}
impl Protected {
    pub(crate) fn stored(
        &self,
        owner: Option<usize>,
        hashes: &BTreeMap<Option<usize>, Vec<u8>>,
    ) -> wire::StoredData {
        let parts = &self.ranges[&owner];
        let sha256 = hashes.get(&owner).cloned().unwrap_or_else(|| vec![0; 32]);
        if self.index.version == 4 {
            let span = parts.first().cloned().unwrap_or_default();
            wire::StoredData {
                offset: span.offset,
                nbytes: span.nbytes,
                sha256,
                parts: vec![],
            }
        } else {
            wire::StoredData {
                offset: 0,
                nbytes: parts.iter().map(|p| p.nbytes).sum(),
                sha256,
                parts: parts.clone(),
            }
        }
    }
    pub(crate) fn envelope(
        &self,
        publisher: &Publisher,
        hashes: &BTreeMap<Option<usize>, Vec<u8>>,
    ) -> Result<Vec<u8>> {
        let signer = publisher.fingerprint();
        let mut shared = self.index.clone();
        shared.ranks.clear();
        let support = self.support[&None].encode_to_vec();
        let stored = self.stored(None, hashes).encode_to_vec();
        let metadata = wire::PublisherMetadata {
            index: Some(shared),
            namespace: publisher.namespace.clone(),
            official_source: publisher.official_source.clone(),
            development: publisher.development,
            ranks: self.index.ranks.len() as u32,
            group_hashes: vec![sha256(&support), sha256(&stored)],
            license: publisher.license.clone(),
        };
        let sign = |kind, body: Vec<u8>| {
            let signature = publisher
                .key
                .sign(&signature_message(&self.package_id, &signer, kind, &body))
                .as_ref()
                .to_vec();
            wire::SignedGroup {
                kind,
                body,
                signature,
            }
        };
        let groups = vec![
            sign(1, metadata.encode_to_vec()),
            sign(2, support),
            sign(3, stored),
        ];
        let ranks = self
            .index
            .ranks
            .iter()
            .enumerate()
            .map(|(rank, bundle)| {
                sign(
                    4,
                    wire::RankProtection {
                        rank: rank as u32,
                        bundle: Some(bundle.clone()),
                        executables: Some(self.code[&rank].clone()),
                        support: Some(self.support[&Some(rank)].clone()),
                        data: Some(self.stored(Some(rank), hashes)),
                    }
                    .encode_to_vec(),
                )
            })
            .collect();
        let encoded = wire::ProtectedIndex {
            version: 4,
            package_id: self.package_id.clone(),
            signer,
            groups,
            ranks,
        }
        .encode_to_vec();
        if encoded.is_empty() || encoded.len() > 128 << 20 {
            return Err(invalid("protected package envelope exceeds metadata limit"));
        }
        Ok(encoded)
    }
}
#[derive(Clone)]
struct Input {
    role: Role,
    sources: Vec<FileSource>,
    plain: u64,
}
fn layout(
    mut index: wire::ModelIndex,
    inputs: &[Input],
    sharded: bool,
    shard_size: u64,
    basename: &str,
    package_id: Vec<u8>,
) -> Result<(Protected, Vec<(String, u64)>)> {
    let owners: Vec<_> = std::iter::once(None)
        .chain((0..index.ranks.len()).map(Some))
        .collect();
    let mut result = Protected {
        source_snapshots: vec![],
        index: index.clone(),
        payloads: BTreeMap::from([(String::new(), vec![])]),
        ranges: owners.iter().map(|o| (*o, vec![])).collect(),
        code: (0..index.ranks.len())
            .map(|r| (r, wire::ResourceCatalog::default()))
            .collect(),
        support: owners
            .iter()
            .map(|o| (*o, wire::ResourceCatalog::default()))
            .collect(),
        package_id,
    };
    let mut positions = BTreeMap::from([(String::new(), 0)]);
    let mut files = 0;
    let mut data_file = String::new();
    let mut locations = Vec::new();
    for input in inputs {
        let role = &input.role;
        let stored = input.plain
            + if role.kind == 2 {
                16 * input.plain.div_ceil(CHUNK)
            } else {
                0
            };
        if sharded && role.kind == 4 {
            if data_file.is_empty()
                || positions[&data_file] > 0
                    && package::align(positions[&data_file])?
                        .checked_add(stored)
                        .and_then(|v| v.checked_add(4096))
                        .is_none_or(|v| v > shard_size)
            {
                files += 1;
                if files > 99 {
                    return Err(invalid("package requires more than 99 data files"));
                }
                data_file = files.to_string();
                positions.insert(data_file.clone(), 0);
            }
        }
        let filename = if sharded && role.kind == 4 {
            data_file.clone()
        } else {
            String::new()
        };
        let cursor = package::align(positions[&filename])?;
        positions.insert(
            filename.clone(),
            cursor
                .checked_add(stored)
                .ok_or_else(|| invalid("payload extent overflow"))?,
        );
        locations.push((filename.clone(), cursor));
        let span = package::span_mut(&mut index, &role.key);
        *span = wire::Span {
            offset: cursor,
            nbytes: stored,
            filename: filename.clone(),
        };
        if role.kind == 4 {
            let parts = result.ranges.get_mut(&role.owner).unwrap();
            if parts.last().is_some_and(|p| p.filename == filename) {
                let last = parts.last_mut().unwrap();
                last.nbytes = cursor + stored - last.offset;
            } else {
                parts.push(span.clone());
            }
        } else {
            let catalog = if role.kind == 2 {
                result.code.get_mut(&role.owner.unwrap()).unwrap()
            } else {
                result.support.get_mut(&role.owner).unwrap()
            };
            catalog.resources.push(wire::ProtectedResource {
                offset: cursor,
                plain_bytes: input.plain,
                stored_bytes: stored,
                sha256: vec![
                    vec![0; 32];
                    if role.kind == 2 {
                        input.plain.div_ceil(CHUNK) as usize
                    } else {
                        1
                    }
                ],
            });
        }
    }
    for (filename, length) in &positions {
        let mut spans: Vec<_> = result
            .ranges
            .values()
            .flat_map(|p| p.iter())
            .filter(|p| &p.filename == filename)
            .map(|p| p.offset)
            .collect();
        spans.sort();
        let end = inputs
            .iter()
            .filter(|i| i.role.kind != 4)
            .map(|i| package::span_mut(&mut index, &i.role.key).clone())
            .filter(|s| &s.filename == filename)
            .map(|s| s.offset)
            .min()
            .unwrap_or(*length);
        for parts in result.ranges.values_mut() {
            for part in parts.iter_mut().filter(|p| &p.filename == filename) {
                part.nbytes = spans
                    .iter()
                    .copied()
                    .find(|offset| *offset > part.offset)
                    .unwrap_or(end)
                    - part.offset;
            }
        }
    }
    let names: BTreeMap<_, _> = if sharded {
        (1..=files)
            .map(|i| {
                (
                    i.to_string(),
                    format!("{basename}-data-{i:02}-of-{files:02}.safetensors"),
                )
            })
            .collect()
    } else {
        BTreeMap::new()
    };
    for name in names.values() {
        super::archive::validate_filename(name)?;
    }
    for input in inputs {
        let span = package::span_mut(&mut index, &input.role.key);
        if let Some(name) = names.get(&span.filename) {
            span.filename = name.clone();
        }
    }
    for parts in result.ranges.values_mut() {
        for part in parts {
            if let Some(name) = names.get(&part.filename) {
                part.filename = name.clone();
            }
        }
    }
    for (filename, _) in &mut locations {
        if let Some(name) = names.get(filename) {
            *filename = name.clone();
        }
    }
    result.payloads = positions
        .keys()
        .map(|f| (names.get(f).cloned().unwrap_or_else(|| f.clone()), vec![]))
        .collect();
    index.version = if sharded { 5 } else { 4 };
    result.index = index;
    Ok((result, locations))
}

pub(crate) fn prepare(
    index: wire::ModelIndex,
    payload: Vec<FilePart>,
    publisher: &Publisher,
    stage: &Path,
    shard_size: u64,
    basename: &str,
    budget: Arc<IoBudget>,
    phase: &helios_core::startup::Phase,
) -> Result<Protected> {
    package::groups(&index, shard_size)?;
    package::validate_index(&index, package::validate_payload(&payload)?)?;
    let mut resources = package::roles(&index)?;
    resources.sort_by_key(|r| r.span.offset);
    for part in &payload {
        let resource = resources
            .partition_point(|r| r.span.offset <= part.offset)
            .checked_sub(1)
            .map(|i| &resources[i])
            .ok_or_else(|| invalid("publication range has no classified resource"))?;
        if part
            .offset
            .checked_add(part.source.nbytes)
            .is_none_or(|end| end > resource.span.offset + resource.span.nbytes)
        {
            return Err(invalid(
                "publication ranges differ from classified resources",
            ));
        }
    }
    let mut inputs: Vec<_> = package::roles(&index)?
        .into_iter()
        .map(|role| {
            if !role.span.filename.is_empty() {
                return Err(invalid(
                    "publication inputs must use a contiguous staging address space",
                ));
            }
            let sources = package::locate(&payload, &role.span)?;
            if role.kind != 4 && sources.len() != 1 {
                return Err(invalid("protected resources require one contiguous source"));
            }
            Ok(Input {
                plain: role.span.nbytes,
                role,
                sources,
            })
        })
        .collect::<Result<_>>()?;
    let mut source_files = BTreeMap::new();
    for input in &inputs {
        for source in &input.sources {
            if !source_files.contains_key(&source.path) {
                source_files.insert(source.path.clone(), SourceSnapshot::open(&source.path)?);
            }
            source_files[&source.path].source(source.offset, source.nbytes)?;
        }
    }
    let source_snapshots: Vec<_> = source_files.values().cloned().collect();
    inputs.sort_by_key(|i| {
        (
            i.role.kind != 4,
            if i.role.kind == 4 {
                i.role.owner.map(|o| o + 1).unwrap_or(0)
            } else {
                0
            },
            i.role.span.offset,
        )
    });
    let mut package_id = vec![0; 32];
    SystemRandom::new()
        .fill(&mut package_id)
        .map_err(|_| invalid("generate package identity"))?;
    let (mut result, mut locations) = layout(
        index.clone(),
        &inputs,
        false,
        shard_size,
        basename,
        package_id.clone(),
    )?;
    let length = locations
        .iter()
        .zip(&inputs)
        .map(|((_, offset), i)| {
            offset
                + i.plain
                + if i.role.kind == 2 {
                    16 * i.plain.div_ceil(CHUNK)
                } else {
                    0
                }
        })
        .max()
        .unwrap_or(0);
    if 4096 + length + result.envelope(publisher, &BTreeMap::new())?.len() as u64 > shard_size
        && inputs.iter().any(|i| i.role.kind == 4)
    {
        (result, locations) = layout(index, &inputs, true, shard_size, basename, package_id)?;
    }
    let mut encrypted_files = BTreeMap::new();
    for input in inputs.iter().filter(|i| i.role.kind == 2) {
        let owner = input.role.owner.unwrap();
        if let std::collections::btree_map::Entry::Vacant(entry) = encrypted_files.entry(owner) {
            let path = stage.join(format!("executables-rank-{owner}.bin"));
            let file = Arc::new(
                OpenOptions::new()
                    .write(true)
                    .read(true)
                    .create_new(true)
                    .open(&path)?,
            );
            entry.insert((path, file));
        }
    }
    let processed: Result<Vec<_>> = inputs
        .par_iter()
        .zip(&locations)
        .map(|(input, (filename, offset))| {
            let mut parts = Vec::new();
            let mut hashes = Vec::new();
            if input.role.kind == 2 {
                let (encrypted, output) = &encrypted_files[&input.role.owner.unwrap()];
                let mut buffer = budget.buffer()?;
                let source = source_files[&input.sources[0].path]
                    .source(input.sources[0].offset, input.plain)?;
                let cipher = resource_key(
                    publisher.public_key(),
                    &result.package_id,
                    *offset,
                    &publisher.namespace,
                )?;
                let mut position = 0;
                let mut counter = 0;
                while position < input.plain {
                    let count = CHUNK.min(input.plain - position);
                    let block = &mut buffer.bytes_mut()[..count as usize];
                    budget.read_at(&source, position, block)?;
                    phase.advance_bytes(count);
                    let tag = cipher
                        .seal_in_place_separate_tag(
                            nonce(counter),
                            aead::Aad::from(chunk_aad(
                                &result.package_id,
                                *offset,
                                input.plain,
                                counter,
                            )),
                            block,
                        )
                        .map_err(|_| invalid("encrypt executable resource"))?;
                    let mut hash = digest::Context::new(&digest::SHA256);
                    hash.update(block);
                    hash.update(tag.as_ref());
                    hashes.push(hash.finish().as_ref().to_vec());
                    let destination = *offset + position + 16 * counter;
                    super::io::write_all_at(output, block, destination)?;
                    super::io::write_all_at(output, tag.as_ref(), destination + count)?;
                    position += count;
                    counter += 1;
                }
                source.check_unchanged()?;
                parts.push(FilePart {
                    offset: *offset,
                    source: FileSource {
                        path: encrypted.clone(),
                        offset: *offset,
                        nbytes: input.plain + 16 * input.plain.div_ceil(CHUNK),
                    },
                });
            } else {
                if input.role.kind == 3 {
                    let source = source_files[&input.sources[0].path]
                        .source(input.sources[0].offset, input.plain)?;
                    let mut context = digest::Context::new(&digest::SHA256);
                    budget.scan(&source, |block| {
                        context.update(block);
                        phase.advance_bytes(block.len() as u64);
                        Ok(())
                    })?;
                    hashes.push(context.finish().as_ref().to_vec());
                }
                let mut cursor = *offset;
                for source in &input.sources {
                    parts.push(FilePart {
                        offset: cursor,
                        source: source.clone(),
                    });
                    cursor += source.nbytes;
                }
            }
            Ok((
                filename.clone(),
                *offset,
                input.role.kind,
                input.role.owner,
                parts,
                hashes,
            ))
        })
        .collect();
    let mut catalog_positions = BTreeMap::new();
    for (owner, catalog) in &result.code {
        for (i, resource) in catalog.resources.iter().enumerate() {
            catalog_positions.insert((2, Some(*owner), resource.offset), i);
        }
    }
    for (owner, catalog) in &result.support {
        for (i, resource) in catalog.resources.iter().enumerate() {
            catalog_positions.insert((3, *owner, resource.offset), i);
        }
    }
    for (filename, offset, kind, owner, parts, hashes) in processed? {
        result.payloads.get_mut(&filename).unwrap().extend(parts);
        if kind != 4 {
            let catalog = if kind == 2 {
                result.code.get_mut(&owner.unwrap()).unwrap()
            } else {
                result.support.get_mut(&owner).unwrap()
            };
            catalog.resources[catalog_positions[&(kind, owner, offset)]].sha256 = hashes;
        }
    }
    for (_, file) in encrypted_files.values() {
        file.sync_all()?;
    }
    for source in &source_snapshots {
        source.check_unchanged()?;
    }
    result.source_snapshots = source_snapshots;
    for asset in &result.index.assets {
        let span = asset
            .payload
            .as_ref()
            .ok_or_else(|| invalid("asset lacks payload"))?;
        let resource = result.support[&None]
            .resources
            .iter()
            .find(|r| r.offset == span.offset)
            .ok_or_else(|| invalid("asset lacks signed resource"))?;
        let expected: String = resource.sha256[0]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if asset.sha256 != expected {
            return Err(invalid("asset digest differs from publisher catalog"));
        }
    }
    for parts in result.payloads.values_mut() {
        parts.sort_by_key(|p| p.offset);
    }
    Ok(result)
}

pub(crate) fn stored_parts(data: &wire::StoredData) -> Result<Vec<wire::Span>> {
    if data.sha256.len() != 32 {
        return Err(invalid("invalid stored-data digest"));
    }
    if data.parts.is_empty() {
        Ok(if data.nbytes == 0 {
            vec![]
        } else {
            vec![wire::Span {
                offset: data.offset,
                nbytes: data.nbytes,
                filename: String::new(),
            }]
        })
    } else {
        if data.offset != 0
            || data
                .parts
                .iter()
                .try_fold(0u64, |sum, p| sum.checked_add(p.nbytes))
                != Some(data.nbytes)
        {
            return Err(invalid("invalid signed stored-data parts"));
        }
        Ok(data.parts.clone())
    }
}

use pyo3::{exceptions::PyValueError, prelude::*};
fn publisher_info(publisher: &Publisher) -> serde_json::Value {
    serde_json::json!({"public_key":publisher.public_key(),"fingerprint":publisher.fingerprint()})
}
#[pyfunction]
#[pyo3(signature=(path,password=None,namespace="local-development",official_source="local-development"))]
fn export_key_info(
    py: Python<'_>,
    path: String,
    password: Option<Vec<u8>>,
    namespace: &str,
    official_source: &str,
) -> PyResult<String> {
    let namespace = namespace.to_owned();
    let official_source = official_source.to_owned();
    py.detach(|| {
        let path = Path::new(&path);
        let file = fs::File::open(path)?;
        let metadata = file.metadata()?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if metadata.uid() != unsafe { libc::getuid() } || metadata.mode() & 0o077 != 0 {
                return Err(invalid(format!(
                    "signing key must have owner-only permissions: {}",
                    path.display()
                )));
            }
        }
        if !metadata.is_file() || metadata.len() > 16384 {
            return Err(invalid(
                "signing key must be a regular file no larger than 16 KiB",
            ));
        }
        #[cfg(windows)]
        validate_private_file(path)?;
        let mut prefix = [0; 37];
        let count = Read::read(&mut file.take(37), &mut prefix)?;
        if password.is_none()
            && prefix[..count].starts_with(b"-----BEGIN ENCRYPTED PRIVATE KEY-----")
        {
            return Ok(serde_json::to_string(
                &serde_json::json!({"encrypted":true}),
            )?);
        }
        let publisher =
            Publisher::load(path, password.as_deref(), namespace, official_source, None)?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(serde_json::to_string(&publisher_info(
            &publisher,
        ))?)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}
#[pyfunction]
fn export_development_publisher(py: Python<'_>, directory: String) -> PyResult<String> {
    py.detach(|| {
        development_publisher(Path::new(&directory))
            .and_then(|value| Ok(serde_json::to_string(&value)?))
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}
fn development_publisher(directory: &Path) -> Result<serde_json::Value> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(directory)?;
    }
    #[cfg(not(unix))]
    fs::create_dir_all(directory)?;
    let path = directory.join("development-signing-key.pem");
    if !path.try_exists()? {
        let key = openssl::pkey::PKey::generate_ed25519()?;
        let mut encoded = key.private_key_to_pem_pkcs8()?;
        let mut temporary = tempfile::Builder::new()
            .prefix(".development-key-")
            .tempfile_in(directory)?;
        #[cfg(windows)]
        restrict_private_file(temporary.path())?;
        temporary.write_all(&encoded)?;
        temporary.as_file().sync_all()?;
        encoded.fill(0);
        match fs::hard_link(temporary.path(), &path) {
            Ok(()) => (),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
            Err(e) => return Err(e.into()),
        }
    }
    let publisher = Publisher::load(
        &path,
        None,
        "local-development".into(),
        "local-development".into(),
        None,
    )?;
    let profile = serde_json::json!([{"name":"Local development signer","namespace":"local-development","public_key":publisher.public_key(),"development":true}]);
    let mut temporary = tempfile::Builder::new()
        .prefix(".development-profile-")
        .tempfile_in(directory)?;
    temporary.write_all(serde_json::to_string_pretty(&profile)?.as_bytes())?;
    temporary.write_all(b"\n")?;
    temporary.as_file().sync_all()?;
    temporary.persist(directory.join("development-publisher.json"))?;
    super::archive::sync_directory(directory)?;
    Ok(
        serde_json::json!({"key_path":path,"public_key":publisher.public_key(),"fingerprint":publisher.fingerprint()}),
    )
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(export_key_info, module)?)?;
    module.add_function(wrap_pyfunction!(export_development_publisher, module)?)?;
    Ok(())
}

#[cfg(windows)]
fn restrict_private_file(path: &Path) -> Result<()> {
    use std::{os::windows::ffi::OsStrExt, ptr};
    use windows_sys::Win32::{
        Foundation::LocalFree,
        Security::{
            Authorization::{
                ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
            },
            DACL_SECURITY_INFORMATION, PROTECTED_DACL_SECURITY_INFORMATION, SetFileSecurityW,
        },
    };
    let text: Vec<u16> = "D:P(A;;FA;;;SY)(A;;FA;;;OW)\0".encode_utf16().collect();
    let path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut descriptor = ptr::null_mut();
    unsafe {
        if ConvertStringSecurityDescriptorToSecurityDescriptorW(
            text.as_ptr(),
            SDDL_REVISION_1,
            &mut descriptor,
            ptr::null_mut(),
        ) == 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
        let result = SetFileSecurityW(
            path.as_ptr(),
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            descriptor,
        );
        LocalFree(descriptor);
        if result == 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    Ok(())
}
#[cfg(windows)]
fn validate_private_file(path: &Path) -> Result<()> {
    use std::{os::windows::ffi::OsStrExt, ptr};
    use windows_sys::Win32::{
        Foundation::LocalFree,
        Security::{
            ACL,
            Authorization::{ConvertStringSidToSidW, GetNamedSecurityInfoW, SE_FILE_OBJECT},
            DACL_SECURITY_INFORMATION, EqualSid, GetAce, OWNER_SECURITY_INFORMATION,
        },
    };
    let path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut owner = ptr::null_mut();
    let mut acl: *mut ACL = ptr::null_mut();
    let mut descriptor = ptr::null_mut();
    let mut allowed = Vec::new();
    unsafe {
        let status = GetNamedSecurityInfoW(
            path.as_ptr(),
            SE_FILE_OBJECT,
            OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
            &mut owner,
            ptr::null_mut(),
            &mut acl,
            ptr::null_mut(),
            &mut descriptor,
        );
        if status != 0 {
            return Err(std::io::Error::from_raw_os_error(status as i32).into());
        }
        let result = (|| -> Result<()> {
            if owner.is_null() || acl.is_null() {
                return Err(invalid("signing key must have owner-only permissions"));
            }
            for text in ["S-1-3-4\0", "S-1-5-18\0"] {
                let text: Vec<u16> = text.encode_utf16().collect();
                let mut sid = ptr::null_mut();
                if ConvertStringSidToSidW(text.as_ptr(), &mut sid) == 0 {
                    return Err(std::io::Error::last_os_error().into());
                }
                allowed.push(sid);
            }
            for i in 0..(*acl).AceCount {
                let mut ace: *mut std::ffi::c_void = ptr::null_mut();
                if GetAce(acl, i as u32, &mut ace) == 0 {
                    return Err(std::io::Error::last_os_error().into());
                }
                let bytes = ace.cast::<u8>();
                let kind = *bytes;
                if kind == 1 {
                    continue;
                }
                let sid = bytes.add(8).cast();
                if kind != 0
                    || !std::iter::once(owner)
                        .chain(allowed.iter().copied())
                        .any(|allowed| EqualSid(sid, allowed) != 0)
                {
                    return Err(invalid("signing key must have owner-only permissions"));
                }
            }
            Ok(())
        })();
        for sid in allowed {
            LocalFree(sid);
        }
        LocalFree(descriptor);
        result
    }
}

#[cfg(not(openssl3))]
fn load_key_seed(encoded: &[u8], password: Option<&[u8]>) -> Result<Vec<u8>> {
    let key = match password {
        Some(password) => openssl::pkey::PKey::private_key_from_pem_passphrase(encoded, password),
        None => openssl::pkey::PKey::private_key_from_pem(encoded),
    }?;
    if key.id() != openssl::pkey::Id::ED25519 {
        return Err(invalid("publishing requires an Ed25519 private key"));
    }
    Ok(key.raw_private_key()?)
}

#[cfg(openssl3)]
fn load_key_seed(encoded: &[u8], password: Option<&[u8]>) -> Result<Vec<u8>> {
    use openssl_sys as ffi;
    use std::ptr;
    struct Import {
        context: *mut ffi::OSSL_LIB_CTX,
        default: *mut ffi::OSSL_PROVIDER,
        legacy: *mut ffi::OSSL_PROVIDER,
        decoder: *mut ffi::OSSL_DECODER_CTX,
        key: *mut ffi::EVP_PKEY,
    }
    impl Drop for Import {
        fn drop(&mut self) {
            unsafe {
                if !self.key.is_null() {
                    ffi::EVP_PKEY_free(self.key);
                }
                if !self.decoder.is_null() {
                    ffi::OSSL_DECODER_CTX_free(self.decoder);
                }
                if !self.legacy.is_null() {
                    ffi::OSSL_PROVIDER_unload(self.legacy);
                }
                if !self.default.is_null() {
                    ffi::OSSL_PROVIDER_unload(self.default);
                }
                if !self.context.is_null() {
                    ffi::OSSL_LIB_CTX_free(self.context);
                }
            }
        }
    }
    let mut import = Import {
        context: ptr::null_mut(),
        default: ptr::null_mut(),
        legacy: ptr::null_mut(),
        decoder: ptr::null_mut(),
        key: ptr::null_mut(),
    };
    let error = || {
        invalid(format!(
            "import Ed25519 PKCS#8 signing key: {}",
            openssl::error::ErrorStack::get()
        ))
    };
    // The private context keeps legacy key algorithms isolated from other OpenSSL callers.
    unsafe {
        import.context = ffi::OSSL_LIB_CTX_new();
        if import.context.is_null() {
            return Err(error());
        }
        import.default = ffi::OSSL_PROVIDER_load(import.context, c"default".as_ptr());
        if import.default.is_null() {
            return Err(error());
        }
        import.legacy = ffi::OSSL_PROVIDER_load(import.context, c"legacy".as_ptr());
        if import.legacy.is_null() {
            return Err(error());
        }
        import.decoder = ffi::OSSL_DECODER_CTX_new_for_pkey(
            &mut import.key,
            c"PEM".as_ptr(),
            ptr::null(),
            c"ED25519".as_ptr(),
            ffi::EVP_PKEY_KEYPAIR,
            import.context,
            ptr::null(),
        );
        if import.decoder.is_null() {
            return Err(error());
        }
        let password = password.unwrap_or(&[]);
        if ffi::OSSL_DECODER_CTX_set_passphrase(import.decoder, password.as_ptr(), password.len())
            != 1
        {
            return Err(error());
        }
        let mut data = encoded.as_ptr();
        let mut length = encoded.len();
        if ffi::OSSL_DECODER_from_data(import.decoder, &mut data, &mut length) != 1
            || import.key.is_null()
        {
            return Err(error());
        }
        if ffi::EVP_PKEY_id(import.key) != ffi::EVP_PKEY_ED25519 {
            return Err(invalid("publishing requires an Ed25519 private key"));
        }
        let mut seed = vec![0; 32];
        let mut length = seed.len();
        if ffi::EVP_PKEY_get_raw_private_key(import.key, seed.as_mut_ptr(), &mut length) != 1
            || length != 32
        {
            seed.fill(0);
            return Err(error());
        }
        Ok(seed)
    }
}

#[cfg(all(test, openssl3))]
mod key_context_tests {
    use super::*;
    #[cfg(unix)]
    #[test]
    fn insecure_signing_key_reports_owner_only_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("signing-key.pem");
        let pem = openssl::pkey::PKey::generate_ed25519()
            .unwrap()
            .private_key_to_pem_pkcs8()
            .unwrap();
        fs::write(&path, &pem).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        let load = || {
            Publisher::load(
                &path,
                None,
                "local-development".into(),
                "local-development".into(),
                None,
            )
        };
        let error = load().err().expect("insecure signing key must be rejected");
        assert!(error.to_string().contains("owner-only permissions"));
        assert_eq!(fs::read(&path).unwrap(), pem);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(load().is_ok());
    }
    fn global_des_available() -> bool {
        unsafe {
            let cipher = openssl_sys::EVP_CIPHER_fetch(
                std::ptr::null_mut(),
                c"DES-CBC".as_ptr(),
                std::ptr::null(),
            );
            let available = !cipher.is_null();
            if available {
                openssl_sys::EVP_CIPHER_free(cipher);
            }
            openssl::error::ErrorStack::get();
            available
        }
    }
    #[test]
    fn key_import_preserves_the_callers_default_algorithm_context() {
        openssl::init();
        let key = openssl::pkey::PKey::generate_ed25519().unwrap();
        let pem = key
            .private_key_to_pem_pkcs8_passphrase(
                openssl::symm::Cipher::aes_256_cbc(),
                b"fixture-password",
            )
            .unwrap();
        let before = global_des_available();
        let mut seed = load_key_seed(&pem, Some(b"fixture-password")).unwrap();
        assert_eq!(seed, key.raw_private_key().unwrap());
        seed.fill(0);
        assert_eq!(global_des_available(), before);
    }
}
