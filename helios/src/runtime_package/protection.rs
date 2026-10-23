//! Publisher signatures and acceptance-bound access to executable resources.
//!
//! Signature verification and publisher acceptance are important to user safety because
//! model packages contain executable GPU code. Signatures authenticate package contents
//! against a signing key; they do not sandbox GPU code.
//! Resolving who a signing key belongs to is the caller's task, from a `PackageClaim`.

use super::{archive::Archive, format::invalid, protobuf::wire};
use crate::Result;
use crate::os::FileReadAt;
use prost::Message;
use ring::{aead, digest, hkdf, signature};
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::{self, Read},
    path::Path,
    sync::Arc,
};

pub(super) const CHUNK: usize = 4 << 20;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Publisher {
    pub name: String,
    pub namespace: String,
    pub public_key: [u8; 32],
    #[serde(default)]
    pub development: bool,
}
impl Publisher {
    pub fn fingerprint(&self) -> String {
        hex(&sha256(&self.public_key))
    }
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty()
            || self.name.len() > 256
            || self.name.chars().any(char::is_control)
            || !namespace_valid(&self.namespace)
            || self.public_key == [0; 32]
            || (self.development && self.namespace != "local-development")
        {
            return Err(invalid("invalid publisher profile"));
        }
        Ok(())
    }
}
pub(super) fn namespace_valid(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 96
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
/// The official source is a repository under the publisher's Hugging Face namespace, or the
/// literal `local-development` marker for development packages.
pub(super) fn source_valid(namespace: &str, development: bool, source: &str) -> bool {
    if development {
        return namespace == "local-development" && source == "local-development";
    }
    let Some(repository) = source.strip_prefix(&format!("https://huggingface.co/{namespace}/"))
    else {
        return false;
    };
    !repository.is_empty()
        && repository.len() <= 96
        && repository.as_bytes()[0].is_ascii_alphanumeric()
        && repository
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}
/// Links are printed to a terminal before acceptance: HTTPS, bounded, and printable ASCII only.
fn link_valid(link: &str) -> bool {
    link.len() <= 512 && link.starts_with("https://") && link.bytes().all(|b| b.is_ascii_graphic())
}

/// The license governing use of the model, as the publisher recorded it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackageLicense {
    pub name: String,
    /// Where the license text is published; empty when the publisher recorded none.
    pub link: String,
}
impl PackageLicense {
    fn validated(license: &wire::License) -> Result<Self> {
        if license.name.is_empty()
            || license.name.len() > 128
            || license.name.chars().any(char::is_control)
            || !(license.link.is_empty() || link_valid(&license.link))
        {
            return Err(invalid("invalid package license metadata"));
        }
        Ok(Self {
            name: license.name.clone(),
            link: license.link.clone(),
        })
    }
}
/// What a package claims before any signature is verified: which key signed it and where it says
/// it was published. The caller resolves the signer's identity from the claim, then verifies the
/// package with the resolved profile; nothing here is trusted until that verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackageClaim {
    /// Lowercase SHA-256 fingerprint of the signing key.
    pub signer: String,
    pub namespace: String,
    pub official_source: String,
    pub development: bool,
}
impl PackageClaim {
    pub fn read(path: &Path) -> Result<Self> {
        let (_, _, _, bytes) = Archive::envelope(path)?;
        let envelope = wire::ProtectedIndex::decode(bytes.as_slice())
            .map_err(|e| invalid(format!("invalid signed envelope: {e}")))?;
        if envelope.version != 4 || envelope.signer.len() != 32 || envelope.groups.is_empty() {
            return Err(invalid(
                "signed format-4 package required; re-export the checkpoint",
            ));
        }
        let metadata = wire::PublisherMetadata::decode(envelope.groups[0].body.as_slice())
            .map_err(|e| invalid(format!("invalid signed metadata: {e}")))?;
        if !namespace_valid(&metadata.namespace)
            || !source_valid(
                &metadata.namespace,
                metadata.development,
                &metadata.official_source,
            )
        {
            return Err(invalid(
                "package publisher namespace or official source mismatch",
            ));
        }
        Ok(Self {
            signer: hex(&envelope.signer),
            namespace: metadata.namespace,
            official_source: metadata.official_source,
            development: metadata.development,
        })
    }
}

pub(super) fn sha256(bytes: &[u8]) -> Vec<u8> {
    digest::digest(&digest::SHA256, bytes).as_ref().to_vec()
}
pub(super) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn fields(values: &[&[u8]]) -> Vec<u8> {
    let mut result = Vec::new();
    for value in values {
        result.extend_from_slice(&(value.len() as u64).to_le_bytes());
        result.extend_from_slice(value);
    }
    result
}
fn signature_message(package: &[u8], signer: &[u8], kind: u32, body: &[u8]) -> Vec<u8> {
    fields(&[
        b"flare-package-signature-v4",
        &4u32.to_le_bytes(),
        package,
        signer,
        &kind.to_le_bytes(),
        body,
    ])
}
// Preserve the signing-key, publisher, package, and resource binding: it is
// important to user safety when executing model packages.
fn resource_key(
    public_key: &[u8],
    package: &[u8],
    offset: u64,
    namespace: &str,
) -> Result<aead::LessSafeKey> {
    let info = fields(&[
        b"flare-executable-v3",
        &offset.to_le_bytes(),
        namespace.as_bytes(),
    ]);
    let prk = hkdf::Salt::new(hkdf::HKDF_SHA256, package).extract(public_key);
    let parts = [info.as_slice()];
    let mut key = [0; 32];
    prk.expand(&parts, hkdf::HKDF_SHA256)
        .map_err(|_| invalid("derive executable key"))?
        .fill(&mut key)
        .map_err(|_| invalid("derive executable key"))?;
    Ok(aead::LessSafeKey::new(
        aead::UnboundKey::new(&aead::AES_256_GCM, &key)
            .map_err(|_| invalid("invalid executable key"))?,
    ))
}
fn aad(package: &[u8], offset: u64, size: u64, chunk: u64) -> Vec<u8> {
    fields(&[
        b"flare-executable-chunk-v3",
        package,
        &offset.to_le_bytes(),
        &size.to_le_bytes(),
        &chunk.to_le_bytes(),
    ])
}

pub struct VerifiedPackage {
    pub(super) archive: Archive,
    pub(super) index: wire::ModelIndex,
    publisher: Publisher,
    official_source: String,
}
pub struct AuthorizedPackage {
    pub(super) archive: Archive,
    pub(super) index: wire::ModelIndex,
}
fn publisher_profiles(publishers: &[Publisher]) -> Result<Vec<Publisher>> {
    let mut profiles = super::publishers::embedded()?;
    for publisher in publishers {
        publisher.validate()?;
        if let Some(previous) = profiles
            .iter()
            .find(|p| p.public_key == publisher.public_key)
        {
            if previous != publisher {
                return Err(invalid("conflicting identities for the same publisher key"));
            }
        } else {
            profiles.push(publisher.clone());
        }
    }
    Ok(profiles)
}

/// Authenticate exact envelope bytes against publisher profiles without opening payload files.
/// The payload length is the signed logical primary-file extent.
pub fn verify_package_metadata(
    bytes: &[u8],
    publishers: &[Publisher],
    payload_length: u64,
) -> Result<(wire::ModelIndex, Publisher)> {
    let profiles = publisher_profiles(publishers)?;
    let (protection, index) = Protection::verify(bytes, &profiles, payload_length, None)?;
    Ok((index, protection.publisher))
}

impl VerifiedPackage {
    pub(super) fn verified_rank(&self) -> Option<usize> {
        self.archive.protection.rank
    }

    /// Authenticate the package's index and publisher binding, for every rank or only `rank` of
    /// a multi-host package, which a copy holding that rank's bytes alone verifies completely.
    pub fn open(path: &Path, publishers: &[Publisher], rank: Option<usize>) -> Result<Self> {
        let profiles = publisher_profiles(publishers)?;
        let (archive, index) = Archive::open(path, &profiles, rank)?;
        let publisher = archive.protection.publisher.clone();
        let official_source = archive.protection.official_source.clone();
        Ok(Self {
            archive,
            index,
            publisher,
            official_source,
        })
    }
    pub fn publisher(&self) -> &Publisher {
        &self.publisher
    }
    pub fn official_source(&self) -> &str {
        &self.official_source
    }
    pub fn license(&self) -> Option<&PackageLicense> {
        self.archive.protection.license.as_ref()
    }
    /// Digest the stored data of every verified rank against the signed index.
    pub fn verify_stored_data(&self) -> Result<()> {
        self.archive.protection.verify_data(&self.archive)
    }
    /// Record the string the user applied to accept the publisher's terms. It derives every
    /// executable key, so a string other than the namespace the exporter encrypted under opens
    /// nothing; the caller obtains it from the user or from remembered acceptance.
    pub fn authorize(mut self, accepted: &str, verify_weights: bool) -> Result<AuthorizedPackage> {
        if verify_weights {
            self.archive.protection.verify_data(&self.archive)?;
        }
        self.archive.protection.accepted = Some(accepted.to_owned());
        Ok(AuthorizedPackage {
            archive: self.archive,
            index: self.index,
        })
    }
}

pub(super) struct Protection {
    publisher: Publisher,
    official_source: String,
    license: Option<PackageLicense>,
    /// The acceptance string, once applied; executable keys derive from it.
    accepted: Option<String>,
    package_id: Vec<u8>,
    resources: BTreeMap<u64, (bool, wire::ProtectedResource)>,
    /// Stored-data ranges: what every rank shares, then each verified rank's in index order.
    data: Vec<wire::StoredData>,
    /// The one rank of a multi-host package verified, whose bundle alone the index then holds.
    rank: Option<usize>,
}
fn add_catalog(
    resources: &mut BTreeMap<u64, (bool, wire::ProtectedResource)>,
    catalog: wire::ResourceCatalog,
    encrypted: bool,
    payload_size: u64,
) -> Result<()> {
    for entry in catalog.resources {
        let count = if encrypted {
            entry.plain_bytes.div_ceil(CHUNK as u64)
        } else {
            1
        };
        let expected = entry.plain_bytes.checked_add(if encrypted {
            count
                .checked_mul(16)
                .ok_or_else(|| invalid("resource size overflow"))?
        } else {
            0
        });
        if entry.plain_bytes == 0
            || expected != Some(entry.stored_bytes)
            || count != entry.sha256.len() as u64
            || entry.sha256.iter().any(|h| h.len() != 32)
            || entry
                .offset
                .checked_add(entry.stored_bytes)
                .is_none_or(|end| end > payload_size)
            || resources.insert(entry.offset, (encrypted, entry)).is_some()
        {
            return Err(invalid("invalid protected resource catalog"));
        }
    }
    Ok(())
}
fn stored_data(
    mut data: wire::StoredData,
    payload_size: u64,
    version: u32,
) -> Result<wire::StoredData> {
    if data.sha256.len() != 32 {
        return Err(invalid("invalid stored-data signature group"));
    }
    if data.parts.is_empty() {
        if data
            .offset
            .checked_add(data.nbytes)
            .is_none_or(|end| end > payload_size)
        {
            return Err(invalid("invalid stored-data signature group"));
        }
        if data.nbytes > 0 {
            data.parts.push(wire::Span {
                offset: data.offset,
                nbytes: data.nbytes,
                filename: String::new(),
            });
        }
    } else if data.offset != 0
        || data
            .parts
            .iter()
            .try_fold(0u64, |sum, part| sum.checked_add(part.nbytes))
            != Some(data.nbytes)
    {
        return Err(invalid("invalid stored-data parts"));
    }
    for part in &data.parts {
        if !part.filename.is_empty() {
            super::archive::validate_filename(&part.filename)?;
            if version != 5 {
                return Err(invalid("companion references require model index 5"));
            }
        }
        if part.nbytes == 0
            || part
                .offset
                .checked_add(part.nbytes)
                .is_none_or(|end| part.filename.is_empty() && end > payload_size)
        {
            return Err(invalid("invalid stored-data part"));
        }
    }
    Ok(data)
}

impl Protection {
    /// Authenticate the envelope: what every rank shares, then every rank, or only `rank` of a
    /// multi-host package, whose host holds that rank's bytes alone.
    pub fn verify(
        bytes: &[u8],
        publishers: &[Publisher],
        payload_size: u64,
        rank: Option<usize>,
    ) -> Result<(Self, wire::ModelIndex)> {
        let envelope = wire::ProtectedIndex::decode(bytes)
            .map_err(|e| invalid(format!("invalid signed envelope: {e}")))?;
        if envelope.version != 4
            || envelope.package_id.len() != 32
            || envelope.signer.len() != 32
            || envelope.groups.len() != 3
            || envelope.ranks.is_empty()
        {
            return Err(invalid(
                "signed format-4 package required; re-export the checkpoint",
            ));
        }
        let publisher = publishers
            .iter()
            .find(|p| sha256(&p.public_key) == envelope.signer)
            .ok_or_else(|| {
                invalid(
                    "unknown package publisher; supply a matching publisher verification profile",
                )
            })?
            .clone();
        publisher.validate()?;
        let authenticate = |kind: u32, group: &wire::SignedGroup, label: String| -> Result<()> {
            if group.kind != kind {
                return Err(invalid("missing, duplicate, or unordered signature group"));
            }
            signature::UnparsedPublicKey::new(&signature::ED25519, publisher.public_key)
                .verify(
                    &signature_message(
                        &envelope.package_id,
                        &envelope.signer,
                        group.kind,
                        &group.body,
                    ),
                    &group.signature,
                )
                .map_err(|_| invalid(format!("publisher signature failed for {label}")))
        };
        for (i, group) in envelope.groups.iter().enumerate() {
            authenticate(i as u32 + 1, group, format!("group {}", i + 1))?;
        }
        let metadata = wire::PublisherMetadata::decode(envelope.groups[0].body.as_slice())
            .map_err(|e| invalid(format!("invalid signed metadata: {e}")))?;
        if metadata.group_hashes.len() != 2
            || metadata
                .group_hashes
                .iter()
                .zip(&envelope.groups[1..])
                .any(|(hash, group)| *hash != sha256(&group.body))
        {
            return Err(invalid("signature group commitment mismatch"));
        }
        if metadata.namespace != publisher.namespace
            || metadata.development != publisher.development
            || !source_valid(
                &publisher.namespace,
                publisher.development,
                &metadata.official_source,
            )
        {
            return Err(invalid(
                "package publisher namespace or official source mismatch",
            ));
        }
        // Packages exported before the field existed carry no license; it stays absent.
        let license = metadata
            .license
            .as_ref()
            .map(PackageLicense::validated)
            .transpose()?;
        let mut index = metadata
            .index
            .ok_or_else(|| invalid("missing signed model index"))?;
        let count = metadata.ranks as usize;
        if !matches!(index.version, 4 | 5)
            || !index.ranks.is_empty()
            || envelope.ranks.len() != count
            || index
                .cluster
                .as_ref()
                .is_some_and(|c| c.world as usize != count)
        {
            return Err(invalid("unsupported model index"));
        }
        let verified = match rank {
            None => 0..count,
            Some(rank) if index.cluster.is_some() && rank < count => rank..rank + 1,
            Some(rank) => {
                return Err(invalid(format!(
                    "rank {rank} is not a separately served rank of this package"
                )));
            }
        };
        let mut resources = BTreeMap::new();
        let shared = wire::ResourceCatalog::decode(envelope.groups[1].body.as_slice())
            .map_err(|e| invalid(e.to_string()))?;
        add_catalog(&mut resources, shared, false, payload_size)?;
        let shared = wire::StoredData::decode(envelope.groups[2].body.as_slice())
            .map_err(|e| invalid(e.to_string()))?;
        let mut data = vec![stored_data(shared, payload_size, index.version)?];
        for rank in verified {
            let group = &envelope.ranks[rank];
            authenticate(4, group, format!("rank {rank}"))?;
            let signed = wire::RankProtection::decode(group.body.as_slice())
                .map_err(|e| invalid(format!("invalid signed rank: {e}")))?;
            let (Some(bundle), Some(stored)) = (signed.bundle, signed.data) else {
                return Err(invalid("incomplete signed rank"));
            };
            if signed.rank as usize != rank {
                return Err(invalid("signed ranks out of order"));
            }
            add_catalog(
                &mut resources,
                signed.executables.unwrap_or_default(),
                true,
                payload_size,
            )?;
            add_catalog(
                &mut resources,
                signed.support.unwrap_or_default(),
                false,
                payload_size,
            )?;
            data.push(stored_data(stored, payload_size, index.version)?);
            index.ranks.push(bundle);
        }
        let protection = Self {
            publisher,
            official_source: metadata.official_source,
            license,
            accepted: None,
            package_id: envelope.package_id,
            resources,
            data,
            rank,
        };
        protection.validate_membership(&index, payload_size)?;
        Ok((protection, index))
    }
    /// The index position of `rank`'s bundle: the rank itself, or the only one held when a host
    /// verified its rank alone.
    pub fn position(&self, rank: usize) -> Result<usize> {
        match self.rank {
            None => Ok(rank),
            Some(only) if only == rank => Ok(0),
            Some(only) => Err(invalid(format!(
                "only rank {only} of this package was verified"
            ))),
        }
    }
    /// Payload ranges of the stored data that serving the bundle at `position` reads: what every
    /// rank shares, then that rank's.
    pub fn stored_ranges(&self, position: usize) -> Vec<wire::Span> {
        [&self.data[0], &self.data[position + 1]]
            .into_iter()
            .flat_map(|range| range.parts.iter().cloned())
            .collect()
    }
    fn validate_membership(&self, index: &wire::ModelIndex, payload_size: u64) -> Result<()> {
        let ranges: Vec<&wire::Span> = self.data.iter().flat_map(|range| &range.parts).collect();
        let within = |range: &wire::StoredData, span: &wire::Span| -> Result<()> {
            if !range.parts.iter().any(|part| {
                span.filename == part.filename
                    && span.offset >= part.offset
                    && span
                        .offset
                        .checked_add(span.nbytes)
                        .is_some_and(|end| end <= part.offset + part.nbytes)
            }) {
                return Err(invalid("stored allocation outside its data range"));
            }
            Ok(())
        };
        let mut spans: Vec<(u32, &wire::Span)> = Vec::new();
        for (position, rank) in index.ranks.iter().enumerate() {
            let mut roles = HashMap::new();
            let mut add = |name: &str, kind: u32| -> Result<()> {
                if !rank.resources.contains_key(name)
                    || roles
                        .insert(name.to_owned(), kind)
                        .is_some_and(|old| old != kind)
                {
                    return Err(invalid("missing or conflicting resource classification"));
                }
                Ok(())
            };
            for module in &rank.modules {
                add(&module.file, 2)?;
                for global in &module.globals {
                    add(&global.file, 3)?;
                }
            }
            for graph in &rank.graphs {
                add(&graph.resource, 2)?;
            }
            for allocation in &rank.allocations {
                if allocation
                    .data_spans
                    .iter()
                    .any(|span| !span.filename.is_empty())
                {
                    return Err(invalid("allocation destination spans cannot name a file"));
                }
                if let Some(file) = &allocation.file {
                    add(file, 4)?;
                }
            }
            if roles.len() != rank.resources.len() {
                return Err(invalid("unclassified package resource"));
            }
            for (name, span) in &rank.resources {
                if roles[name] == 4 {
                    within(&self.data[position + 1], span)?;
                }
                spans.push((roles[name], span));
            }
        }
        for asset in &index.assets {
            let span = asset
                .payload
                .as_ref()
                .ok_or_else(|| invalid("missing asset range"))?;
            let (_, resource) = self
                .resources
                .get(&span.offset)
                .ok_or_else(|| invalid("unsigned asset"))?;
            if resource.sha256.len() != 1 || asset.sha256 != hex(&resource.sha256[0]) {
                return Err(invalid("asset digest differs from publisher catalog"));
            }
            spans.push((3, span));
        }
        for (owner, span) in super::engram::spans(index, self.rank)? {
            let position = match owner {
                None => 0,
                Some(rank) => self.position(rank)? + 1,
            };
            let data = self
                .data
                .get(position)
                .ok_or_else(|| invalid("embedding partition owner is absent"))?;
            within(data, span)?;
            spans.push((4, span));
        }
        spans.sort_by_key(|(_, s)| (&s.filename, s.offset));
        let mut end = 0;
        let mut protected = 0;
        let mut filename = "";
        let mut primary_end = 0;
        for (kind, span) in spans {
            if span.filename != filename {
                end = 0;
                filename = &span.filename;
            }
            if !span.filename.is_empty() {
                super::archive::validate_filename(&span.filename)?;
                if index.version != 5 || kind != 4 {
                    return Err(invalid(
                        "only stored data in model index 5 may reference a companion",
                    ));
                }
            }
            if span.nbytes == 0 || span.offset < end {
                return Err(invalid("empty or overlapping package resources"));
            }
            end = span
                .offset
                .checked_add(span.nbytes)
                .filter(|n| !span.filename.is_empty() || *n <= payload_size)
                .ok_or_else(|| invalid("resource exceeds package"))?;
            if span.filename.is_empty() {
                primary_end = end;
            }
            if kind != 4 {
                let (encrypted, entry) = self
                    .resources
                    .get(&span.offset)
                    .ok_or_else(|| invalid("unsigned executable or supporting resource"))?;
                if *encrypted != (kind == 2) || entry.stored_bytes != span.nbytes {
                    return Err(invalid("resource signature classification mismatch"));
                }
                if ranges.iter().any(|range| {
                    range.filename == span.filename
                        && span.offset < range.offset + range.nbytes
                        && range.offset < end
                }) {
                    return Err(invalid(
                        "executable or supporting resource inside stored data",
                    ));
                }
                protected += 1;
            }
        }
        let mut ordered = ranges.clone();
        ordered.sort_by_key(|range| (&range.filename, range.offset));
        if ordered.windows(2).any(|pair| {
            pair[1].filename == pair[0].filename && pair[1].offset < pair[0].offset + pair[0].nbytes
        }) {
            return Err(invalid("overlapping stored-data ranges"));
        }
        // A host verifying one rank cannot see the others' resources, which complete the payload.
        if protected != self.resources.len() || (self.rank.is_none() && primary_end != payload_size)
        {
            return Err(invalid("unreferenced package resources"));
        }
        Ok(())
    }
    pub fn data_parts(&self) -> impl Iterator<Item = &wire::Span> {
        self.data.iter().flat_map(|data| &data.parts)
    }
    fn verify_data(&self, archive: &Archive) -> Result<()> {
        let mut block = vec![0; CHUNK];
        for range in &self.data {
            let mut hash = digest::Context::new(&digest::SHA256);
            for part in &range.parts {
                let mut pos = 0;
                while pos < part.nbytes {
                    let count = (part.nbytes - pos).min(block.len() as u64) as usize;
                    archive.read_logical(part, pos, &mut block[..count])?;
                    hash.update(&block[..count]);
                    pos += count as u64;
                }
            }
            if hash.finish().as_ref() != range.sha256 {
                return Err(invalid("stored-data content verification failed"));
            }
        }
        Ok(())
    }
    pub fn reader(&self, file: Arc<File>, start: u64, span: &wire::Span) -> Result<ResourceReader> {
        let (encrypted, entry) = self
            .resources
            .get(&span.offset)
            .ok_or_else(|| invalid("resource lacks publisher authentication"))?;
        if !span.filename.is_empty() || span.nbytes != entry.stored_bytes {
            return Err(invalid("resource length mismatch"));
        }
        let accepted = self
            .accepted
            .as_deref()
            .ok_or_else(|| invalid("publisher acceptance required before resource access"))?;
        let key = encrypted
            .then(|| {
                resource_key(
                    &self.publisher.public_key,
                    &self.package_id,
                    entry.offset,
                    accepted,
                )
            })
            .transpose()?;
        Ok(ResourceReader {
            file,
            start,
            entry: entry.clone(),
            package_id: self.package_id.clone(),
            key,
            chunk: 0,
            buffer: Vec::new(),
            position: 0,
        })
    }
}

pub(super) struct ResourceReader {
    file: Arc<File>,
    start: u64,
    entry: wire::ProtectedResource,
    package_id: Vec<u8>,
    key: Option<aead::LessSafeKey>,
    chunk: usize,
    buffer: Vec<u8>,
    position: usize,
}
impl ResourceReader {
    pub fn len(&self) -> u64 {
        self.entry.plain_bytes
    }
    fn refill(&mut self) -> Result<bool> {
        if self.chunk == self.entry.sha256.len() {
            return Ok(false);
        }
        let encrypted = self.key.is_some();
        let plain_offset = if encrypted {
            self.chunk as u64 * CHUNK as u64
        } else {
            0
        };
        let count = if encrypted {
            (self.entry.plain_bytes - plain_offset).min(CHUNK as u64) + 16
        } else {
            self.entry.plain_bytes
        };
        self.buffer.resize(super::protobuf::size(count)?, 0);
        self.file
            .read_exact_at(
                &mut self.buffer,
                self.start + plain_offset + if encrypted { self.chunk as u64 * 16 } else { 0 },
            )
            .map_err(|e| e.to_string())?;
        if sha256(&self.buffer) != self.entry.sha256[self.chunk] {
            return Err(invalid("publisher resource content verification failed"));
        }
        if let Some(key) = &self.key {
            let mut nonce = [0; 12];
            nonce[4..].copy_from_slice(&(self.chunk as u64).to_be_bytes());
            let length = key
                .open_in_place(
                    aead::Nonce::assume_unique_for_key(nonce),
                    aead::Aad::from(aad(
                        &self.package_id,
                        self.entry.offset,
                        self.entry.plain_bytes,
                        self.chunk as u64,
                    )),
                    &mut self.buffer,
                )
                .map_err(|_| invalid("executable decryption failed"))?
                .len();
            self.buffer.truncate(length);
        }
        self.chunk += 1;
        self.position = 0;
        Ok(true)
    }
}
impl Read for ResourceReader {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        if self.position == self.buffer.len()
            && !self.refill().map_err(|e| io::Error::other(e.to_string()))?
        {
            return Ok(0);
        }
        let n = output.len().min(self.buffer.len() - self.position);
        output[..n].copy_from_slice(&self.buffer[self.position..self.position + n]);
        self.position += n;
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialized_primary_spans_and_stored_ranges_keep_their_legacy_meaning() {
        let span = wire::Span::decode(&b"\x08\x09\x10\x07"[..]).unwrap();
        assert_eq!(
            (span.offset, span.nbytes, span.filename.as_str()),
            (9, 7, "")
        );
        let mut bytes = vec![8, 7, 18, 32];
        bytes.extend([0; 32]);
        bytes.extend([24, 9]);
        let data = wire::StoredData::decode(bytes.as_slice()).unwrap();
        assert!(data.parts.is_empty());
        let normalized = stored_data(data.clone(), 16, 4).unwrap();
        assert_eq!(normalized.parts, vec![span]);
        assert!(stored_data(data, 15, 4).is_err());
        let empty = wire::StoredData {
            sha256: sha256(&[]).to_vec(),
            ..Default::default()
        };
        assert!(stored_data(empty, 0, 4).unwrap().parts.is_empty());
    }
}
