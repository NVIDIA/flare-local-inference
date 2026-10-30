//! Assemble signed-payload package files from captured source ranges.
//!
//! Capture records where each semantic weight group's bytes live in the rank's
//! logical payload concatenation. This module turns those ranges into payload
//! files, computes their content identities over the exact payload bytes, and
//! records the initializers that place them back into allocations. It performs
//! no container framing or cache management of its own: payload
//! files use [`container`], publication uses [`crate::package::cache`], and the
//! caller owns signatures.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::package::container::{self, Slice, Source, Writer as ContainerWriter};
use crate::package::protection::{compress_stream, EnvelopeBuilder};
use crate::package::schema::{self, wire};
use crate::package::{PayloadId, invalid};
use prost::Message;

fn io(context: &str, error: std::io::Error) -> crate::Error {
    format!("{context}: {error}").into()
}

/// One range of a payload, placed at `destination_offset` in `allocation`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturedSpan {
    pub allocation: String,
    pub destination_offset: u64,
    pub source_offset: u64,
    pub nbytes: u64,
}

/// One semantic payload to assemble from captured source ranges.
#[derive(Clone, Debug)]
pub struct CapturedGroup {
    pub name: String,
    /// Role the caller records in the signed payload descriptor.
    pub role: i32,
    /// Package-relative companion path, for example `weights/layer-000.safetensors`.
    pub relative: String,
    pub spans: Vec<CapturedSpan>,
}

/// One contiguous source file range in the rank's logical payload concatenation.
#[derive(Clone, Debug)]
pub struct CapturedSource {
    pub offset: u64,
    pub path: PathBuf,
    pub file_offset: u64,
    pub nbytes: u64,
}

#[derive(Clone, Debug)]
pub struct CapturedRank {
    pub rank: u32,
    pub groups: Vec<CapturedGroup>,
    pub sources: Vec<CapturedSource>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssembledPayload {
    pub name: String,
    pub id: PayloadId,
    pub length: u64,
    pub role: i32,
    pub relative: String,
    pub path: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssembledInitializer {
    pub allocation: String,
    pub destination_offset: u64,
    pub payload: PayloadId,
    pub payload_offset: u64,
    pub nbytes: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AssembledRank {
    pub rank: u32,
    pub payloads: Vec<AssembledPayload>,
    pub initializers: Vec<AssembledInitializer>,
}

/// Positions reads inside the rank's logical payload concatenation.
struct Sources<'a> {
    parts: &'a [CapturedSource],
}

impl<'a> Sources<'a> {
    fn new(parts: &'a [CapturedSource]) -> Result<Self, crate::Error> {
        let mut end = 0;
        for part in parts {
            if part.offset < end || part.nbytes == 0 {
                return Err(invalid("captured sources must be ordered and nonempty"));
            }
            end = part
                .offset
                .checked_add(part.nbytes)
                .ok_or_else(|| invalid("captured source extent overflow"))?;
        }
        Ok(Self { parts })
    }

    fn copy(&self, offset: u64, length: u64, output: &mut impl Write) -> Result<(), crate::Error> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| invalid("captured source range overflow"))?;
        let mut cursor = offset;
        while cursor < end {
            let part = self
                .parts
                .iter()
                .find(|part| part.offset <= cursor && cursor < part.offset + part.nbytes)
                .ok_or_else(|| invalid("captured source range is not covered"))?;
            let available = (part.offset + part.nbytes - cursor).min(end - cursor);
            let mut file = File::open(&part.path).map_err(|e| io("open captured source", e))?;
            file.seek(SeekFrom::Start(part.file_offset + (cursor - part.offset)))
                .map_err(|e| io("seek captured source", e))?;
            let mut remaining = available;
            let mut buffer = vec![0u8; 1 << 20];
            while remaining > 0 {
                let chunk = remaining.min(buffer.len() as u64) as usize;
                file.read_exact(&mut buffer[..chunk])
                    .map_err(|e| io("read captured source", e))?;
                output
                    .write_all(&buffer[..chunk])
                    .map_err(|e| io("write payload", e))?;
                remaining -= chunk as u64;
            }
            cursor += available;
        }
        Ok(())
    }
}

/// Assemble every declared payload of every captured rank into `directory`.
///
/// Each payload's identity is the SHA-256 of its exact payload bytes, and its
/// file is the deterministic one-payload companion wrapper. Bytes that no span
/// covers stay absent: allocations are zero-filled on load.
pub fn assemble(
    directory: &Path,
    ranks: &[CapturedRank],
) -> Result<Vec<AssembledRank>, crate::Error> {
    let mut assembled = Vec::with_capacity(ranks.len());
    for rank in ranks {
        let sources = Sources::new(&rank.sources)?;
        let mut seen = std::collections::BTreeSet::new();
        let mut payloads = Vec::with_capacity(rank.groups.len());
        let mut initializers = Vec::new();
        for group in &rank.groups {
            if group.name.is_empty() || !seen.insert(group.relative.clone()) {
                return Err(invalid("duplicate or empty payload name"));
            }
            let length: u64 = group.spans.iter().try_fold(0u64, |total, span| {
                if span.nbytes == 0 {
                    return Err(invalid("payload span must be nonempty"));
                }
                total
                    .checked_add(span.nbytes)
                    .ok_or_else(|| invalid("payload length overflow"))
            })?;
            if length == 0 {
                return Err(invalid("payload has no spans"));
            }
            let path = directory.join(&group.relative);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| io("create payload directory", e))?;
            }
            let staging = path.with_extension("payload");
            let mut output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&staging)
                .map_err(|e| io("create payload file", e))?;
            let mut hasher = Sha256::new();
            for span in &group.spans {
                let mut writer = HashingWriter {
                    inner: &mut output,
                    hasher: &mut hasher,
                };
                sources.copy(span.source_offset, span.nbytes, &mut writer)?;
            }
            output.sync_all().map_err(|e| io("sync payload file", e))?;
            drop(output);
            let id = PayloadId::from_bytes(hasher.finish());
            let staged = File::open(&staging).map_err(|e| io("open assembled payload", e))?;
            let companion = Slice::new(std::sync::Arc::new(staged), 0, length)?;
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|e| io("create payload companion", e))?;
            container::write_companion(&mut file, &companion)?;
            file.sync_all()
                .map_err(|e| io("sync payload companion", e))?;
            std::fs::remove_file(&staging).map_err(|e| io("remove assembled payload", e))?;
            let mut payload_offset = 0;
            for span in &group.spans {
                initializers.push(AssembledInitializer {
                    allocation: span.allocation.clone(),
                    destination_offset: span.destination_offset,
                    payload: id,
                    payload_offset,
                    nbytes: span.nbytes,
                });
                payload_offset += span.nbytes;
            }
            payloads.push(AssembledPayload {
                name: group.name.clone(),
                id,
                length,
                role: group.role,
                relative: group.relative.clone(),
                path,
            });
        }
        assembled.push(AssembledRank {
            rank: rank.rank,
            payloads,
            initializers,
        });
    }
    Ok(assembled)
}

/// Where published payloads go: a plain directory or an immutable cache entry.
pub enum Destination<'a> {
    Directory(&'a Path),
    Cache(
        &'a crate::package::cache::Root,
        &'a mut crate::package::cache::Staging,
    ),
}

/// One payload supplied by the caller as pinned stored bytes.
///
/// Length is the slice length: there is no separate path/nbytes pair, so the
/// bytes hashed for verification are exactly the bytes published.
#[derive(Clone)]
pub struct PublishPayload {
    pub role: wire::PayloadRole,
    /// Package-relative companion path, or empty when the primary holds it.
    pub relative: String,
    pub source: Slice,
    /// Identity the caller assigned from these bytes; publish refuses a mismatch.
    pub id: PayloadId,
    /// Primary tensor name for package-scoped payloads such as frontend assets.
    pub tensor: Option<String>,
}

/// Pin one file region as a publication source.
pub fn pin_file(path: &Path, nbytes: u64) -> Result<Slice, crate::Error> {
    let file = File::open(path).map_err(|e| io("open payload", e))?;
    Slice::new(std::sync::Arc::new(file), 0, nbytes)
}

/// One plaintext executable resource: a cubin or captured graph bytes.
#[derive(Clone, Debug)]
pub struct PublishResource {
    pub kind: wire::ExecutableKind,
    pub path: PathBuf,
    pub nbytes: u64,
}

pub struct PublishRank {
    /// Rank placement and execution geometry. `resources` must be empty.
    pub body: wire::RankBody,
    pub payloads: Vec<PublishPayload>,
    pub resources: Vec<PublishResource>,
}

pub struct PublishDeployment {
    pub body: wire::DeploymentBody,
    pub ranks: Vec<PublishRank>,
    /// Deployment-scoped payloads such as engram planes.
    pub payloads: Vec<PublishPayload>,
}

pub struct PublishRequest {
    pub package: wire::PackageBody,
    /// Package-scoped payloads such as frontend assets.
    pub payloads: Vec<PublishPayload>,
    pub deployments: Vec<PublishDeployment>,
    pub package_id: [u8; 32],
    pub public_key: [u8; 32],
    /// Accepted publisher namespace; must equal the signed publisher namespace.
    pub accepted: String,
    /// Reuse weight, tuned, and initial-state payloads already in the cache.
    pub no_weights: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublishedPayload {
    pub id: PayloadId,
    pub nbytes: u64,
    pub role: i32,
    /// Companion path, or empty when the primary holds the payload.
    pub relative: String,
    /// Primary tensor name when the payload is embedded in the primary.
    pub tensor: Option<String>,
}

#[derive(Debug)]
pub struct Published {
    /// Serialized signed envelope, also written as the primary's manifest tensor.
    pub manifest: Vec<u8>,
    pub payloads: Vec<PublishedPayload>,
    pub directory: PathBuf,
}

/// Publish one prepared package in the directory canonical form.
///
/// Executable resources are stored as independent zstd frames; weight and state
/// payloads are installed verbatim after their bytes match the signed identity.
/// Bodies are committed and signed with the publisher key. Container framing,
/// content hashing, and cache publication come from their owning modules.
pub fn publish(
    request: &PublishRequest,
    key: &ring::signature::Ed25519KeyPair,
    destination: Destination<'_>,
) -> Result<Published, crate::Error> {
    publish_inner(request, key, destination, None)
}

/// Publish using immutable cached weight objects selected from a previous package.
///
/// The caller authenticates the previous package and checks weight-layout compatibility.
/// Weight sources must be the exact pinned slices of the supplied objects. Publication
/// links those objects without hashing their bytes; fresh data retains full verification.
pub fn publish_reusing_cached_weights(
    request: &PublishRequest,
    key: &ring::signature::Ed25519KeyPair,
    destination: Destination<'_>,
    weights: &[crate::package::cache::PayloadObject],
) -> Result<Published, crate::Error> {
    if !matches!(destination, Destination::Cache(_, _)) {
        return Err(invalid(
            "weight reuse requires publication into the package cache",
        ));
    }
    publish_inner(request, key, destination, Some(weights))
}

fn publish_inner(
    request: &PublishRequest,
    key: &ring::signature::Ed25519KeyPair,
    mut destination: Destination<'_>,
    weights: Option<&[crate::package::cache::PayloadObject]>,
) -> Result<Published, crate::Error> {
    if request.deployments.is_empty() {
        return Err(invalid(
            "published packages require at least one deployment",
        ));
    }
    let progress = crate::startup::StartupProgress::stderr_rows();
    let preflight_phase = progress.phase("Validate package publication", None);
    preflight(request, &destination)?;
    preflight_phase.finish(format!("{} deployments", request.deployments.len()));
    let frontend_bytes = request.payloads.iter().map(|payload| payload.nbytes()).sum();
    let frontend_phase = progress.phase("Verify frontend payloads", Some(frontend_bytes));
    let mut package = request.package.clone();
    package.format_version = schema::FORMAT_VERSION;
    package.package_id = request.package_id.to_vec();
    package.payloads.clear();
    package.deployments.clear();

    let mut catalog = schema::Catalog::new();
    let mut published: BTreeMap<PayloadId, PublishedPayload> = BTreeMap::new();
    let mut locations = Vec::new();
    let mut located: BTreeMap<PayloadId, ()> = BTreeMap::new();
    // Hash every payload source concurrently before the ordered publication loops below; they
    // then only compare identities. Sources reused from cached weights are not hashed at all.
    let mut verified = if weights.is_some() {
        VerifiedSources::default()
    } else {
        let sources: Vec<&PublishPayload> = request
            .deployments
            .iter()
            .flat_map(|deployment| {
                deployment.ranks.iter().flat_map(|rank| rank.payloads.iter()).chain(deployment.payloads.iter())
            })
            .collect();
        let total = sources.iter().map(|payload| payload.nbytes()).sum();
        let phase = progress.phase("Verify payload sources", Some(total));
        let verified = VerifiedSources::hash(&sources, &phase)?;
        phase.finish(format!("{total} source bytes verified by {} workers", verified.workers));
        verified
    };
    let mut primary: Vec<(String, Vec<u8>)> = Vec::new();
    let directory = match &destination {
        Destination::Directory(path) => path.to_path_buf(),
        Destination::Cache(_, staging) => staging.directory().to_path_buf(),
    };

    for payload in &request.payloads {
        let mut bytes = Vec::with_capacity(schema::size(payload.nbytes())?);
        payload.source.copy_to(&mut bytes)?;
        if bytes.len() as u64 != payload.nbytes() {
            return Err(invalid("payload length disagrees with its source slice"));
        }
        let id = PayloadId::of(&bytes);
        if id != payload.id {
            return Err(invalid(format!(
                "payload {} bytes do not match the assigned identity",
                payload.relative
            )));
        }
        let tensor = payload
            .tensor
            .clone()
            .ok_or_else(|| invalid("package payload requires a primary tensor name"))?;
        primary.push((tensor.clone(), bytes));
        let descriptor = wire::PayloadDescriptor {
            id: id.as_bytes().to_vec(),
            nbytes: payload.nbytes(),
        };
        package.payloads.push(descriptor.clone());
        catalog.insert(&descriptor)?;
        if located.insert(id, ()).is_none() {
            locations.push(wire::PayloadLocation {
                payload_id: id.as_bytes().to_vec(),
                file: String::new(),
                tensor: tensor.clone(),
            });
        }
        published.entry(id).or_insert(PublishedPayload {
            id,
            nbytes: payload.nbytes(),
            role: payload.role as i32,
            relative: String::new(),
            tensor: Some(tensor.clone()),
        });
        frontend_phase.advance_bytes(payload.nbytes());
    }

    frontend_phase.finish(format!("{frontend_bytes} bytes"));
    let mut resource_id = 1u64;
    let mut resource_ids = std::collections::BTreeSet::new();
    let mut deployments = Vec::new();
    let mut ranks = Vec::new();
    for (index, deployment) in request.deployments.iter().enumerate() {
        let mut body = deployment.body.clone();
        body.ranks.clear();
        body.payloads.clear();
        for rank in &deployment.ranks {
            let mut rank_body = rank.body.clone();
            rank_body.resources.clear();
            rank_body.payloads.clear();
            if rank.resources.windows(2).any(|pair| {
                pair[0].kind == wire::ExecutableKind::Graph
                    && pair[1].kind == wire::ExecutableKind::Cubin
            }) {
                return Err(invalid("rank resources must list cubins before graphs"));
            }
            let executable_bytes = rank.resources.iter().map(|resource| resource.nbytes).sum();
            let executable_phase = progress.phase(
                &format!("Compress and publish rank {} executables", rank.body.rank),
                Some(executable_bytes),
            );
            let mut groups: Vec<(wire::ExecutableKind, Vec<usize>)> = Vec::new();
            for (position, resource) in rank.resources.iter().enumerate() {
                match groups.last_mut() {
                    Some((kind, indices)) if *kind == resource.kind => indices.push(position),
                    _ => groups.push((resource.kind, vec![position])),
                }
            }
            for (kind, indices) in groups {
                let name = match kind {
                    wire::ExecutableKind::Cubin => "kernels",
                    _ => "graphs",
                };
                let relative = format!(
                    "deployments/{}/rank-{}/{name}.safetensors",
                    body.deployment_id, rank_body.rank
                );
                let scratch = staging_name(&destination, &format!(".sealing-{}", rank_body.rank));
                let mut output = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&scratch)
                    .map_err(|e| io("create executable payload", e))?;
                let mut stored_end = 0u64;
                let mut plaintext_bytes = 0u64;
                let mut spans = Vec::with_capacity(indices.len());
                for position in indices {
                    let resource = &rank.resources[position];
                    executable_phase.detail(format!("compressing {}", resource.path.display()));
                    let compressed = compress_stream(
                        std::io::BufReader::new(
                            File::open(&resource.path)
                                .map_err(|e| io("open executable plaintext", e))?,
                        ),
                        resource.nbytes,
                        &mut output,
                    )?;
                    stored_end = stored_end
                        .checked_add(compressed.stored_nbytes)
                        .ok_or_else(|| invalid("executable payload length overflow"))?;
                    spans.push((
                        resource_id,
                        resource.nbytes,
                        stored_end,
                        compressed.chunk_sha256,
                    ));
                    resource_id += 1;
                    plaintext_bytes += resource.nbytes;
                }
                output
                    .sync_all()
                    .map_err(|e| io("sync executable payload", e))?;
                drop(output);
                let payload_len = std::fs::metadata(&scratch)
                    .map_err(|e| io("stat executable payload", e))?
                    .len();
                if payload_len != stored_end {
                    return Err(invalid(
                        "executable payload length disagrees with its stored ranges",
                    ));
                }
                if payload_len == 0 {
                    return Err(invalid("executable payload is empty"));
                }
                // Hash every fresh occurrence before deduplication. Identical rank groups
                // share one companion; later occurrences still verify their bytes and length.
                executable_phase.detail(format!("verifying and installing {payload_len} compressed bytes"));
                let id = hash_file(&scratch, payload_len)?;
                if !located.contains_key(&id) {
                    install_path_bytes(
                        &scratch,
                        id,
                        payload_len,
                        &relative,
                        &mut destination,
                    )?;
                    locations.push(wire::PayloadLocation {
                        payload_id: id.as_bytes().to_vec(),
                        file: relative.clone(),
                        tensor: container::COMPANION_TENSOR.to_string(),
                    });
                    located.insert(id, ());
                } else if published.get(&id).is_none_or(|payload| payload.nbytes != payload_len) {
                    return Err(invalid(format!(
                        "executable payload {} length differs from its first occurrence",
                        id
                    )));
                }
                std::fs::remove_file(&scratch).map_err(|e| io("remove sealed scratch", e))?;
                executable_phase.advance_bytes(plaintext_bytes);
                let descriptor = wire::PayloadDescriptor {
                    id: id.as_bytes().to_vec(),
                    nbytes: payload_len,
                };
                rank_body.payloads.push(descriptor.clone());
                body.payloads.push(descriptor.clone());
                catalog.insert(&descriptor)?;
                published.entry(id).or_insert(PublishedPayload {
                    id,
                    nbytes: payload_len,
                    role: executable_role(kind),
                    relative: relative.clone(),
                    tensor: None,
                });
                let mut previous = 0u64;
                for (id_value, decoded_nbytes, stop, chunk_sha256) in spans {
                    rank_body.resources.push(wire::ExecutableResource {
                        resource_id: id_value,
                        kind: kind as i32,
                        stored: Some(wire::PayloadSlice {
                            payload_id: id.as_bytes().to_vec(),
                            offset: previous,
                            nbytes: stop - previous,
                        }),
                        decoded_nbytes,
                        chunk_nbytes: schema::MAX_CHUNK as u32,
                        chunk_sha256,
                    });
                    previous = stop;
                }
                if previous != payload_len {
                    return Err(invalid(
                        "executable payload length disagrees with its slices",
                    ));
                }
            }
            executable_phase.finish(format!("{executable_bytes} plaintext bytes"));
            let payload_bytes = rank.payloads.iter().map(|payload| payload.nbytes()).sum();
            let data_phase = progress.phase(
                &format!("{} rank {} data", if weights.is_some() { "Publish" } else { "Verify and publish" }, rank.body.rank),
                Some(payload_bytes),
            );
            let mut reused_bytes = 0;
            for payload in &rank.payloads {
                data_phase.detail(format!("{} ({} bytes)", payload.relative, payload.nbytes()));
                let id = payload.id;
                if reuse_weight(payload, weights, &mut destination, &mut locations, located.contains_key(&id))? {
                    reused_bytes += payload.nbytes();
                    located.insert(id, ());
                } else if located.contains_key(&id) {
                    verified.check(payload)?;
                } else {
                    install_payload_verified(
                        payload,
                        &mut destination,
                        &mut locations,
                        request.no_weights,
                        &verified,
                    )?;
                    located.insert(id, ());
                }
                data_phase.advance_bytes(payload.nbytes());
                let descriptor = wire::PayloadDescriptor {
                    id: id.as_bytes().to_vec(),
                    nbytes: payload.nbytes(),
                };
                rank_body.payloads.push(descriptor.clone());
                body.payloads.push(descriptor.clone());
                catalog.insert(&descriptor)?;
                published.entry(id).or_insert(PublishedPayload {
                    id,
                    nbytes: payload.nbytes(),
                    role: payload.role as i32,
                    relative: payload.relative.clone(),
                    tensor: None,
                });
            }
            data_phase.finish(data_result(payload_bytes, reused_bytes));
            let ranks_count =
                u32::try_from(deployment.ranks.len()).map_err(|_| invalid("too many ranks"))?;
            schema::validate_rank(&rank_body, ranks_count, &mut catalog, &mut resource_ids)?;
            coalesce_initializers(&mut rank_body.initializers)?;
            let encoded = wire::RankBody::encode_to_vec(&rank_body);
            body.ranks.push(wire::RankCommitment {
                rank: rank_body.rank,
                body_digest: crate::package::protection::sha256(&encoded).to_vec(),
            });
            ranks.push((index as u32, rank_body.rank, encoded));
        }
        let payload_bytes = deployment.payloads.iter().map(|payload| payload.nbytes()).sum();
        let data_phase = progress.phase(if weights.is_some() { "Publish shared data" } else { "Verify and publish shared data" }, Some(payload_bytes));
        // Verify cached shared payloads and copy the ones the cache still lacks concurrently. The
        // ordered loop below remains the only place that links, locates, and catalogs them, and
        // finds these objects already verified.
        let mut installed = std::collections::HashSet::new();
        if let Destination::Cache(root, _) = &destination {
            let mut pending = Vec::new();
            let mut seen = std::collections::BTreeSet::new();
            for payload in &deployment.payloads {
                let weight = matches!(
                    payload.role,
                    wire::PayloadRole::Weights | wire::PayloadRole::TunedWeights
                );
                let reused = weights.is_some_and(|objects| objects.iter().any(|object| object.id() == payload.id));
                if located.contains_key(&payload.id)
                    || (weight && (request.no_weights || reused))
                    || !seen.insert(payload.id)
                {
                    continue;
                }
                pending.push(payload);
            }
            if !pending.is_empty() {
                data_phase.detail(format!("installing {} shared payloads", pending.len()));
                verified.install(&pending, root, &data_phase)?;
                installed.extend(pending.iter().map(|payload| source_address(payload)));
            }
        }
        let mut reused_bytes = 0;
        for payload in &deployment.payloads {
            data_phase.detail(format!("{} ({} bytes)", payload.relative, payload.nbytes()));
            let id = payload.id;
            if reuse_weight(payload, weights, &mut destination, &mut locations, located.contains_key(&id))? {
                reused_bytes += payload.nbytes();
                located.insert(id, ());
            } else if located.contains_key(&id) {
                verified.check(payload)?;
            } else {
                install_payload_verified(
                    payload,
                    &mut destination,
                    &mut locations,
                    request.no_weights,
                    &verified,
                )?;
                located.insert(id, ());
            }
            if !installed.contains(&source_address(payload)) {
                data_phase.advance_bytes(payload.nbytes());
            }
            let descriptor = wire::PayloadDescriptor {
                id: id.as_bytes().to_vec(),
                nbytes: payload.nbytes(),
            };
            body.payloads.push(descriptor.clone());
            catalog.insert(&descriptor)?;
            published.entry(id).or_insert(PublishedPayload {
                id,
                nbytes: payload.nbytes(),
                role: payload.role as i32,
                relative: payload.relative.clone(),
                tensor: None,
            });
        }
        data_phase.finish(data_result(payload_bytes, reused_bytes));
        let encoded = wire::DeploymentBody::encode_to_vec(&body);
        schema::validate_deployment(&body, &mut catalog)?;
        package.deployments.push(wire::DeploymentCommitment {
            deployment_id: body.deployment_id.clone(),
            body_digest: crate::package::protection::sha256(&encoded).to_vec(),
        });
        deployments.push((index as u32, encoded));
    }
    let metadata_phase = progress.phase("Sign and publish package metadata", None);
    schema::validate_package(&package)?;
    let mut builder = EnvelopeBuilder::new(
        wire::PackageBody::encode_to_vec(&package),
        request.public_key,
        request.package_id,
    )?;
    for (index, body) in deployments {
        builder.deployment(index, body)?;
    }
    for (index, rank, body) in ranks {
        builder.rank(index, rank, body)?;
    }
    let manifest = builder.sign(key)?;
    let publisher = crate::runtime_package::Publisher {
        name: request
            .package
            .publisher
            .as_ref()
            .map(|p| p.namespace.clone())
            .unwrap_or_default(),
        namespace: package
            .publisher
            .as_ref()
            .map(|p| p.namespace.clone())
            .unwrap_or_default(),
        public_key: request.public_key,
        development: package.publisher.as_ref().is_some_and(|p| p.development),
    };
    let verified = crate::package::protection::verify(
        &manifest,
        std::slice::from_ref(&publisher),
        crate::package::protection::Selection::Deployment {
            deployment: 0,
            rank: None,
        },
    )?;
    verified.authorize(&request.accepted)?;
    let set = wire::LocationSet {
        version: schema::FORMAT_VERSION,
        locations,
        available_deployments: request
            .deployments
            .iter()
            .map(|deployment| deployment.body.deployment_id.clone())
            .collect(),
    };
    schema::validate_locations(&set)?;
    let bytes = primary_bytes(&manifest, &set, &primary)?;
    match &mut destination {
        Destination::Directory(root) => write_primary(root, &bytes)?,
        Destination::Cache(_, staging) => staging.write("model.safetensors", &bytes)?,
    }
    metadata_phase.finish(format!("{} metadata bytes", bytes.len()));
    let mut published: Vec<PublishedPayload> = published.into_values().collect();
    published.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(Published {
        manifest,
        payloads: published,
        directory,
    })
}

/// Publish a fully built staging directory at `destination` with one atomic
/// no-replace rename, then make the parent entry durable.
///
/// Shared by the cache entry export and by exports without a configured cache,
/// so both have identical durability and no-replacement semantics.
pub fn publish_staged_directory(staging: &Path, destination: &Path) -> Result<(), crate::Error> {
    let parent = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(|e| io("create export parent", e))?;
    crate::package::cache::synchronize(staging)?;
    crate::package::cache::rename_noreplace(staging, destination).map_err(|error| {
        invalid(format!(
            "publish staged directory at {}: {error}",
            destination.display()
        ))
    })?;
    sync_parent(destination)
}

pub fn sync_destination_parent(path: &Path) -> Result<(), crate::Error> {
    sync_parent(path)
}

fn sync_parent(path: &Path) -> Result<(), crate::Error> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    crate::package::cache::syncdir(parent)
}

fn staging_name(destination: &Destination<'_>, name: &str) -> PathBuf {
    match destination {
        Destination::Directory(root) => root.join(name),
        Destination::Cache(_, staging) => staging.directory().join(name),
    }
}

fn write_primary(root: &Path, bytes: &[u8]) -> Result<(), crate::Error> {
    let temporary = root.join(".model.safetensors.tmp");
    if root.join("model.safetensors").exists() {
        return Err(invalid("primary already exists in the export directory"));
    }
    std::fs::write(&temporary, bytes).map_err(|e| io("write staged primary", e))?;
    std::fs::hard_link(&temporary, root.join("model.safetensors"))
        .map_err(|e| io("publish primary", e))?;
    std::fs::remove_file(&temporary).map_err(|e| io("remove staged primary", e))?;
    Ok(())
}

fn hash_file(path: &Path, nbytes: u64) -> Result<PayloadId, crate::Error> {
    hash_reader(File::open(path).map_err(|e| io("open payload", e))?, nbytes)
}

/// Merge adjacent initializer runs without changing their covered bytes or order.
///
/// The rank body has already passed schema validation, so an initializer has a
/// source slice and its destination and payload ranges are in bounds. Only
/// exact source and destination continuity for one allocation, payload, and
/// role is merged; holes, overlaps, and non-adjacent runs stay distinct.
fn coalesce_initializers(
    initializers: &mut Vec<wire::Initializer>,
) -> Result<(), crate::Error> {
    let mut merged = Vec::with_capacity(initializers.len());
    for initializer in initializers.drain(..) {
        let length = match merged.last() {
            Some(previous) => initializer_merge_length(previous, &initializer)?,
            None => None,
        };
        if let Some(length) = length {
            let previous = merged
                .last_mut()
                .expect("initializer merge requires a previous run");
            previous
                .source
                .as_mut()
                .and_then(|source| source.slice.as_mut())
                .expect("validated initializer has a source slice")
                .nbytes = length;
        } else {
            merged.push(initializer);
        }
    }
    *initializers = merged;
    Ok(())
}

fn initializer_merge_length(
    previous: &wire::Initializer,
    next: &wire::Initializer,
) -> Result<Option<u64>, crate::Error> {
    let Some(previous_source) = previous
        .source
        .as_ref()
        .and_then(|source| source.slice.as_ref())
    else {
        return Ok(None);
    };
    let Some(next_source) = next.source.as_ref().and_then(|source| source.slice.as_ref()) else {
        return Ok(None);
    };
    if previous.allocation != next.allocation
        || previous.source.as_ref().map(|source| source.role) != next.source.as_ref().map(|source| source.role)
        || previous_source.payload_id != next_source.payload_id
    {
        return Ok(None);
    }
    let source_end = previous_source
        .offset
        .checked_add(previous_source.nbytes)
        .ok_or_else(|| invalid("initializer source range overflow"))?;
    let destination_end = previous
        .destination_offset
        .checked_add(previous_source.nbytes)
        .ok_or_else(|| invalid("initializer destination range overflow"))?;
    if source_end != next_source.offset || destination_end != next.destination_offset {
        return Ok(None);
    }
    previous_source
        .nbytes
        .checked_add(next_source.nbytes)
        .map(Some)
        .ok_or_else(|| invalid("initializer length overflow"))
}

/// Validate every destination path and name before any file is created.
fn preflight(request: &PublishRequest, destination: &Destination<'_>) -> Result<(), crate::Error> {
    let mut paths = std::collections::BTreeSet::new();
    let mut tensors = std::collections::BTreeSet::new();
    // Package-scoped payloads are embedded in the primary as named tensors:
    // they have no companion path and must not touch the filesystem here.
    for payload in &request.payloads {
        if !payload.relative.is_empty() {
            return Err(invalid(
                "package payloads are embedded and have no companion path",
            ));
        }
        let tensor = payload
            .tensor
            .as_ref()
            .ok_or_else(|| invalid("package payload requires a primary tensor name"))?;
        container::validate_tensor_name(tensor)?;
        if !tensors.insert(tensor.clone()) {
            return Err(invalid("duplicate package payload tensor name"));
        }
        if payload.nbytes() == 0 {
            return Err(invalid("package payload has no bytes"));
        }
    }
    for deployment in &request.deployments {
        safe_id(&deployment.body.deployment_id, "deployment id")?;
        for payload in deployment
            .ranks
            .iter()
            .flat_map(|rank| rank.payloads.iter())
            .chain(deployment.payloads.iter())
        {
            container::validate_relative_path(&payload.relative).map_err(|error| {
                invalid(format!(
                    "payload destination {:?} is invalid: {error}",
                    payload.relative
                ))
            })?;
            schema::location_path(&payload.relative).map_err(|error| {
                invalid(format!(
                    "payload destination {:?} is invalid: {error}",
                    payload.relative
                ))
            })?;
            if !paths.insert(payload.relative.clone()) {
                return Err(invalid(format!(
                    "duplicate payload destination {:?}",
                    payload.relative
                )));
            }
            if destination.exists(&payload.relative) {
                return Err(invalid(format!(
                    "payload destination {:?} already exists",
                    payload.relative
                )));
            }
        }
    }
    if matches!(destination, Destination::Directory(root) if root.join("model.safetensors").exists())
    {
        return Err(invalid("primary already exists in the export directory"));
    }
    Ok(())
}

/// Identifiers embedded in generated paths are restricted to safe components.
fn safe_id(value: &str, what: &str) -> Result<(), crate::Error> {
    let safe = !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        && !value.starts_with('.');
    if safe {
        Ok(())
    } else {
        Err(invalid(format!("{what} is not a safe path component")))
    }
}

impl Destination<'_> {
    fn exists(&self, relative: &str) -> bool {
        match self {
            Destination::Directory(root) => root.join(relative).exists(),
            Destination::Cache(_, staging) => staging.directory().join(relative).exists(),
        }
    }
}

fn primary_bytes(
    manifest: &[u8],
    set: &wire::LocationSet,
    payloads: &[(String, Vec<u8>)],
) -> Result<Vec<u8>, crate::Error> {
    let mut bytes = Vec::new();
    let mut writer = ContainerWriter::new(&mut bytes, container::PAYLOAD_FORMAT)?;
    writer.add(
        crate::package::reader::MANIFEST,
        manifest.len() as u64,
        Source::Bytes(manifest.to_vec()),
    )?;
    let locations = wire::LocationSet::encode_to_vec(set);
    writer.add(
        crate::package::reader::LOCATIONS,
        locations.len() as u64,
        Source::Bytes(locations),
    )?;
    for (name, bytes) in payloads {
        writer.add(name, bytes.len() as u64, Source::Bytes(bytes.clone()))?;
    }
    writer.finish()?;
    Ok(bytes)
}

fn executable_role(kind: wire::ExecutableKind) -> i32 {
    match kind {
        wire::ExecutableKind::Cubin => wire::PayloadRole::Kernels as i32,
        _ => wire::PayloadRole::Graphs as i32,
    }
}

impl PublishPayload {
    fn nbytes(&self) -> u64 {
        self.source.len()
    }

    fn reader(&self) -> SliceReader<'_> {
        SliceReader {
            slice: &self.source,
            position: 0,
        }
    }

    fn digest(&self) -> Result<PayloadId, crate::Error> {
        hash_reader(self.reader(), self.nbytes())
    }
}

fn verify_payload_source(payload: &PublishPayload) -> Result<PayloadId, crate::Error> {
    let actual = payload.digest()?;
    if actual != payload.id {
        return Err(invalid(format!(
            "payload {} bytes do not match the assigned identity",
            payload.relative
        )));
    }
    Ok(actual)
}

/// Payload sources whose bytes already hashed to their assigned identity, keyed by address, and
/// cache objects this export installed or hashed ahead of the ordered publication loop.
#[derive(Default)]
struct VerifiedSources {
    sources: std::collections::HashSet<usize>,
    installed: BTreeMap<PayloadId, crate::package::cache::PayloadObject>,
    workers: usize,
}

const MAX_SOURCE_VERIFY_WORKERS: usize = 8;

fn source_address(payload: &PublishPayload) -> usize {
    payload as *const PublishPayload as usize
}

/// Run `work` over `payloads` on parallel workers, largest first: one synchronous reader leaves
/// NVMe queues idle, and the last worker should not start a long payload alone. The first error
/// stops scheduling and is returned once running work drains. Returns the worker count and each
/// result with its payload index.
fn largest_first<T: Send>(
    payloads: &[&PublishPayload],
    work: impl Fn(&PublishPayload) -> Result<T, crate::Error> + Sync,
) -> Result<(usize, Vec<(usize, T)>), crate::Error> {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let workers = std::thread::available_parallelism()
        .map_or(1, |workers| workers.get())
        .min(MAX_SOURCE_VERIFY_WORKERS)
        .min(payloads.len());
    let mut order: Vec<usize> = (0..payloads.len()).collect();
    order.sort_by_key(|&index| std::cmp::Reverse(payloads[index].nbytes()));
    let cursor = AtomicUsize::new(0);
    let failure: Mutex<Option<crate::Error>> = Mutex::new(None);
    let results: Mutex<Vec<(usize, T)>> = Mutex::new(Vec::with_capacity(payloads.len()));
    let mut spawn_error = None;
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let worker = std::thread::Builder::new().spawn_scoped(scope, || loop {
                if failure.lock().expect("payload worker mutex").is_some() {
                    break;
                }
                let Some(&index) = order.get(cursor.fetch_add(1, Ordering::Relaxed)) else {
                    break;
                };
                match work(payloads[index]) {
                    Ok(value) => results.lock().expect("payload worker mutex").push((index, value)),
                    Err(error) => {
                        failure.lock().expect("payload worker mutex").get_or_insert(error);
                        break;
                    }
                }
            });
            if let Err(error) = worker {
                spawn_error = Some(error);
                break;
            }
        }
    });
    if let Some(error) = spawn_error {
        return Err(invalid(format!("spawn payload worker: {error}")));
    }
    if let Some(error) = failure.into_inner().expect("payload worker mutex") {
        return Err(error);
    }
    Ok((workers, results.into_inner().expect("payload worker mutex")))
}

impl VerifiedSources {
    /// Hash `payloads` against their assigned identities on parallel readers.
    fn hash(payloads: &[&PublishPayload], phase: &crate::startup::Phase) -> Result<Self, crate::Error> {
        let (workers, _) = largest_first(payloads, |payload| {
            verify_payload_source(payload)?;
            phase.advance_bytes(payload.nbytes());
            Ok(())
        })?;
        Ok(Self {
            sources: payloads.iter().map(|payload| source_address(payload)).collect(),
            installed: BTreeMap::new(),
            workers,
        })
    }

    fn check(&self, payload: &PublishPayload) -> Result<PayloadId, crate::Error> {
        if self.sources.contains(&source_address(payload)) {
            return Ok(payload.id);
        }
        verify_payload_source(payload)
    }

    /// Resolve cache objects for `payloads`, which have distinct identities, on the parallel
    /// readers. Each runs the ordered loop's cache path: check the source, then hash a cached
    /// object against its identity unless it is the very source span, or copy the source into the
    /// cache and hash the written object. The ordered loop then links these objects without
    /// hashing them again.
    fn install(
        &mut self,
        payloads: &[&PublishPayload],
        root: &crate::package::cache::Root,
        phase: &crate::startup::Phase,
    ) -> Result<(), crate::Error> {
        let this = &*self;
        let (_, objects) = largest_first(payloads, |payload| {
            this.check(payload)?;
            let object = match root.payload(payload.id)? {
                Some(object) => {
                    if object.length() != payload.nbytes() {
                        return Err(invalid(format!(
                            "cached payload {} length differs from its declaration",
                            payload.id
                        )));
                    }
                    if !same_bytes(object.slice(), &payload.source)
                        && object.slice().digest()? != payload.id
                    {
                        return Err(invalid(format!(
                            "cached payload {} bytes do not match their identity",
                            payload.id
                        )));
                    }
                    object
                }
                None => {
                    let object =
                        root.insert_payload(payload.id, payload.nbytes(), payload.reader())?;
                    if object.id() != payload.id {
                        return Err(invalid(
                            "cache payload identity differs from the signed identity",
                        ));
                    }
                    object
                }
            };
            phase.advance_bytes(payload.nbytes());
            Ok(object)
        })?;
        for (index, object) in objects {
            // Each source was checked before its object was resolved.
            self.sources.insert(source_address(payloads[index]));
            self.installed.insert(object.id(), object);
        }
        Ok(())
    }

    /// Whether `object` is the file this export installed or hashed under its identity.
    fn installed(&self, object: &crate::package::cache::PayloadObject) -> bool {
        self.installed
            .get(&object.id())
            .is_some_and(|held| same_bytes(held.slice(), object.slice()))
    }
}

/// Whether two slices name the same bytes of the same file, so one hash covers both.
#[cfg(unix)]
fn same_bytes(left: &Slice, right: &Slice) -> bool {
    use std::os::unix::fs::MetadataExt;
    if left.offset() != right.offset() || left.len() != right.len() {
        return false;
    }
    match (left.file().metadata(), right.file().metadata()) {
        (Ok(left), Ok(right)) => left.dev() == right.dev() && left.ino() == right.ino(),
        _ => false,
    }
}

#[cfg(not(unix))]
fn same_bytes(_left: &Slice, _right: &Slice) -> bool {
    false
}

/// Positional reader over a pinned safetensors payload slice.
struct SliceReader<'a> {
    slice: &'a Slice,
    position: u64,
}

impl Read for SliceReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if self.position >= self.slice.len() {
            return Ok(0);
        }
        let length = (self.slice.len() - self.position).min(buffer.len() as u64) as usize;
        self.slice
            .read_at(self.position, &mut buffer[..length])
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        self.position += length as u64;
        Ok(length)
    }
}

fn data_result(total: u64, reused: u64) -> String {
    if reused == 0 {
        format!("{total} source bytes verified")
    } else {
        format!(
            "{reused} cached weight bytes reused, {} fresh bytes verified",
            total - reused
        )
    }
}

fn reuse_weight(
    payload: &PublishPayload,
    weights: Option<&[crate::package::cache::PayloadObject]>,
    destination: &mut Destination<'_>,
    locations: &mut Vec<wire::PayloadLocation>,
    located: bool,
) -> Result<bool, crate::Error> {
    let Some(weights) = weights.filter(|_| {
        matches!(
            payload.role,
            wire::PayloadRole::Weights | wire::PayloadRole::TunedWeights
        )
    }) else {
        return Ok(false);
    };
    if !weights.iter().any(|object| object.id() == payload.id) {
        return Ok(false);
    }
    let object = weights
        .iter()
        .find(|object| {
            object.id() == payload.id
                && std::sync::Arc::ptr_eq(object.file(), payload.source.file())
                && object.slice().offset() == payload.source.offset()
                && object.length() == payload.nbytes()
        })
        .ok_or_else(|| {
            invalid(format!(
                "reused weight {} does not reference its leased cache object",
                payload.relative
            ))
        })?;
    if !located {
        let Destination::Cache(_, staging) = destination else {
            return Err(invalid("weight reuse requires the package cache"));
        };
        staging.link(object, &payload.relative)?;
        locations.push(wire::PayloadLocation {
            payload_id: payload.id.as_bytes().to_vec(),
            file: payload.relative.clone(),
            tensor: container::COMPANION_TENSOR.to_string(),
        });
    }
    Ok(true)
}

#[cfg(test)]
fn install_payload(
    payload: &PublishPayload,
    destination: &mut Destination<'_>,
    locations: &mut Vec<wire::PayloadLocation>,
    no_weights: bool,
) -> Result<PayloadId, crate::Error> {
    install_payload_verified(payload, destination, locations, no_weights, &VerifiedSources::default())
}

fn install_payload_verified(
    payload: &PublishPayload,
    destination: &mut Destination<'_>,
    locations: &mut Vec<wire::PayloadLocation>,
    no_weights: bool,
    verified: &VerifiedSources,
) -> Result<PayloadId, crate::Error> {
    // Only stored weight bytes must already exist. Initial state, frontend
    // assets, and compressed executable resources (kernels/graphs) are installed
    // from their fresh source.
    if no_weights && matches!(
        payload.role,
        wire::PayloadRole::Weights | wire::PayloadRole::TunedWeights
    ) {
        verified.check(payload)?;
        match destination {
            Destination::Cache(root, staging) => {
                let object = root.payload(payload.id)?.ok_or_else(|| {
                    invalid(format!(
                        "no-weights export is missing cached payload {} (group {})",
                        payload.id, payload.relative
                    ))
                })?;
                if object.length() != payload.nbytes() {
                    return Err(invalid(format!(
                        "cached payload {} length differs from its declaration",
                        payload.id
                    )));
                }
                // The object name is not evidence: hash the pinned cached bytes, unless they are
                // the very source bytes already hashed against this identity.
                if !same_bytes(object.slice(), &payload.source) && object.slice().digest()? != payload.id {
                    return Err(invalid(format!(
                        "cached payload {} bytes do not match their identity",
                        payload.id
                    )));
                }
                // Link the leased cached object into the staged entry and
                // record its location so the published closure is complete.
                staging.link(&object, &payload.relative)?;
                locations.push(wire::PayloadLocation {
                    payload_id: payload.id.as_bytes().to_vec(),
                    file: payload.relative.clone(),
                    tensor: container::COMPANION_TENSOR.to_string(),
                });
            }
            Destination::Directory(_) => {
                return Err(invalid(
                    "no-weights export requires the package cache, not a directory",
                ));
            }
        }
    } else {
        install_payload_bytes_verified(payload, destination, verified)?;
        locations.push(wire::PayloadLocation {
            payload_id: payload.id.as_bytes().to_vec(),
            file: payload.relative.clone(),
            tensor: container::COMPANION_TENSOR.to_string(),
        });
    }
    Ok(payload.id)
}

/// Install one staged compressed file by path (executable resources).
fn install_path_bytes(
    source: &Path,
    id: PayloadId,
    nbytes: u64,
    relative: &str,
    destination: &mut Destination<'_>,
) -> Result<(), crate::Error> {
    let payload = PublishPayload {
        role: wire::PayloadRole::Kernels,
        relative: relative.to_string(),
        source: pin_file(source, nbytes)?,
        id,
        tensor: None,
    };
    install_payload_bytes(&payload, destination)
}

fn install_payload_bytes(
    payload: &PublishPayload,
    destination: &mut Destination<'_>,
) -> Result<(), crate::Error> {
    install_payload_bytes_verified(payload, destination, &VerifiedSources::default())
}

fn install_payload_bytes_verified(
    payload: &PublishPayload,
    destination: &mut Destination<'_>,
    verified: &VerifiedSources,
) -> Result<(), crate::Error> {
    verified.check(payload)?;
    match destination {
        Destination::Directory(root) => {
            write_verified_companion(root, &payload.relative, payload)?;
        }
        Destination::Cache(root, staging) => {
            // Reuse an already installed object instead of capturing a duplicate
            // wrapper. The caller's fresh bytes are still hashed, so a cache hit
            // never substitutes for verification of the payload being published.
            let object = match root.payload(payload.id)? {
                Some(object) => {
                    if object.length() != payload.nbytes() {
                        return Err(invalid(format!(
                            "cached payload {} length differs from its declaration",
                            payload.id
                        )));
                    }
                    // The object name is not evidence: a warm hit must also
                    // validate the cached bytes before linking them, unless they are
                    // the very source bytes already hashed against this identity or
                    // the object this export installed and hashed.
                    if !same_bytes(object.slice(), &payload.source)
                        && !verified.installed(&object)
                        && object.slice().digest()? != payload.id
                    {
                        return Err(invalid(format!(
                            "cached payload {} bytes do not match their identity",
                            payload.id
                        )));
                    }
                    object
                }
                None => {
                    let object =
                        root.insert_payload(payload.id, payload.nbytes(), payload.reader())?;
                    if object.id() != payload.id {
                        return Err(invalid(
                            "cache payload identity differs from the signed identity",
                        ));
                    }
                    object
                }
            };
            staging.link(&object, &payload.relative)?;
        }
    }
    Ok(())
}

/// Stream one payload into its companion wrapper, then verify the bytes that
/// were actually written against the signed identity. No raw temporary copy is
/// created: the source is read once and the written companion is hashed.
fn write_verified_companion(
    directory: &Path,
    relative: &str,
    payload: &PublishPayload,
) -> Result<(), crate::Error> {
    let nbytes = payload.nbytes();
    let id = payload.id;
    let path = directory.join(relative);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| io("create payload directory", e))?;
    }
    let mut input = payload.reader();
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| io("create companion", e))?;
    let mut consumed = 0u64;
    let result = (|| -> Result<(), crate::Error> {
        {
            let reader = &mut input;
            let consumed = &mut consumed;
            let mut writer = ContainerWriter::new(&mut output, container::PAYLOAD_FORMAT)?;
            writer.add(
                container::COMPANION_TENSOR,
                nbytes,
                Source::Reader(Box::new(move |offset: u64, buffer: &mut [u8]| {
                    if offset != *consumed {
                        return Err(invalid("payload read is not sequential"));
                    }
                    reader
                        .read_exact(buffer)
                        .map_err(|e| io("read payload", e))?;
                    *consumed += buffer.len() as u64;
                    Ok(())
                })),
            )?;
            writer.finish()?;
        }
        output.sync_all().map_err(|e| io("sync companion", e))?;
        if consumed != nbytes {
            return Err(invalid("companion length disagrees with the signed length"));
        }
        // Hash what was published, not what we believe we read.
        let file = File::open(&path).map_err(|e| io("open companion", e))?;
        let slice = Slice::new(std::sync::Arc::new(file), container_base(&path)?, nbytes)?;
        if slice.digest()? != id {
            return Err(invalid("payload bytes do not match their identity"));
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&path);
    }
    result
}

/// Byte offset of the companion payload inside its wrapper.
fn container_base(path: &Path) -> Result<u64, crate::Error> {
    let container = container::Container::open(path)?;
    Ok(container.tensor(container::COMPANION_TENSOR)?.offset())
}

fn hash_reader(mut reader: impl Read, nbytes: u64) -> Result<PayloadId, crate::Error> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut remaining = nbytes;
    while remaining > 0 {
        let chunk = remaining.min(buffer.len() as u64) as usize;
        reader
            .read_exact(&mut buffer[..chunk])
            .map_err(|e| io("read payload", e))?;
        hasher.update(&buffer[..chunk]);
        remaining -= chunk as u64;
    }
    Ok(PayloadId::from_bytes(hasher.finish()))
}

struct HashingWriter<'a, W: Write> {
    inner: &'a mut W,
    hasher: &'a mut Sha256,
}

impl<W: Write> Write for HashingWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        // Hash exactly what the sink accepted, so a short write never hashes
        // unwritten bytes and a retry never hashes the same bytes twice.
        let written = self.inner.write(bytes)?;
        self.hasher.update(&bytes[..written]);
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Minimal streaming SHA-256 so payload identity is computed over payload bytes.
struct Sha256 {
    context: ring::digest::Context,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            context: ring::digest::Context::new(&ring::digest::SHA256),
        }
    }

    fn update(&mut self, bytes: &[u8]) {
        self.context.update(bytes);
    }

    fn finish(self) -> [u8; 32] {
        let mut hash = [0u8; 32];
        hash.copy_from_slice(self.context.finish().as_ref());
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "flare-export-{}-{unique}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_source(directory: &Path, name: &str, bytes: &[u8]) -> CapturedSource {
        let path = directory.join(name);
        std::fs::write(&path, bytes).unwrap();
        CapturedSource {
            offset: 0,
            path,
            file_offset: 0,
            nbytes: bytes.len() as u64,
        }
    }

    fn group(relative: &str, spans: Vec<CapturedSpan>) -> CapturedGroup {
        CapturedGroup {
            name: relative.to_string(),
            role: 0,
            relative: relative.to_string(),
            spans,
        }
    }

    #[test]
    fn assembly_hashes_exact_payload_bytes_and_maps_initializers() {
        let source = scratch("a-source");
        let output = scratch("a-output");
        let bytes = b"0123456789abcdef".to_vec();
        let source = write_source(&source, "storage.bin", &bytes);
        let ranks = vec![CapturedRank {
            rank: 0,
            groups: vec![group(
                "weights/layer-000.safetensors",
                vec![
                    CapturedSpan {
                        allocation: "device-0".into(),
                        destination_offset: 0x1000,
                        source_offset: 0,
                        nbytes: 4,
                    },
                    CapturedSpan {
                        allocation: "device-0".into(),
                        destination_offset: 0x1100,
                        source_offset: 8,
                        nbytes: 8,
                    },
                ],
            )],
            sources: vec![source],
        }];
        let assembled = assemble(&output, &ranks).unwrap();
        let payload = &assembled[0].payloads[0];
        let expected = PayloadId::of(b"012389abcdef");
        assert_eq!(payload.id, expected);
        assert_eq!(payload.length, 12);
        let container = container::Container::open(&payload.path).unwrap();
        let stored = container.tensor(container::COMPANION_TENSOR).unwrap();
        assert_eq!(stored.len(), 12);
        let mut raw = Vec::new();
        stored.copy_to(&mut raw).unwrap();
        assert_eq!(raw, b"012389abcdef".to_vec());
        assert_eq!(
            assembled[0].initializers,
            vec![
                AssembledInitializer {
                    allocation: "device-0".into(),
                    destination_offset: 0x1000,
                    payload: expected,
                    payload_offset: 0,
                    nbytes: 4,
                },
                AssembledInitializer {
                    allocation: "device-0".into(),
                    destination_offset: 0x1100,
                    payload: expected,
                    payload_offset: 4,
                    nbytes: 8,
                },
            ]
        );
    }

    #[test]
    fn uncovered_source_ranges_are_rejected() {
        let source = scratch("a-source");
        let output = scratch("a-output");
        let part = write_source(&source, "storage.bin", b"abcd");
        let ranks = vec![CapturedRank {
            rank: 0,
            groups: vec![group(
                "weights/layer-000.safetensors",
                vec![CapturedSpan {
                    allocation: "device-0".into(),
                    destination_offset: 0,
                    source_offset: 1,
                    nbytes: 8,
                }],
            )],
            sources: vec![part],
        }];
        assert!(assemble(&output, &ranks).is_err());
    }

    struct OneByteWriter(Vec<u8>);

    impl Write for OneByteWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.is_empty() {
                return Ok(0);
            }
            self.0.push(bytes[0]);
            Ok(1)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn hashing_writer_hashes_each_written_byte_exactly_once() {
        let content = b"payload bytes that are written one at a time";
        let mut hasher = Sha256::new();
        let mut sink = OneByteWriter(Vec::new());
        {
            let mut writer = HashingWriter {
                inner: &mut sink,
                hasher: &mut hasher,
            };
            // A single call must not hash the whole buffer if the sink accepts
            // one byte; the retry loop then hashes each accepted byte once.
            writer.write_all(content).unwrap();
        }
        assert_eq!(sink.0, content);
        let expected = PayloadId::of(content);
        assert_eq!(PayloadId::from_bytes(hasher.finish()), expected);
    }

    fn initializer(
        allocation: &str,
        destination_offset: u64,
        payload: &[u8],
        source_offset: u64,
        nbytes: u64,
        role: wire::PayloadRole,
    ) -> wire::Initializer {
        wire::Initializer {
            allocation: allocation.into(),
            destination_offset,
            source: Some(wire::PayloadUse {
                slice: Some(wire::PayloadSlice {
                    payload_id: payload.to_vec(),
                    offset: source_offset,
                    nbytes,
                }),
                role: role as i32,
            }),
        }
    }

    #[test]
    fn initializer_coalescing_merges_only_exact_adjacent_runs() {
        let bytes = b"payload";
        let mut initializers = vec![
            initializer("device-0", 0, bytes, 0, 4, wire::PayloadRole::Weights),
            initializer("device-0", 4, bytes, 4, 4, wire::PayloadRole::Weights),
            initializer("device-0", 9, bytes, 8, 4, wire::PayloadRole::Weights),
            initializer("device-0", 13, bytes, 20, 4, wire::PayloadRole::Weights),
            initializer("device-0", 17, b"other", 24, 4, wire::PayloadRole::Weights),
            initializer("device-1", 0, bytes, 0, 4, wire::PayloadRole::Weights),
            initializer("device-0", 21, bytes, 28, 4, wire::PayloadRole::TunedWeights),
        ];
        coalesce_initializers(&mut initializers).unwrap();
        assert_eq!(initializers.len(), 6);
        assert_eq!(
            (
                initializers[0].destination_offset,
                initializers[0].source.as_ref().unwrap().slice.as_ref().unwrap().offset,
                initializers[0].source.as_ref().unwrap().slice.as_ref().unwrap().nbytes,
            ),
            (0, 0, 8)
        );
        assert_eq!(initializers[1].destination_offset, 9);
        assert_eq!(initializers[2].destination_offset, 13);
        assert_eq!(initializers[3].destination_offset, 17);
        assert_eq!(
            initializers[3].source.as_ref().unwrap().slice.as_ref().unwrap().payload_id,
            b"other"
        );
        assert_eq!(initializers[4].allocation, "device-1");
        assert_eq!(initializers[5].destination_offset, 21);
    }

    #[test]
    fn initializer_coalescing_keeps_role_boundary_without_other_differences() {
        let payload = b"payload";
        let mut initializers = vec![
            initializer("device-0", 0, payload, 0, 4, wire::PayloadRole::Weights),
            initializer("device-0", 4, payload, 4, 4, wire::PayloadRole::TunedWeights),
        ];
        coalesce_initializers(&mut initializers).unwrap();
        assert_eq!(initializers.len(), 2);
        assert_eq!(initializers[0].source.as_ref().unwrap().slice.as_ref().unwrap().nbytes, 4);
        assert_eq!(initializers[1].destination_offset, 4);
        assert_eq!(initializers[1].source.as_ref().unwrap().role, wire::PayloadRole::TunedWeights as i32);
    }

    #[test]
    fn initializer_coalescing_keeps_payload_id_boundary_without_other_differences() {
        let first = b"first";
        let second = b"other";
        let mut initializers = vec![
            initializer("device-0", 0, first, 0, 4, wire::PayloadRole::Weights),
            initializer("device-0", 4, second, 4, 4, wire::PayloadRole::Weights),
        ];
        coalesce_initializers(&mut initializers).unwrap();
        assert_eq!(initializers.len(), 2);
        assert_eq!(initializers[0].source.as_ref().unwrap().slice.as_ref().unwrap().nbytes, 4);
        assert_eq!(initializers[1].destination_offset, 4);
        assert_eq!(initializers[1].source.as_ref().unwrap().slice.as_ref().unwrap().offset, 4);
    }

    #[test]
    fn staged_directory_publication_preserves_occupied_destinations() {
        let root = scratch("staged-directory");
        let stage = root.join("stage");
        std::fs::create_dir_all(stage.join("weights")).unwrap();
        std::fs::write(stage.join("model.safetensors"), b"primary").unwrap();
        std::fs::write(stage.join("weights/layer-000.safetensors"), b"weights").unwrap();
        let occupied = root.join("occupied");
        std::fs::create_dir(&occupied).unwrap();
        std::fs::write(occupied.join("user-file"), b"keep").unwrap();
        assert!(publish_staged_directory(&stage, &occupied).is_err());
        assert_eq!(std::fs::read(occupied.join("user-file")).unwrap(), b"keep");
        assert!(!occupied.join("model.safetensors").exists());
        assert!(stage.join("model.safetensors").is_file());

        let destination = root.join("export/package");
        publish_staged_directory(&stage, &destination).unwrap();
        assert!(!stage.exists());
        assert_eq!(
            std::fs::read(destination.join("model.safetensors")).unwrap(),
            b"primary"
        );
        assert_eq!(
            std::fs::read(destination.join("weights/layer-000.safetensors")).unwrap(),
            b"weights"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn staged_directory_rejects_symlinks_before_publication() {
        let root = scratch("staged-symlink");
        let stage = root.join("stage");
        std::fs::create_dir(&stage).unwrap();
        std::fs::write(root.join("user-file"), b"keep").unwrap();
        std::os::unix::fs::symlink(root.join("user-file"), stage.join("model.safetensors"))
            .unwrap();
        let destination = root.join("package");
        assert!(publish_staged_directory(&stage, &destination).is_err());
        assert!(!destination.exists());
        assert_eq!(std::fs::read(root.join("user-file")).unwrap(), b"keep");
        std::fs::remove_dir_all(root).unwrap();
    }

    fn corrupt_cached_payload(root: &crate::package::cache::Root, id: PayloadId) {
        let object = root.payload(id).unwrap().unwrap();
        let path = object.path().to_path_buf();
        let offset = object.slice().offset();
        drop(object);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        use std::io::{Seek, SeekFrom, Write};
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(offset + 1)).unwrap();
        file.write_all(&[0x5a]).unwrap();
        file.sync_all().unwrap();
    }

    #[test]
    fn warm_hits_reject_corrupted_cached_bytes_with_a_valid_header() {
        use crate::package::cache::Root;
        let root_dir = scratch("corrupt-hit");
        let cache = root_dir.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        let root = Root::open(&cache).unwrap();
        let bytes = b"prepared weight bytes".to_vec();
        let id = PayloadId::of(&bytes);
        let path = root_dir.join("weight.bin");
        std::fs::write(&path, &bytes).unwrap();
        root.capture_payload(bytes.len() as u64, bytes.as_slice()).unwrap();
        corrupt_cached_payload(&root, id);
        let payload = PublishPayload {
            role: wire::PayloadRole::Weights,
            relative: "weights/weight.safetensors".into(),
            source: pin_file(&path, bytes.len() as u64).unwrap(),
            id,
            tensor: None,
        };
        // Normal export reuses the object; the corrupted cached bytes must fail.
        let mut staging = root.stage("model", "entry-normal").unwrap();
        let error = install_payload_bytes(&payload, &mut Destination::Cache(&root, &mut staging))
            .unwrap_err();
        assert!(error.to_string().contains("cached payload"), "{error}");

        // --no-weights initial state with the same identity must also reject it.
        let state_root_dir = scratch("corrupt-state");
        let state_cache = state_root_dir.join("cache");
        std::fs::create_dir_all(&state_cache).unwrap();
        let state_root = Root::open(&state_cache).unwrap();
        let state = b"initial state bytes".to_vec();
        let state_id = PayloadId::of(&state);
        let state_path = state_root_dir.join("state.bin");
        std::fs::write(&state_path, &state).unwrap();
        state_root
            .capture_payload(state.len() as u64, state.as_slice())
            .unwrap();
        corrupt_cached_payload(&state_root, state_id);
        let state_payload = PublishPayload {
            role: wire::PayloadRole::InitialState,
            relative: "state/initial.safetensors".into(),
            source: pin_file(&state_path, state.len() as u64).unwrap(),
            id: state_id,
            tensor: None,
        };
        let mut staging = state_root.stage("model", "entry-state").unwrap();
        let error = install_payload_bytes(
            &state_payload,
            &mut Destination::Cache(&state_root, &mut staging),
        )
        .unwrap_err();
        assert!(error.to_string().contains("cached payload"), "{error}");
        let _ = std::fs::remove_dir_all(&root_dir);
        let _ = std::fs::remove_dir_all(&state_root_dir);
    }

    #[test]
    fn warm_hits_reject_corrupted_source_bytes_before_reuse() {
        use crate::package::cache::Root;
        let root_dir = scratch("corrupt-source");
        let cache = root_dir.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        let root = Root::open(&cache).unwrap();
        let bytes = b"prepared weight bytes".to_vec();
        let id = PayloadId::of(&bytes);
        let path = root_dir.join("weight.bin");
        std::fs::write(&path, &bytes).unwrap();
        root.capture_payload(bytes.len() as u64, bytes.as_slice()).unwrap();
        let source = pin_file(&path, bytes.len() as u64).unwrap();
        std::fs::write(&path, vec![0x5a; bytes.len()]).unwrap();
        let payload = PublishPayload {
            role: wire::PayloadRole::Weights,
            relative: "weights/weight.safetensors".into(),
            source,
            id,
            tensor: None,
        };
        let mut staging = root.stage("model", "source-normal").unwrap();
        let error = install_payload_bytes(&payload, &mut Destination::Cache(&root, &mut staging))
            .unwrap_err();
        assert!(error.to_string().contains("bytes do not match the assigned identity"), "{error}");

        let mut locations = Vec::new();
        let mut staging = root.stage("model", "source-no-weights").unwrap();
        let error = install_payload(
            &payload,
            &mut Destination::Cache(&root, &mut staging),
            &mut locations,
            true,
        )
        .unwrap_err();
        assert!(error.to_string().contains("bytes do not match the assigned identity"), "{error}");
        assert!(locations.is_empty(), "rejected source recorded a staged location");
        assert_eq!(
            std::fs::read_dir(staging.directory()).unwrap().count(),
            0,
            "rejected no-weights source left staged files"
        );
        assert!(
            root.versions("model").unwrap_or_default().is_empty(),
            "rejected no-weights source published an entry"
        );
        assert_eq!(
            root.payload(id).unwrap().unwrap().slice().digest().unwrap(),
            id,
            "the valid cached object was mutated or accepted as source evidence"
        );
        let _ = std::fs::remove_dir_all(root_dir);
    }

    #[test]
    fn install_path_bytes_rechecks_the_reopened_pinned_slice() {
        let root = scratch("path-recheck");
        let path = root.join("kernels.plain");
        let bytes = b"sealed executable bytes".to_vec();
        let id = PayloadId::of(&bytes);
        std::fs::write(&path, vec![0x5a; bytes.len()]).unwrap();
        let destination = root.join("package");
        let mut destination = Destination::Directory(&destination);
        let error = install_path_bytes(
            &path,
            id,
            bytes.len() as u64,
            "deployments/sm120-tp1/rank-0/kernels.safetensors",
            &mut destination,
        )
        .unwrap_err();
        assert!(error.to_string().contains("bytes do not match the assigned identity"), "{error}");
        assert!(!root.join("package").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn duplicate_deployment_payload_occurrence_is_still_source_checked() {
        use crate::package::cache::Root;
        let root_dir = scratch("duplicate-deployment");
        let cache = root_dir.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        let root = Root::open(&cache).unwrap();
        let bytes = b"same payload".to_vec();
        let corrupt = vec![0x5a; bytes.len()];
        let id = PayloadId::of(&bytes);
        let valid_path = root_dir.join("valid.bin");
        let corrupt_path = root_dir.join("corrupt.bin");
        std::fs::write(&valid_path, &bytes).unwrap();
        std::fs::write(&corrupt_path, &corrupt).unwrap();
        let graph_path = root_dir.join("graph.bin");
        std::fs::write(&graph_path, b"graph").unwrap();
        let payload = |relative: &str, path: &Path| PublishPayload {
            role: wire::PayloadRole::Weights,
            relative: relative.into(),
            source: pin_file(path, bytes.len() as u64).unwrap(),
            id,
            tensor: None,
        };
        let request = PublishRequest {
            package: wire::PackageBody {
                format_version: schema::FORMAT_VERSION,
                package_id: vec![1; 32],
                model_identity: "model".into(),
                hardware_identity: "sm120".into(),
                publisher: None,
                frontend: None,
                sampling_defaults: None,
                payloads: vec![],
                assets: vec![],
                deployments: vec![],
            },
            payloads: vec![],
            deployments: vec![PublishDeployment {
                body: wire::DeploymentBody {
                    deployment_id: "sm120-tp1".into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 1,
                    contract: None,
                    cluster: None,
                    rebuild: None,
                    engram: None,
                    payloads: vec![],
                    ranks: vec![],
                },
                ranks: vec![PublishRank {
                    body: wire::RankBody {
                        rank: 0,
                        deployment_id: "sm120-tp1".into(),
                        context: Some(crate::runtime_package::wire::Context::default()),
                        allocations: vec![wire::Allocation {
                            name: "device-0".into(),
                            nbytes: 1,
                            alignment: 1,
                            address: Some(0x1000),
                            memory: 0,
                            sizing: None,
                            relocations: vec![],
                        }],
                        graphs: vec![wire::Graph {
                            name: "decode".into(),
                            flags: 0,
                            resource: 0,
                            node_count: 1,
                            edge_count: 0,
                            functions: vec![],
                        }],
                        ..Default::default()
                    },
                    payloads: vec![],
                    resources: vec![PublishResource {
                        kind: wire::ExecutableKind::Graph,
                        path: graph_path,
                        nbytes: 5,
                    }],
                }],
                payloads: vec![
                    payload("deployments/sm120-tp1/weights/one.safetensors", &valid_path),
                    payload("deployments/sm120-tp1/weights/two.safetensors", &corrupt_path),
                ],
            }],
            package_id: [1; 32],
            public_key: [2; 32],
            accepted: "example".into(),
            no_weights: false,
        };
        let key = ring::signature::Ed25519KeyPair::from_pkcs8(
            ring::signature::Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new())
                .unwrap()
                .as_ref(),
        )
        .unwrap();
        let mut staging = root.stage("model", "duplicate-deployment").unwrap();
        let error = publish(&request, &key, Destination::Cache(&root, &mut staging)).unwrap_err();
        assert!(error.to_string().contains("bytes do not match the assigned identity"), "{error}");
        let _ = std::fs::remove_dir_all(root_dir);
    }

    #[test]
    fn no_weights_requires_only_weight_roles() {
        use crate::package::cache::Root;
        let root_dir = scratch("noweights-roles");
        let cache = root_dir.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        let root = Root::open(&cache).unwrap();
        let weight_bytes = b"prepared weight".to_vec();
        let weight_id = PayloadId::of(&weight_bytes);
        let weight_path = root_dir.join("weight.bin");
        std::fs::write(&weight_path, &weight_bytes).unwrap();
        root.capture_payload(weight_bytes.len() as u64, weight_bytes.as_slice())
            .unwrap();
        let state_bytes = b"new initial state".to_vec();
        let state_id = PayloadId::of(&state_bytes);
        let state_path = root_dir.join("state.bin");
        std::fs::write(&state_path, &state_bytes).unwrap();
        let kernel_bytes = b"compressed kernel".to_vec();
        let kernel_path = root_dir.join("kernel.bin");
        std::fs::write(&kernel_path, &kernel_bytes).unwrap();
        let graph_bytes = b"compressed graph".to_vec();
        let graph_path = root_dir.join("graph.bin");
        std::fs::write(&graph_path, &graph_bytes).unwrap();
        let payload = |role, path: &std::path::PathBuf, id, bytes: &[u8]| PublishPayload {
            role,
            relative: format!("weights/{}", path.file_name().unwrap().to_string_lossy()),
            source: pin_file(path, bytes.len() as u64).unwrap(),
            id,
            tensor: None,
        };
        let _key = ring::signature::Ed25519KeyPair::from_pkcs8(
            ring::signature::Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new())
                .unwrap()
                .as_ref(),
        )
        .unwrap();
        let package_id = [7u8; 32];
        use ring::signature::KeyPair;
        let key = ring::signature::Ed25519KeyPair::from_pkcs8(
            ring::signature::Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new())
                .unwrap()
                .as_ref(),
        )
        .unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let request = PublishRequest {
            package: wire::PackageBody {
                format_version: schema::FORMAT_VERSION,
                package_id: package_id.to_vec(),
                model_identity: "model".into(),
                hardware_identity: "gb202".into(),
                publisher: Some(wire::PublisherMetadata {
                    namespace: "example".into(),
                    official_source: "https://huggingface.co/example/model".into(),
                    development: false,
                    license_name: String::new(),
                    license_link: String::new(),
                    ranks: 0,
                }),
                frontend: Some(crate::runtime_package::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    crate::runtime_package::wire::DefaultSamplingParameters::default(),
                ),
                payloads: vec![],
                assets: vec![],
                deployments: vec![],
            },
            payloads: vec![],
            deployments: vec![PublishDeployment {
                body: wire::DeploymentBody {
                    deployment_id: "gb202-tp1".into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 1,
                    contract: Some(crate::runtime_package::wire::ServingContract::default()),
                    cluster: None,
                    rebuild: None,
                    engram: None,
                    payloads: vec![],
                    ranks: vec![],
                },
                ranks: vec![PublishRank {
                    body: wire::RankBody {
                        rank: 0,
                        deployment_id: "gb202-tp1".into(),
                        context: Some(crate::runtime_package::wire::Context::default()),
                        modules: vec![wire::Module {
                            name: "kernel".into(),
                            cubin: 0,
                            globals: vec![],
                        }],
                        functions: vec![wire::Function {
                            module: "kernel".into(),
                            symbol: "decode".into(),
                            parameter_sizes: vec![],
                            attributes: vec![],
                        }],
                        graphs: vec![wire::Graph {
                            name: "decode".into(),
                            flags: 0,
                            resource: 1,
                            node_count: 1,
                            edge_count: 0,
                            functions: vec![0],
                        }],
                        allocations: vec![wire::Allocation {
                            name: "device-0".into(),
                            nbytes: weight_bytes.len() as u64,
                            alignment: 256,
                            address: Some(0x1000),
                            memory: 0,
                            sizing: None,
                            relocations: vec![],
                        }],
                        ..Default::default()
                    },
                    payloads: vec![
                        payload(wire::PayloadRole::Weights, &weight_path, weight_id, &weight_bytes),
                        payload(wire::PayloadRole::InitialState, &state_path, state_id, &state_bytes),
                    ],
                    resources: vec![
                        PublishResource {
                            kind: wire::ExecutableKind::Cubin,
                            path: kernel_path,
                            nbytes: kernel_bytes.len() as u64,
                        },
                        PublishResource {
                            kind: wire::ExecutableKind::Graph,
                            path: graph_path,
                            nbytes: graph_bytes.len() as u64,
                        },
                    ],
                }],
                payloads: vec![],
            }],
            package_id,
            public_key,
            accepted: "example".into(),
            no_weights: true,
        };
        // A deployment with no graphs/executable resources is rejected only by
        // deployment validation, so this exercises role gating first: fresh
        // state and kernel payloads are accepted, cached weights are required.
        let entry = "gb202-tp1-000000000000";
        let mut staging = root.stage("model", entry).unwrap();
        let result = publish(&request, &key, Destination::Cache(&root, &mut staging));
        match result {
            Ok(_) => {
                // Fresh objects were written; cached weight reused.
                assert!(root.payload(state_id).unwrap().is_some());
                assert!(root.payload(weight_id).unwrap().is_some());
            }
            Err(error) => panic!("fresh state/executable payloads must not require cache: {error}"),
        }
        let object = root.payload(weight_id).unwrap().unwrap();
        let mut request = request;
        let weight = request.deployments[0].ranks[0].payloads.iter_mut()
            .find(|payload| payload.role == wire::PayloadRole::Weights).unwrap();
        let live_source = weight.source.clone();
        weight.source = object.slice().clone();
        let mut fast = root.stage("model", "reuse").unwrap();
        publish_reusing_cached_weights(&request, &key, Destination::Cache(&root, &mut fast), &[object.clone()]).unwrap();

        request.deployments[0].ranks[0].payloads.iter_mut()
            .find(|payload| payload.role == wire::PayloadRole::Weights).unwrap().source = live_source;
        let mut unpinned = root.stage("model", "unpinned").unwrap();
        let error = publish_reusing_cached_weights(&request, &key, Destination::Cache(&root, &mut unpinned), &[object.clone()]).unwrap_err();
        assert!(error.to_string().contains("leased cache object"), "{error}");

        request.deployments[0].ranks[0].payloads.iter_mut()
            .find(|payload| payload.role == wire::PayloadRole::Weights).unwrap().source = object.slice().clone();
        std::fs::write(&state_path, vec![0xff; state_bytes.len()]).unwrap();
        let mut changed_state = root.stage("model", "changed-state").unwrap();
        let error = publish_reusing_cached_weights(&request, &key, Destination::Cache(&root, &mut changed_state), &[object]).unwrap_err();
        assert!(error.to_string().contains("bytes do not match"), "{error}");
        let _ = std::fs::remove_dir_all(&root_dir);
    }

    #[test]
    fn publish_never_replaces_an_existing_destination_directory() {
        let root = scratch("noreplace");
        let destination = root.join("package");
        std::fs::create_dir(&destination).unwrap();
        std::fs::write(destination.join("user-file"), b"keep").unwrap();
        let sentinel = destination.join("model.safetensors");
        assert!(!sentinel.exists());
        let request = PublishRequest {
            package: wire::PackageBody {
                format_version: schema::FORMAT_VERSION,
                package_id: vec![1; 32],
                model_identity: "model".into(),
                hardware_identity: "sm120".into(),
                publisher: None,
                frontend: None,
                sampling_defaults: None,
                payloads: vec![],
                assets: vec![],
                deployments: vec![],
            },
            payloads: vec![],
            deployments: vec![PublishDeployment {
                body: wire::DeploymentBody {
                    deployment_id: "sm120-tp1".into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 1,
                    contract: Some(crate::runtime_package::wire::ServingContract::default()),
                    cluster: None,
                    rebuild: None,
                    engram: None,
                    payloads: vec![],
                    ranks: vec![],
                },
                ranks: vec![],
                payloads: vec![],
            }],
            package_id: [1; 32],
            public_key: [2; 32],
            accepted: "example".into(),
            no_weights: false,
        };
        let key = ring::signature::Ed25519KeyPair::from_pkcs8(
            ring::signature::Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new())
                .unwrap()
                .as_ref(),
        )
        .unwrap();
        assert!(publish(&request, &key, Destination::Directory(&destination)).is_err());
        assert_eq!(
            std::fs::read(destination.join("user-file")).unwrap(),
            b"keep"
        );
        assert!(!sentinel.exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn failed_publication_leaves_no_partial_public_tree() {
        let root = scratch("partial");
        let destination = root.join("package");
        let bytes = b"payload".to_vec();
        let path = root.join("payload.bin");
        std::fs::write(&path, &bytes).unwrap();
        let request = PublishRequest {
            package: wire::PackageBody {
                format_version: schema::FORMAT_VERSION,
                package_id: vec![1; 32],
                model_identity: "model".into(),
                hardware_identity: "sm120".into(),
                publisher: Some(wire::PublisherMetadata {
                    namespace: "example".into(),
                    official_source: String::new(),
                    development: false,
                    license_name: String::new(),
                    license_link: String::new(),
                    ranks: 1,
                }),
                frontend: None,
                sampling_defaults: None,
                payloads: vec![],
                assets: vec![],
                deployments: vec![],
            },
            payloads: vec![],
            deployments: vec![PublishDeployment {
                body: wire::DeploymentBody {
                    deployment_id: "sm120-tp1".into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 1,
                    contract: None,
                    cluster: None,
                    rebuild: None,
                    engram: None,
                    payloads: vec![],
                    ranks: vec![],
                },
                ranks: vec![PublishRank {
                    body: wire::RankBody {
                        rank: 0,
                        deployment_id: "sm120-tp1".into(),
                        allocations: vec![],
                        initializers: vec![],
                        ..Default::default()
                    },
                    payloads: vec![PublishPayload {
                        role: wire::PayloadRole::Weights,
                        relative: "weights/layer-000.safetensors".into(),
                        source: pin_file(&path, bytes.len() as u64).unwrap(),
                        id: PayloadId::of(b"other"),
                        tensor: None,
                    }],
                    resources: vec![],
                }],
                payloads: vec![],
            }],
            package_id: [1; 32],
            public_key: [2; 32],
            accepted: "example".into(),
            no_weights: false,
        };
        let key = ring::signature::Ed25519KeyPair::from_pkcs8(
            ring::signature::Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new())
                .unwrap()
                .as_ref(),
        )
        .unwrap();
        assert!(publish(&request, &key, Destination::Directory(&destination)).is_err());
        assert!(!destination.exists());
        let leftovers: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains("flare") || name == "package")
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn publish_rejects_traversal_paths_before_writing() {
        let root = scratch("traversal");
        let outside = root.parent().unwrap().join("flare-export-escape");
        let payload_path = root.join("payload.bin");
        std::fs::write(&payload_path, b"bytes").unwrap();
        let request = PublishRequest {
            package: wire::PackageBody {
                format_version: schema::FORMAT_VERSION,
                package_id: vec![1; 32],
                model_identity: "model".into(),
                hardware_identity: "sm120".into(),
                publisher: Some(wire::PublisherMetadata {
                    namespace: "example".into(),
                    official_source: String::new(),
                    development: false,
                    license_name: String::new(),
                    license_link: String::new(),
                    ranks: 1,
                }),
                frontend: None,
                sampling_defaults: None,
                payloads: vec![],
                assets: vec![],
                deployments: vec![],
            },
            payloads: vec![],
            deployments: vec![PublishDeployment {
                body: wire::DeploymentBody {
                    deployment_id: "../escape".into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 1,
                    contract: None,
                    cluster: None,
                    rebuild: None,
                    engram: None,
                    payloads: vec![],
                    ranks: vec![],
                },
                ranks: vec![],
                payloads: vec![],
            }],
            package_id: [1; 32],
            public_key: [2; 32],
            accepted: "example".into(),
            no_weights: false,
        };
        let key = ring::signature::Ed25519KeyPair::from_pkcs8(
            ring::signature::Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new())
                .unwrap()
                .as_ref(),
        )
        .unwrap();
        assert!(publish(&request, &key, Destination::Directory(&root)).is_err());
        assert!(!outside.exists());
        assert!(!root.join("model.safetensors").exists());
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
    }

    #[test]
    fn publish_writes_signed_directory_and_seals_executables() {
        use crate::package::protection::{self, Selection};
        use crate::runtime_package::Publisher;
        use ring::signature::KeyPair;

        let root = scratch("publish");
        let weight_bytes = b"prepared weights".to_vec();
        let weight_path = root.join("layer.prepared");
        std::fs::write(&weight_path, &weight_bytes).unwrap();
        let cubin_plaintext = b"fake cubin bytes".to_vec();
        let cubin_path = root.join("kernel.plain");
        std::fs::write(&cubin_path, &cubin_plaintext).unwrap();
        let graph_plaintext = b"fake graph record".to_vec();
        let graph_path = root.join("graph.plain");
        std::fs::write(&graph_path, &graph_plaintext).unwrap();
        let id = PayloadId::of(&weight_bytes);
        let tokenizer = b"{\"model\":{\"type\":\"WordLevel\"}}".to_vec();
        let frontend_path = root.join("tokenizer.bytes");
        std::fs::write(&frontend_path, &tokenizer).unwrap();
        let frontend_id = PayloadId::of(&tokenizer);
        let rng = ring::rand::SystemRandom::new();
        let pkcs8 = ring::signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = ring::signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let package_id = [9u8; 32];
        let package = wire::PackageBody {
            format_version: schema::FORMAT_VERSION,
            package_id: package_id.to_vec(),
            model_identity: "model".into(),
            hardware_identity: "sm120".into(),
            publisher: Some(wire::PublisherMetadata {
                namespace: "example".into(),
                official_source: "https://huggingface.co/example/model".into(),
                development: false,
                license_name: "Example".into(),
                license_link: String::new(),
                ranks: 1,
            }),
            frontend: Some(crate::runtime_package::wire::FrontendSettings::default()),
            sampling_defaults: Some(
                crate::runtime_package::wire::DefaultSamplingParameters::default(),
            ),
            payloads: vec![],
            assets: vec![wire::FrontendAsset {
                name: "tokenizer".into(),
                payload: Some(wire::PayloadUse {
                    slice: Some(wire::PayloadSlice {
                        payload_id: frontend_id.as_bytes().to_vec(),
                        offset: 0,
                        nbytes: tokenizer.len() as u64,
                    }),
                    role: wire::PayloadRole::Frontend as i32,
                }),
                sha256: crate::package::protection::sha256(&tokenizer).to_vec(),
            }],
            deployments: vec![],
        };
        let rank = wire::RankBody {
            rank: 0,
            deployment_id: "sm120-tp1".into(),
            context: Some(crate::runtime_package::wire::Context::default()),
            reservations: vec![],
            allocations: vec![wire::Allocation {
                name: "device-0".into(),
                nbytes: weight_bytes.len() as u64,
                alignment: 256,
                address: Some(0x1000),
                memory: 0,
                sizing: None,
                relocations: vec![],
            }],
            initializers: vec![
                wire::Initializer {
                    allocation: "device-0".into(),
                    destination_offset: 0,
                    source: Some(wire::PayloadUse {
                        slice: Some(wire::PayloadSlice {
                            payload_id: id.as_bytes().to_vec(),
                            offset: 0,
                            nbytes: 8,
                        }),
                        role: wire::PayloadRole::Weights as i32,
                    }),
                },
                wire::Initializer {
                    allocation: "device-0".into(),
                    destination_offset: 8,
                    source: Some(wire::PayloadUse {
                        slice: Some(wire::PayloadSlice {
                            payload_id: id.as_bytes().to_vec(),
                            offset: 8,
                            nbytes: weight_bytes.len() as u64 - 8,
                        }),
                        role: wire::PayloadRole::Weights as i32,
                    }),
                },
            ],
            payloads: vec![],
            resources: vec![],
            modules: vec![wire::Module {
                name: "kernel".into(),
                cubin: 0,
                globals: vec![],
            }],
            functions: vec![wire::Function {
                module: "kernel".into(),
                symbol: "decode".into(),
                parameter_sizes: vec![],
                attributes: vec![],
            }],
            graphs: vec![wire::Graph {
                name: "decode".into(),
                flags: 0,
                resource: 1,
                node_count: 1,
                edge_count: 0,
                functions: vec![0],
            }],
            events: vec![],
            peers: vec![],
            execution: None,
            roce: None,
            prepared_weights: None,
        };
        let request = PublishRequest {
            package,
            payloads: vec![PublishPayload {
                role: wire::PayloadRole::Frontend,
                relative: String::new(),
                source: pin_file(&frontend_path, tokenizer.len() as u64).unwrap(),
                id: frontend_id,
                tensor: Some("frontend.tokenizer".into()),
            }],
            deployments: vec![PublishDeployment {
                body: wire::DeploymentBody {
                    deployment_id: "sm120-tp1".into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 1,
                    contract: Some(crate::runtime_package::wire::ServingContract::default()),
                    cluster: None,
                    rebuild: None,
                    engram: None,
                    payloads: vec![],
                    ranks: vec![],
                },
                payloads: vec![],
                ranks: vec![PublishRank {
                    body: rank,
                    payloads: vec![PublishPayload {
                        role: wire::PayloadRole::Weights,
                        relative: "weights/layer-000.safetensors".into(),
                        source: pin_file(&weight_path, weight_bytes.len() as u64).unwrap(),
                        id,
                        tensor: None,
                    }],
                    resources: vec![
                        PublishResource {
                            kind: wire::ExecutableKind::Cubin,
                            path: cubin_path,
                            nbytes: cubin_plaintext.len() as u64,
                        },
                        PublishResource {
                            kind: wire::ExecutableKind::Graph,
                            path: graph_path,
                            nbytes: graph_plaintext.len() as u64,
                        },
                    ],
                }],
            }],
            package_id,
            public_key,
            accepted: "example".into(),
            no_weights: false,
        };
        let published = publish(&request, &key, Destination::Directory(&root)).unwrap();
        assert_eq!(published.payloads.len(), 4);
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let metadata = crate::package::reader::Metadata::open(&root).unwrap();
        assert_eq!(metadata.envelope, published.manifest);
        let verified = protection::verify(
            &metadata.envelope,
            &[publisher],
            Selection::Deployment {
                deployment: 0,
                rank: Some(0),
            },
        )
        .unwrap();
        let catalog = verified.catalog().clone();
        let files = metadata
            .payloads(&catalog, [id.as_bytes().as_slice()])
            .unwrap();
        files.verify().unwrap();
        let mut stored_weight = Vec::new();
        files
            .get(id.as_bytes())
            .unwrap()
            .copy_to(&mut stored_weight)
            .unwrap();
        assert_eq!(stored_weight, weight_bytes);
        let coalesced = &verified.ranks()[0].initializers;
        assert_eq!(coalesced.len(), 1);
        assert_eq!(coalesced[0].allocation, "device-0");
        assert_eq!(coalesced[0].destination_offset, 0);
        let coalesced_source = coalesced[0]
            .source
            .as_ref()
            .unwrap()
            .slice
            .as_ref()
            .unwrap();
        assert_eq!(coalesced_source.offset, 0);
        assert_eq!(coalesced_source.nbytes, weight_bytes.len() as u64);
        let authorized = verified.authorize("example").unwrap();
        let resource = &authorized.verified().ranks()[0].resources[0];
        let path = metadata
            .locations
            .locations
            .iter()
            .find(|location| {
                location.payload_id == resource.stored.as_ref().unwrap().payload_id
            })
            .unwrap();
        let (_, file) = container::open_relative(&root, &path.file).unwrap();
        let container = container::Container::open_pinned(file).unwrap();
        let slice = container.tensor(&path.tensor).unwrap();
        let mut stored = Vec::new();
        slice.copy_to(&mut stored).unwrap();
        let plaintext = authorized.open_resource(resource, &stored).unwrap();
        assert_eq!(plaintext, cubin_plaintext);
        assert!(root.join("weights/layer-000.safetensors").exists());
        assert!(
            root.join("deployments/sm120-tp1/rank-0/kernels.safetensors")
                .exists()
        );
        assert!(
            root.join("deployments/sm120-tp1/rank-0/graphs.safetensors")
                .exists()
        );
        let primary = container::Container::open(&root.join("model.safetensors")).unwrap();
        let mut embedded = Vec::new();
        primary
            .tensor("frontend.tokenizer")
            .unwrap()
            .copy_to(&mut embedded)
            .unwrap();
        assert_eq!(embedded, tokenizer);
    }

    #[test]
    fn identical_rank_executable_groups_install_one_companion() {
        use crate::package::cache::Root;
        use crate::package::protection::{self, Selection};
        use crate::runtime_package::Publisher;
        use ring::signature::KeyPair;

        let work = scratch("identical-rank-executables");
        let source_dir = work.join("sources");
        std::fs::create_dir_all(&source_dir).unwrap();
        let cubin = b"identical executable cubin bytes".to_vec();
        let graph = b"identical executable graph bytes".to_vec();
        let cubin_path = source_dir.join("kernel.cubin");
        let graph_path = source_dir.join("graph.bin");
        std::fs::write(&cubin_path, &cubin).unwrap();
        std::fs::write(&graph_path, &graph).unwrap();
        let rng = ring::rand::SystemRandom::new();
        let pkcs8 = ring::signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = ring::signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let package_id = [17u8; 32];
        let deployment_id = "sm120-tp2";
        let rank = |number: u32| wire::RankBody {
            rank: number,
            deployment_id: deployment_id.into(),
            context: Some(crate::runtime_package::wire::Context::default()),
            allocations: vec![wire::Allocation {
                name: "storage".into(),
                nbytes: 1,
                alignment: 256,
                address: Some(0x1000),
                memory: 0,
                sizing: None,
                relocations: vec![],
            }],
            modules: vec![wire::Module {
                name: "kernel".into(),
                cubin: 0,
                globals: vec![],
            }],
            functions: vec![wire::Function {
                module: "kernel".into(),
                symbol: "decode".into(),
                parameter_sizes: vec![],
                attributes: vec![],
            }],
            graphs: vec![wire::Graph {
                name: "decode".into(),
                resource: 1,
                node_count: 1,
                edge_count: 0,
                functions: vec![0],
                ..Default::default()
            }],
            ..Default::default()
        };
        let request = PublishRequest {
            package: wire::PackageBody {
                format_version: schema::FORMAT_VERSION,
                package_id: package_id.to_vec(),
                model_identity: "model".into(),
                hardware_identity: "sm120".into(),
                publisher: Some(wire::PublisherMetadata {
                    namespace: "example".into(),
                    official_source: "https://huggingface.co/example/model".into(),
                    development: false,
                    license_name: String::new(),
                    license_link: String::new(),
                    ranks: 2,
                }),
                frontend: Some(crate::runtime_package::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    crate::runtime_package::wire::DefaultSamplingParameters::default(),
                ),
                payloads: vec![],
                assets: vec![],
                deployments: vec![],
            },
            payloads: vec![],
            deployments: vec![PublishDeployment {
                body: wire::DeploymentBody {
                    deployment_id: deployment_id.into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 2,
                    contract: Some(crate::runtime_package::wire::ServingContract::default()),
                    cluster: None,
                    rebuild: None,
                    engram: None,
                    payloads: vec![],
                    ranks: vec![],
                },
                ranks: (0..2)
                    .map(|number| PublishRank {
                        body: rank(number),
                        payloads: vec![],
                        resources: vec![
                            PublishResource {
                                kind: wire::ExecutableKind::Cubin,
                                path: cubin_path.clone(),
                                nbytes: cubin.len() as u64,
                            },
                            PublishResource {
                                kind: wire::ExecutableKind::Graph,
                                path: graph_path.clone(),
                                nbytes: graph.len() as u64,
                            },
                        ],
                    })
                    .collect(),
                payloads: vec![],
            }],
            package_id,
            public_key,
            accepted: "example".into(),
            no_weights: false,
        };
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };

        let verify = |directory: &Path| {
            let metadata = crate::package::reader::Metadata::open(directory).unwrap();
            let verified = protection::verify(
                &metadata.envelope,
                std::slice::from_ref(&publisher),
                Selection::Deployment {
                    deployment: 0,
                    rank: None,
                },
            )
            .unwrap();
            let authorized = verified.authorize("example").unwrap();
            let ranks = authorized.verified().ranks();
            assert_eq!(ranks.len(), 2);
            for rank in ranks {
                for (resource, expected) in [
                    (&rank.resources[0], cubin.as_slice()),
                    (&rank.resources[1], graph.as_slice()),
                ] {
                    let stored = resource.stored.as_ref().unwrap();
                    let files = metadata
                        .payloads(authorized.verified().catalog(), [stored.payload_id.as_slice()])
                        .unwrap();
                    files.verify().unwrap();
                    let mut bytes = vec![0u8; stored.nbytes as usize];
                    files
                        .get(&stored.payload_id)
                        .unwrap()
                        .read_at(stored.offset, &mut bytes)
                        .unwrap();
                    assert_eq!(authorized.open_resource(resource, &bytes).unwrap(), expected);
                }
            }
            assert!(
                directory
                    .join("deployments/sm120-tp2/rank-0/kernels.safetensors")
                    .is_file()
            );
            assert!(
                directory
                    .join("deployments/sm120-tp2/rank-0/graphs.safetensors")
                    .is_file()
            );
            assert!(
                !directory
                    .join("deployments/sm120-tp2/rank-1/kernels.safetensors")
                    .exists()
            );
            assert!(
                !directory
                    .join("deployments/sm120-tp2/rank-1/graphs.safetensors")
                    .exists()
            );
        };

        let directory = work.join("directory");
        std::fs::create_dir_all(&directory).unwrap();
        let published = publish(&request, &key, Destination::Directory(&directory)).unwrap();
        assert_eq!(published.payloads.len(), 2);
        verify(&directory);
        drop(published);

        let cache_dir = work.join("cache");
        std::fs::create_dir_all(&cache_dir).unwrap();
        let root = Root::open(&cache_dir).unwrap();
        let mut staging = root.stage("model", "entry").unwrap();
        let published =
            publish(&request, &key, Destination::Cache(&root, &mut staging)).unwrap();
        assert_eq!(published.payloads.len(), 2);
        let entry = staging.commit().unwrap();
        verify(entry.directory());
        drop(published);
        drop(entry);
        drop(root);
        drop(request);
        let _ = std::fs::remove_dir_all(&work);
    }

    #[test]
    fn shared_payloads_install_concurrently_with_serial_bookkeeping() {
        use crate::package::cache::Root;
        use crate::package::protection::{self, Selection};
        use crate::runtime_package::Publisher;
        use ring::signature::KeyPair;

        let work = scratch("shared-concurrent");
        let source_dir = work.join("sources");
        std::fs::create_dir_all(&source_dir).unwrap();
        let write = |name: &str, bytes: &[u8]| {
            let path = source_dir.join(name);
            std::fs::write(&path, bytes).unwrap();
            path
        };
        let pattern = |seed: u8, nbytes: usize| -> Vec<u8> {
            (0..nbytes).map(|index| (index as u8).wrapping_mul(31).wrapping_add(seed)).collect()
        };
        let cubin = b"shared payload test cubin".to_vec();
        let graph = b"shared payload test graph".to_vec();
        let cubin_path = write("kernel.cubin", &cubin);
        let graph_path = write("graph.bin", &graph);
        // Two large weight tables, a second occurrence of the first from another file, and a
        // state payload whose object is already cached before publication.
        let table_a = pattern(1, 3 << 20);
        let table_b = pattern(2, 2 << 20);
        let state = pattern(3, 4096);
        let (id_a, id_b, id_state) =
            (PayloadId::of(&table_a), PayloadId::of(&table_b), PayloadId::of(&state));
        let path_a = write("table-a.bin", &table_a);
        let path_a_copy = write("table-a-copy.bin", &table_a);
        let path_b = write("table-b.bin", &table_b);
        let path_state = write("state.bin", &state);
        let deployment_id = "sm120-tp1";
        let shared = |role, name: &str, path: &Path, id| PublishPayload {
            role,
            relative: format!("deployments/{deployment_id}/shared/{name}.safetensors"),
            source: pin_file(path, std::fs::metadata(path).unwrap().len()).unwrap(),
            id,
            tensor: None,
        };
        let rng = ring::rand::SystemRandom::new();
        let pkcs8 = ring::signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = ring::signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let package_id = [23u8; 32];
        let request = PublishRequest {
            package: wire::PackageBody {
                format_version: schema::FORMAT_VERSION,
                package_id: package_id.to_vec(),
                model_identity: "model".into(),
                hardware_identity: "sm120".into(),
                publisher: Some(wire::PublisherMetadata {
                    namespace: "example".into(),
                    official_source: "https://huggingface.co/example/model".into(),
                    development: false,
                    license_name: String::new(),
                    license_link: String::new(),
                    ranks: 1,
                }),
                frontend: Some(crate::runtime_package::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    crate::runtime_package::wire::DefaultSamplingParameters::default(),
                ),
                payloads: vec![],
                assets: vec![],
                deployments: vec![],
            },
            payloads: vec![],
            deployments: vec![PublishDeployment {
                body: wire::DeploymentBody {
                    deployment_id: deployment_id.into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 1,
                    contract: Some(crate::runtime_package::wire::ServingContract::default()),
                    cluster: None,
                    rebuild: None,
                    engram: None,
                    payloads: vec![],
                    ranks: vec![],
                },
                ranks: vec![PublishRank {
                    body: wire::RankBody {
                        rank: 0,
                        deployment_id: deployment_id.into(),
                        context: Some(crate::runtime_package::wire::Context::default()),
                        allocations: vec![wire::Allocation {
                            name: "storage".into(),
                            nbytes: 1,
                            alignment: 256,
                            address: Some(0x1000),
                            memory: 0,
                            sizing: None,
                            relocations: vec![],
                        }],
                        modules: vec![wire::Module {
                            name: "kernel".into(),
                            cubin: 0,
                            globals: vec![],
                        }],
                        functions: vec![wire::Function {
                            module: "kernel".into(),
                            symbol: "decode".into(),
                            parameter_sizes: vec![],
                            attributes: vec![],
                        }],
                        graphs: vec![wire::Graph {
                            name: "decode".into(),
                            resource: 1,
                            node_count: 1,
                            edge_count: 0,
                            functions: vec![0],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                    payloads: vec![],
                    resources: vec![
                        PublishResource {
                            kind: wire::ExecutableKind::Cubin,
                            path: cubin_path,
                            nbytes: cubin.len() as u64,
                        },
                        PublishResource {
                            kind: wire::ExecutableKind::Graph,
                            path: graph_path,
                            nbytes: graph.len() as u64,
                        },
                    ],
                }],
                payloads: vec![
                    shared(wire::PayloadRole::InitialState, "state", &path_state, id_state),
                    shared(wire::PayloadRole::Weights, "table-b", &path_b, id_b),
                    shared(wire::PayloadRole::Weights, "table-a", &path_a, id_a),
                    shared(wire::PayloadRole::Weights, "table-a-again", &path_a_copy, id_a),
                ],
            }],
            package_id,
            public_key,
            accepted: "example".into(),
            no_weights: false,
        };
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let shared_order = |directory: &Path| {
            let metadata = crate::package::reader::Metadata::open(directory).unwrap();
            let verified = protection::verify(
                &metadata.envelope,
                std::slice::from_ref(&publisher),
                Selection::Deployment { deployment: 0, rank: None },
            )
            .unwrap();
            let order: Vec<Vec<u8>> = verified
                .deployment()
                .unwrap()
                .payloads
                .iter()
                .map(|descriptor| descriptor.id.clone())
                .collect();
            (metadata, order)
        };
        let blobs = |cache: &Path| {
            std::fs::read_dir(cache.join("blobs"))
                .unwrap()
                .filter(|item| item.as_ref().unwrap().file_type().unwrap().is_file())
                .count()
        };

        // The directory destination keeps the serial installation: the reference bookkeeping.
        let directory = work.join("directory");
        std::fs::create_dir_all(&directory).unwrap();
        publish(&request, &key, Destination::Directory(&directory)).unwrap();
        let (reference, reference_order) = shared_order(&directory);

        let cache_dir = work.join("cache");
        std::fs::create_dir_all(&cache_dir).unwrap();
        let root = Root::open(&cache_dir).unwrap();
        root.capture_payload(state.len() as u64, state.as_slice()).unwrap();
        let mut staging = root.stage("model", "entry").unwrap();
        publish(&request, &key, Destination::Cache(&root, &mut staging)).unwrap();
        let entry = staging.commit().unwrap();
        let (metadata, order) = shared_order(entry.directory());
        // Signatures are deterministic, so equal envelopes mean equal bodies and catalogs.
        assert_eq!(metadata.envelope, reference.envelope);
        assert_eq!(metadata.locations, reference.locations);
        let tail = order[order.len() - 4..].to_vec();
        assert_eq!(order, reference_order);
        let expected: Vec<Vec<u8>> =
            [id_state, id_b, id_a, id_a].iter().map(|id| id.as_bytes().to_vec()).collect();
        assert_eq!(tail, expected, "shared payloads must keep their declared catalog order");
        // One object per identity: three shared payloads plus the kernel and graph companions.
        assert_eq!(blobs(&cache_dir), 5);
        for (id, bytes) in [(id_a, &table_a), (id_b, &table_b), (id_state, &state)] {
            let object = root.payload(id).unwrap().unwrap();
            assert_eq!(object.slice().digest().unwrap(), id);
            let mut stored = Vec::new();
            object.slice().copy_to(&mut stored).unwrap();
            assert_eq!(&stored, bytes);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let object = root.payload(id_a).unwrap().unwrap();
            let inode = std::fs::metadata(object.path()).unwrap().ino();
            for name in ["table-a", "table-a-again"] {
                let location = metadata
                    .locations
                    .locations
                    .iter()
                    .find(|location| location.file.ends_with(&format!("/{name}.safetensors")));
                // Only the first occurrence is located; both names resolve to one object.
                if let Some(location) = location {
                    let linked = entry.directory().join(&location.file);
                    assert_eq!(std::fs::metadata(linked).unwrap().ino(), inode);
                }
            }
        }
        drop(entry);

        // A corrupted source still fails verification and installs nothing for it.
        let fresh = pattern(4, 1 << 20);
        let fresh_id = PayloadId::of(&fresh);
        let fresh_path = write("fresh.bin", &fresh);
        let mut corrupted = request;
        corrupted.deployments[0]
            .payloads
            .push(shared(wire::PayloadRole::Weights, "fresh", &fresh_path, fresh_id));
        std::fs::write(&fresh_path, vec![0x5a; fresh.len()]).unwrap();
        let mut staging = root.stage("model", "corrupted").unwrap();
        let error = publish(&corrupted, &key, Destination::Cache(&root, &mut staging)).unwrap_err();
        assert!(error.to_string().contains("bytes do not match the assigned identity"), "{error}");
        assert!(root.payload(fresh_id).unwrap().is_none());
        drop(staging);

        // Reusing cached weights skips the source pass, so the concurrent insertion itself must
        // check the fresh shared source before copying it.
        let object_a = root.payload(id_a).unwrap().unwrap();
        let object_b = root.payload(id_b).unwrap().unwrap();
        let mut reuse = corrupted;
        let fresh_state = reuse.deployments[0].payloads.last_mut().unwrap();
        fresh_state.role = wire::PayloadRole::InitialState;
        for payload in &mut reuse.deployments[0].payloads {
            if payload.id == id_a {
                payload.source = object_a.slice().clone();
            } else if payload.id == id_b {
                payload.source = object_b.slice().clone();
            }
        }
        let objects = [object_a.clone(), object_b.clone()];
        let mut staging = root.stage("model", "reuse-corrupted").unwrap();
        let error =
            publish_reusing_cached_weights(&reuse, &key, Destination::Cache(&root, &mut staging), &objects)
                .unwrap_err();
        assert!(error.to_string().contains("bytes do not match the assigned identity"), "{error}");
        assert!(root.payload(fresh_id).unwrap().is_none());
        drop(staging);

        std::fs::write(&fresh_path, &fresh).unwrap();
        let mut staging = root.stage("model", "reuse").unwrap();
        publish_reusing_cached_weights(&reuse, &key, Destination::Cache(&root, &mut staging), &objects)
            .unwrap();
        let entry = staging.commit().unwrap();
        let (_, order) = shared_order(entry.directory());
        assert_eq!(
            order[order.len() - 5..].to_vec(),
            [id_state, id_b, id_a, id_a, fresh_id].iter().map(|id| id.as_bytes().to_vec()).collect::<Vec<_>>()
        );
        let installed = root.payload(fresh_id).unwrap().unwrap();
        assert_eq!(installed.slice().digest().unwrap(), fresh_id);
        assert_eq!(blobs(&cache_dir), 6);
        drop((entry, installed, objects, object_a, object_b));

        // A cached object is hashed by the concurrent pass before it is linked: corrupting the
        // cached state in place fails publication although its source is intact.
        let state_object = root.payload(id_state).unwrap().unwrap();
        let state_path = state_object.path().to_path_buf();
        let state_end = (state_object.slice().offset() + state_object.slice().len()) as usize;
        drop(state_object);
        let mut permissions = std::fs::metadata(&state_path).unwrap().permissions();
        #[allow(clippy::permissions_set_readonly_false)]
        permissions.set_readonly(false);
        std::fs::set_permissions(&state_path, permissions).unwrap();
        let mut stored = std::fs::read(&state_path).unwrap();
        stored[state_end - 1] ^= 0xff;
        std::fs::write(&state_path, &stored).unwrap();
        let mut staging = root.stage("model", "corrupted-cache").unwrap();
        let error = publish(&reuse, &key, Destination::Cache(&root, &mut staging)).unwrap_err();
        assert!(
            error.to_string().contains(&format!("cached payload {id_state} bytes do not match their identity")),
            "{error}"
        );
        drop((staging, root));
        let _ = std::fs::remove_dir_all(&work);
    }

    #[test]
    fn duplicate_payload_paths_are_rejected() {
        let source = scratch("a-source");
        let output = scratch("a-output");
        let part = write_source(&source, "storage.bin", b"abcd");
        let span = CapturedSpan {
            allocation: "device-0".into(),
            destination_offset: 0,
            source_offset: 0,
            nbytes: 4,
        };
        let ranks = vec![CapturedRank {
            rank: 0,
            groups: vec![
                group("weights/layer-000.safetensors", vec![span.clone()]),
                group("weights/layer-000.safetensors", vec![span]),
            ],
            sources: vec![part],
        }];
        assert!(assemble(&output, &ranks).is_err());
    }
}
