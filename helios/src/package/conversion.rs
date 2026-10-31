/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Keyless conversion between the packed file and directory canonical forms.
//!
//! Conversion copies signed metadata, derivations, frontend assets and payload
//! bytes exactly; only the unsigned `flare.locations` tensor is re-encoded. It
//! checks body commitments, validates the descriptor closure and hashes every
//! located payload against its signed identity. Signatures and compressed executables are
//! preserved byte-for-byte but are not authenticated here: publisher trust
//! belongs to `protection::verify`, which callers run before or after
//! conversion.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::Write as _,
    path::{Path, PathBuf},
};

use prost::Message;

use super::{
    PayloadId,
    container::{self, COMPANION_TENSOR, Source, Writer},
    derivation, invalid, protection,
    reader::{self, Metadata},
    schema::{self, Catalog, wire},
};
use crate::{Result, runtime_package::wire as runtime};
use ring::digest::{Context as DigestContext, SHA256};

/// Container form of a payload-format package.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    Packed,
    Directory,
}

/// Classifies a payload-format package without choosing a deployment.
///
/// Returns `None` for a package that is not in the payload format, so callers
/// can dispatch legacy conversion instead of treating it as corrupt.
pub fn form(source: &Path) -> Result<Option<Form>> {
    let path = if source.is_dir() {
        source.join("model.safetensors")
    } else {
        source.to_path_buf()
    };
    let container = match container::Container::open(&path) {
        Ok(container) => container,
        Err(_) => return Ok(None),
    };
    if container.format() != container::PAYLOAD_FORMAT {
        return Ok(None);
    }
    // Directory form is selected by the argument or by any external payload
    // location. A packed primary keeps every payload inside itself. This lets
    // callers pass either the package directory or its model.safetensors.
    let metadata = Metadata::open(source)?;
    Ok(Some(if directory_form(source, &metadata) {
        Form::Directory
    } else {
        Form::Packed
    }))
}

fn directory_form(source: &Path, metadata: &Metadata) -> bool {
    source.is_dir()
        || metadata
            .locations
            .locations
            .iter()
            .any(|location| !location.file.is_empty())
}

/// Rewrites a directory package as one packed container.
pub fn pack(source: &Path, output: &Path) -> Result<()> {
    pack_selected(source, output, None)
}

/// Choose a unique canonical companion path.
fn unique_companion_path(
    file: &str,
    tensor: &str,
    identity: PayloadId,
    used: &mut BTreeSet<String>,
) -> Result<String> {
    let candidate = crate::package::container::normalized_payload_companion(file, tensor, identity);
    if candidate != "model.safetensors"
        && container::validate_relative_path(&candidate).is_ok()
        && !container::path_conflicts(&candidate, used)
    {
        used.insert(candidate.clone());
        return Ok(candidate);
    }
    let fallback = format!("packed/{}.safetensors", identity.to_hex());
    if !container::path_conflicts(&fallback, used) {
        used.insert(fallback.clone());
        return Ok(fallback);
    }
    Err(invalid(format!(
        "cannot assign a unique companion path to payload {identity}"
    )))
}

fn packed_tensor_for(
    file: &str,
    tensor: &str,
    identity: PayloadId,
    used_paths: &mut BTreeSet<String>,
    reserved: &BTreeSet<String>,
) -> Result<String> {
    let relative = crate::package::container::normalized_payload_companion(file, tensor, identity);
    let candidate = container::packed_tensor_name(&relative).ok();
    let candidate = candidate.filter(|name| {
        !reserved.contains(name)
            && !name.starts_with("flare.")
            && !name.starts_with("frontend.")
    });
    if let Some(candidate) = candidate {
        if !used_paths.contains(&candidate) {
            return Ok(candidate);
        }
    }
    for suffix in 0u32.. {
        let fallback = if suffix == 0 {
            format!("packed.{}", identity.to_hex())
        } else {
            format!("packed.{}.{suffix}", identity.to_hex())
        };
        if !reserved.contains(&fallback) && !used_paths.contains(&fallback) {
            return Ok(fallback);
        }
    }
    unreachable!("unbounded packed tensor suffix search")
}

/// Packs only the closure of the selected deployments.
///
/// `selected` is `None` for every advertised deployment, or deployment indexes
/// for a ready cache entry holding a subset. Unselected payloads may be absent;
/// signed bodies and locations for them are preserved when present.
pub fn pack_selected(source: &Path, output: &Path, selected: Option<&[usize]>) -> Result<()> {
    let verified = Verified::open(source, selected)?;
    if !directory_form(source, &verified.metadata) {
        return Err(invalid("package is already in packed form"));
    }
    let payloads = verified.payloads()?;
    payloads.verify()?;

    let mut locations = Vec::with_capacity(verified.locations.locations.len());
    let mut packed = Vec::new();
    let mut names = BTreeSet::new();
    let reserved: BTreeSet<String> = verified
        .metadata
        .primary()
        .tensor_names()
        .map(str::to_owned)
        .collect();
    for location in &verified.locations.locations {
        if location.file.is_empty() {
            locations.push(location.clone());
            continue;
        }
        let identity = PayloadId::from_bytes(
            location
                .payload_id
                .as_slice()
                .try_into()
                .map_err(|_| invalid("payload id must be 32 bytes"))?,
        );
        let name =
            packed_tensor_for(&location.file, &location.tensor, identity, &mut names, &reserved)?;
        if !names.insert(name.clone()) {
            return Err(invalid("two companions map to one packed tensor name"));
        }
        packed.push((name.clone(), location.payload_id.clone()));
        locations.push(wire::PayloadLocation {
            payload_id: location.payload_id.clone(),
            file: String::new(),
            tensor: name,
        });
    }
    let rewritten = wire::LocationSet {
        version: schema::FORMAT_VERSION,
        locations,
        available_deployments: verified.locations.available_deployments.clone(),
    };
    schema::validate_locations(&rewritten)?;

    if output.exists() {
        return Err(invalid(format!("destination exists: {}", output.display())));
    }
    let staging = unique_staging(output, "pack")?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staging)
        .map_err(|e| invalid(format!("create packed package: {e}")))?;
    let guard = TempFile::new(&staging);
    let mut writer = Writer::new(file, verified.metadata.primary().format())?;
    for name in verified.metadata.primary().tensor_names() {
        if name == reader::LOCATIONS || name.starts_with(container::ALIGNMENT_PREFIX) {
            continue;
        }
        let slice = verified.metadata.primary().tensor(name)?;
        writer.add(name, slice.len(), Source::Slice(slice))?;
    }
    let locations = rewritten.encode_to_vec();
    writer.add(
        reader::LOCATIONS,
        locations.len() as u64,
        Source::Bytes(locations),
    )?;
    for (name, id) in packed {
        let slice = payloads.get(&id)?.clone();
        writer.add(&name, slice.len(), Source::Slice(slice))?;
    }
    writer.finish()?;
    publish_staged_file(&staging, output)?;
    guard.keep();
    Ok(())
}

/// Rewrites a packed package as a directory with deterministic companions.
pub fn unpack(source: &Path, output: &Path) -> Result<()> {
    unpack_selected(source, output, None)
}

/// Unpacks only the closure of the selected deployments.
pub fn unpack_selected(source: &Path, output: &Path, selected: Option<&[usize]>) -> Result<()> {
    let verified = Verified::open(source, selected)?;
    if directory_form(source, &verified.metadata) {
        return Err(invalid("package is already in directory form"));
    }
    let payloads = verified.payloads()?;
    payloads.verify()?;

    let mut locations = Vec::with_capacity(verified.locations.locations.len());
    let mut companions = Vec::new();
    let mut paths = BTreeSet::from(["model.safetensors".to_owned()]);
    for location in &verified.locations.locations {
        if location.file.is_empty() && primary_payload(&location.tensor) {
            locations.push(location.clone());
            continue;
        }
        let identity = PayloadId::from_bytes(
            location
                .payload_id
                .as_slice()
                .try_into()
                .map_err(|_| invalid("payload id must be 32 bytes"))?,
        );
        let relative = unique_companion_path(
            &location.file,
            &location.tensor,
            identity,
            &mut paths,
        )?;
        companions.push((
            location.tensor.clone(),
            relative.clone(),
            payloads.get(&location.payload_id)?.clone(),
        ));
        locations.push(wire::PayloadLocation {
            payload_id: location.payload_id.clone(),
            file: relative,
            tensor: COMPANION_TENSOR.into(),
        });
    }
    let rewritten = wire::LocationSet {
        version: schema::FORMAT_VERSION,
        locations,
        available_deployments: verified.locations.available_deployments.clone(),
    };
    schema::validate_locations(&rewritten)?;

    if output.exists() {
        return Err(invalid(format!("destination exists: {}", output.display())));
    }
    let staging = unique_staging(output, "unpack")?;
    fs::create_dir(&staging).map_err(|e| invalid(format!("create package staging: {e}")))?;
    let result = (|| {
        for (_tensor, relative, slice) in &companions {
            container::write_payload_companion(&staging, relative, slice)?;
        }
        let primary = staging.join("model.safetensors");
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&primary)
            .map_err(|e| invalid(format!("create package primary: {e}")))?;
        let mut writer = Writer::new(file, verified.metadata.primary().format())?;
        for name in verified.metadata.primary().tensor_names() {
            if name == reader::LOCATIONS || !primary_payload(name) {
                continue;
            }
            let slice = verified.metadata.primary().tensor(name)?;
            writer.add(name, slice.len(), Source::Slice(slice))?;
        }
        let locations = rewritten.encode_to_vec();
        writer.add(
            reader::LOCATIONS,
            locations.len() as u64,
            Source::Bytes(locations),
        )?;
        writer.finish()?;
        crate::package::cache::sync_file(&primary)
            .map_err(|e| invalid(format!("sync package primary: {e}")))?;
        // Companion entries must be durable before the directory becomes visible,
        // and an existing destination is never replaced.
        crate::package::cache::synchronize(&staging)?;
        crate::package::cache::rename_noreplace(&staging, output).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                invalid(format!("destination exists: {}", output.display()))
            } else {
                invalid(format!("publish package directory: {e}"))
            }
        })?;
        sync_parent(output)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

/// Byte relation between a derived-size payload and canonical payloads.
///
/// This is the combiner primitive: `Identical` references an existing signed
/// payload, `Corroborated` means a supplied candidate derivation reproduced the
/// target bytes exactly, and `PerSize` means the bytes must be stored for that
/// deployment. Replacing signed initializer or catalog bindings with shared
/// slices changes signed bodies and requires the publisher to sign them again;
/// this function never rewrites metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sharing {
    Identical(PayloadId),
    Corroborated,
    PerSize,
}

/// Compares a target payload with canonical payloads byte-for-byte.
///
/// When `candidate` is supplied it is checked with
/// [`derivation::verify_output`], which is the only acceptance test that permits
/// sharing a derived range. No digest list is trusted in place of the bytes.
pub fn sharing(
    target: &container::Slice,
    canonical: &BTreeMap<PayloadId, container::Slice>,
    candidate: Option<(
        &derivation::Derivation,
        &BTreeMap<PayloadId, container::Slice>,
    )>,
    limits: &derivation::Limits,
) -> Result<Sharing> {
    for (id, source) in canonical {
        if source.len() == target.len() && same_bytes(source, target)? {
            return Ok(Sharing::Identical(*id));
        }
    }
    if let Some((candidate, sources)) = candidate {
        derivation::verify_output(target, candidate, sources, limits)?;
        return Ok(Sharing::Corroborated);
    }
    Ok(Sharing::PerSize)
}

fn same_bytes(first: &container::Slice, second: &container::Slice) -> Result<bool> {
    if first.len() != second.len() {
        return Ok(false);
    }
    let mut left = vec![0u8; container::IO_CHUNK];
    let mut right = vec![0u8; container::IO_CHUNK];
    let mut offset = 0;
    while offset < first.len() {
        let count = (first.len() - offset).min(left.len() as u64) as usize;
        first.read_at(offset, &mut left[..count])?;
        second.read_at(offset, &mut right[..count])?;
        if left[..count] != right[..count] {
            return Ok(false);
        }
        offset += count as u64;
    }
    Ok(true)
}

/// Payload tensors inside the primary keep their place; other packed tensors
/// become directory companions. Frontend and metadata namespaces are reserved.
fn primary_payload(tensor: &str) -> bool {
    tensor.starts_with("flare.") || tensor.starts_with("frontend.")
}

struct Verified {
    metadata: Metadata,
    catalog: Catalog,
    locations: wire::LocationSet,
    ranks: Vec<wire::RankBody>,
}

impl Verified {
    fn open(source: &Path, selected: Option<&[usize]>) -> Result<Self> {
        let wanted = |index: usize| selected.is_none_or(|values| values.contains(&index));
        let metadata = Metadata::open(source)?;
        let envelope = protection::envelope(&metadata.envelope)?;
        let package = wire::PackageBody::decode(envelope.package_body.as_slice())
            .map_err(|e| invalid(format!("invalid package body: {e}")))?;
        let mut catalog = schema::validate_package(&package)?;

        if let Some(values) = selected {
            for index in values {
                if *index >= package.deployments.len() {
                    return Err(invalid("selected deployment is outside the package"));
                }
            }
        }
        let mut scope = Catalog::new();
        scope.insert_all(&package.payloads)?;
        let mut deployments: BTreeMap<u32, wire::DeploymentBody> = BTreeMap::new();
        for signed in &envelope.deployments {
            let commitment = package
                .deployments
                .get(signed.index as usize)
                .ok_or_else(|| invalid("deployment body index is out of range"))?;
            let body = wire::DeploymentBody::decode(signed.body.as_slice())
                .map_err(|e| invalid(format!("invalid deployment body: {e}")))?;
            if body.deployment_id != commitment.deployment_id {
                return Err(invalid("deployment body disagrees with its commitment"));
            }
            if protection::sha256(&signed.body).as_slice() != commitment.body_digest {
                return Err(invalid("deployment body differs from its commitment"));
            }
            schema::validate_deployment(&body, &mut catalog)?;
            if wanted(signed.index as usize) {
                scope.insert_all(&body.payloads)?;
            }
            if deployments.insert(signed.index, body).is_some() {
                return Err(invalid("duplicate deployment body"));
            }
        }
        if deployments.len() != package.deployments.len() {
            return Err(invalid("package is missing a deployment body"));
        }

        // Rank bodies are commitment-checked and contribute payload descriptors;
        // full execution validation belongs to the native loader.
        let mut covered: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
        let mut selected_ranks: Vec<wire::RankBody> = Vec::new();
        let mut consumers: BTreeSet<Vec<u8>> = BTreeSet::new();
        for asset in &package.assets {
            if let Some(use_) = &asset.payload
                && let Some(slice) = &use_.slice
            {
                consumers.insert(slice.payload_id.clone());
            }
        }
        for (index, deployment) in &deployments {
            if wanted(*index as usize)
                && let Some(engram) = &deployment.engram
            {
                for id in engram_payload_ids(engram) {
                    consumers.insert(id);
                }
            }
        }
        for signed in &envelope.ranks {
            let deployment = deployments
                .get(&signed.deployment)
                .ok_or_else(|| invalid("rank body names an absent deployment"))?;
            let body = wire::RankBody::decode(signed.body.as_slice())
                .map_err(|e| invalid(format!("invalid rank body: {e}")))?;
            if body.deployment_id != deployment.deployment_id || body.rank != signed.rank {
                return Err(invalid(
                    "rank body identity differs from its envelope record",
                ));
            }
            let rank_count = if deployment.tensor_parallel == 0 {
                deployment.ranks.len() as u32
            } else {
                deployment.tensor_parallel
            };
            if body.rank >= rank_count {
                return Err(invalid("rank id is outside its deployment"));
            }
            let commitment = deployment
                .ranks
                .iter()
                .find(|commitment| commitment.rank == signed.rank)
                .ok_or_else(|| invalid("rank body has no signed commitment"))?;
            if protection::sha256(&signed.body).as_slice() != commitment.body_digest {
                return Err(invalid("rank body differs from its commitment"));
            }
            if !covered
                .entry(signed.deployment)
                .or_default()
                .insert(signed.rank)
            {
                return Err(invalid("duplicate rank body"));
            }
            catalog.insert_all(&body.payloads)?;
            if wanted(signed.deployment as usize) {
                selected_ranks.push(body.clone());
                scope.insert_all(&body.payloads)?;
                rank_consumers(&body, &mut consumers);
            }
        }
        for (index, deployment) in &deployments {
            if !wanted(*index as usize) {
                continue;
            }
            let covered = covered.get(index).cloned().unwrap_or_default();
            if covered.len() != deployment.ranks.len()
                || deployment
                    .ranks
                    .iter()
                    .any(|commitment| !covered.contains(&commitment.rank))
            {
                return Err(invalid("deployment is missing a rank body"));
            }
        }

        let limits = derivation::Limits::default();
        let mut recipes: BTreeMap<Vec<u8>, derivation::Derivation> = BTreeMap::new();
        if let Some(set) = &metadata.derivations {
            for recipe in derivation::from_wire(set, &catalog, &limits)? {
                recipes.insert(recipe.target.as_bytes().to_vec(), recipe);
            }
        }
        // Expand consumers through the derivation graph: a target's recipes make
        // its sources required in turn, including multi-hop chains.
        let mut needed = consumers.clone();
        let mut pending: Vec<Vec<u8>> = consumers.iter().cloned().collect();
        while let Some(id) = pending.pop() {
            let Some(recipe) = recipes.get(&id) else {
                continue;
            };
            for operation in &recipe.operations {
                if let derivation::Operation::Copy(copy) = operation {
                    let source = copy.source.as_bytes().to_vec();
                    if needed.insert(source.clone()) {
                        pending.push(source);
                    }
                }
            }
        }
        for id in &needed {
            if let Some(recipe) = recipes.get(id) {
                scope.length(id)?;
                derivation::validate_sources(std::slice::from_ref(recipe), &scope, &limits)?;
            }
        }
        let located: BTreeSet<Vec<u8>> = metadata
            .locations
            .locations
            .iter()
            .map(|location| location.payload_id.clone())
            .collect();
        let mut visiting = BTreeSet::new();
        let mut available_ids = BTreeSet::new();
        for id in &needed {
            if !available(
                id,
                &located,
                &recipes_bytes(&recipes),
                &mut visiting,
                &mut available_ids,
            ) {
                return Err(invalid(format!(
                    "signed payload {} has no location and no complete derivation",
                    hex(id)
                )));
            }
        }
        let locations = wire::LocationSet {
            version: schema::FORMAT_VERSION,
            locations: metadata
                .locations
                .locations
                .iter()
                .filter(|location| needed.contains(&location.payload_id))
                .cloned()
                .collect(),
            available_deployments: if selected.is_some() {
                deployments
                    .iter()
                    .filter(|(index, _)| wanted(**index as usize))
                    .map(|(_, body)| body.deployment_id.clone())
                    .collect()
            } else {
                metadata.locations.available_deployments.clone()
            },
        };
        Ok(Self {
            metadata,
            catalog,
            locations,
            ranks: selected_ranks,
        })
    }

    /// Opens and hashes every located payload against its signed identity.
    fn payloads(&self) -> Result<reader::PayloadFiles> {
        let ids: Vec<&[u8]> = self
            .locations
            .locations
            .iter()
            .map(|location| location.payload_id.as_slice())
            .collect();
        let payloads = self.metadata.payloads(&self.catalog, ids)?;
        payloads.verify()?;
        Ok(payloads)
    }
}

fn rank_consumers(body: &wire::RankBody, out: &mut BTreeSet<Vec<u8>>) {
    for initializer in &body.initializers {
        if let Some(use_) = &initializer.source
            && let Some(slice) = &use_.slice
        {
            out.insert(slice.payload_id.clone());
        }
    }
    for module in &body.modules {
        for global in &module.globals {
            if let Some(use_) = &global.contents
                && let Some(slice) = &use_.slice
            {
                out.insert(slice.payload_id.clone());
            }
        }
    }
    for resource in &body.resources {
        if let Some(slice) = &resource.stored {
            out.insert(slice.payload_id.clone());
        }
    }
}

fn recipes_bytes(
    recipes: &BTreeMap<Vec<u8>, derivation::Derivation>,
) -> BTreeMap<Vec<u8>, Vec<Vec<u8>>> {
    recipes
        .iter()
        .map(|(target, recipe)| {
            let sources = recipe
                .operations
                .iter()
                .filter_map(|operation| match operation {
                    derivation::Operation::Copy(copy) => Some(copy.source.as_bytes().to_vec()),
                    derivation::Operation::Fill(_) => None,
                })
                .collect();
            (target.clone(), sources)
        })
        .collect()
}

/// True when a payload is located, or is a derivation target whose copy sources
/// are recursively available. Fills have no sources.
fn available(
    id: &[u8],
    located: &BTreeSet<Vec<u8>>,
    derivations: &BTreeMap<Vec<u8>, Vec<Vec<u8>>>,
    visiting: &mut BTreeSet<Vec<u8>>,
    memo: &mut BTreeSet<Vec<u8>>,
) -> bool {
    if located.contains(id) || memo.contains(id) {
        return true;
    }
    let Some(sources) = derivations.get(id) else {
        return false;
    };
    if !visiting.insert(id.to_vec()) {
        return false;
    }
    let complete = sources
        .iter()
        .all(|source| available(source, located, derivations, visiting, memo));
    visiting.remove(id);
    if complete {
        memo.insert(id.to_vec());
    }
    complete
}

/// One prepared owner's bytes, as a contiguous payload range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnerSource {
    pub payload: PayloadId,
    pub offset: u64,
    pub nbytes: u64,
}

/// One bounded initializer mapping of a prepared owner's allocation bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnerSegment {
    pub owner_offset: u64,
    pub payload: PayloadId,
    pub offset: u64,
    pub nbytes: u64,
}

/// One prepared weight view of a rank, with the declared packing and sources
/// needed to decide whether another deployment may share its bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedOwner {
    pub name: String,
    pub group: String,
    pub allocation: String,
    pub allocation_offset: u64,
    pub nbytes: u64,
    pub element_bytes: u32,
    pub shape: Vec<u64>,
    pub stride: Vec<u64>,
    pub packing: Option<wire::PackingParameters>,
    /// Declared statistic domain of any quantization the owner carries.
    pub statistics_domain: String,
    pub padding: Vec<wire::WeightPadding>,
    pub sources: Vec<wire::WeightSource>,
    pub source: Option<OwnerSource>,
    /// Sorted allocation-to-payload mappings for owners split across initializers.
    pub segments: Vec<OwnerSegment>,
}

/// Owners and payload slices of one rank of one prepared deployment.
#[derive(Clone)]
pub struct PreparedRank {
    pub rank: u32,
    pub owners: Vec<PreparedOwner>,
    pub payloads: BTreeMap<PayloadId, container::Slice>,
}

/// Byte relation accepted for one owner of a derived rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OwnerShare {
    /// Same payload bytes at the same offset.
    Identical { payload: PayloadId, offset: u64 },
    /// A contiguous copy from a canonical payload, verified byte-for-byte.
    Copy {
        source: PayloadId,
        source_offset: u64,
        contiguous: u64,
    },
    /// A rectangular 2- or 3-D copy, verified byte-for-byte. The recipe is
    /// carried in [`RankPlan::recipes`] for initializer-level emission.
    Rectangular { source: PayloadId },
    /// Multiple bounded copies from canonical payloads.
    Mapped,
    /// Incompatible packing, sources, or bytes: store for this deployment.
    Independent,
}

/// Mapped copy runs plus the target ranges that still need an exception payload.
pub struct PlannedTarget {
    pub operations: Vec<derivation::Operation>,
    pub gaps: Vec<(u64, u64)>,
    pub length: u64,
}

/// Per-rank owner relations produced by the combiner.
pub struct RankPlan {
    pub rank: u32,
    pub owners: BTreeMap<String, OwnerShare>,
    pub recipes: BTreeMap<String, derivation::Derivation>,
    pub targets: BTreeMap<PayloadId, PlannedTarget>,
}

/// Reads the prepared weight owners of one deployment with their payload bytes.
pub fn prepared_ranks(source: &Path, deployment: usize) -> Result<Vec<PreparedRank>> {
    let verified = Verified::open(source, Some(&[deployment]))?;
    let files = verified.payloads()?;
    let mut payloads = BTreeMap::new();
    for (id, slice) in files.iter() {
        payloads.insert(
            PayloadId::from_bytes(
                id.try_into()
                    .map_err(|_| invalid("payload id must be 32 bytes"))?,
            ),
            slice.clone(),
        );
    }
    let mut ranks = Vec::with_capacity(verified.ranks.len());
    for body in &verified.ranks {
        let mut mappings: BTreeMap<String, Vec<(u64, u64, PayloadId, u64)>> = BTreeMap::new();
        for initializer in &body.initializers {
            let Some(slice) = initializer.source.as_ref().and_then(|use_| use_.slice.as_ref()) else {
                continue;
            };
            let Ok(payload) = slice.payload_id.as_slice().try_into().map(PayloadId::from_bytes) else {
                continue;
            };
            let Some(end) = initializer.destination_offset.checked_add(slice.nbytes) else {
                continue;
            };
            mappings
                .entry(initializer.allocation.clone())
                .or_default()
                .push((
                    initializer.destination_offset,
                    end,
                    payload,
                    slice.offset,
                ));
        }
        for entries in mappings.values_mut() {
            entries.sort_by_key(|entry| entry.0);
        }
        let mut owners = Vec::new();
        if let Some(catalog) = &body.prepared_weights {
            for view in &catalog.views {
                let allocation = body
                    .allocations
                    .get(view.allocation as usize)
                    .map(|allocation| allocation.name.as_str())
                    .unwrap_or_default();
                let segments = mapped_owner_segments(view, mappings.get(allocation).map(Vec::as_slice));
                owners.push(PreparedOwner {
                    name: view.name.clone(),
                    group: view.group.clone(),
                    allocation: body
                        .allocations
                        .get(view.allocation as usize)
                        .map(|allocation| allocation.name.clone())
                        .unwrap_or_default(),
                    allocation_offset: view.offset,
                    nbytes: view.nbytes,
                    element_bytes: view.element_bytes,
                    shape: view.shape.clone(),
                    stride: view.stride.clone(),
                    packing: view.packing.clone(),
                    statistics_domain: view.statistics_domain.clone(),
                    padding: view.padding.clone(),
                    sources: view.sources.clone(),
                    source: owner_source(&segments, view.nbytes),
                    segments,
                });
            }
        }
        ranks.push(PreparedRank {
            rank: body.rank,
            owners,
            payloads: payloads.clone(),
        });
    }
    Ok(ranks)
}

fn owner_source(segments: &[OwnerSegment], nbytes: u64) -> Option<OwnerSource> {
    match segments {
        [segment] if segment.owner_offset == 0 && segment.nbytes == nbytes => Some(OwnerSource {
            payload: segment.payload,
            offset: segment.offset,
            nbytes,
        }),
        _ => None,
    }
}

fn mapped_owner_segments(
    view: &wire::PreparedWeightView,
    entries: Option<&[(u64, u64, PayloadId, u64)]>,
) -> Vec<OwnerSegment> {
    let Some(entries) = entries else {
        return Vec::new();
    };
    let Some(view_end) = view.offset.checked_add(view.nbytes) else {
        return Vec::new();
    };
    let start = entries.partition_point(|entry| entry.1 <= view.offset);
    let mut segments = Vec::new();
    for (destination, end, payload, source_offset) in &entries[start..] {
        if *destination >= view_end {
            break;
        }
        let overlap_start = (*destination).max(view.offset);
        let overlap_end = (*end).min(view_end);
        if overlap_start >= overlap_end {
            continue;
        }
        let Some(owner_offset) = overlap_start.checked_sub(view.offset) else {
            return Vec::new();
        };
        let Some(offset) = source_offset.checked_add(overlap_start - *destination) else {
            return Vec::new();
        };
        let Some(nbytes) = overlap_end.checked_sub(overlap_start) else {
            return Vec::new();
        };
        segments.push(OwnerSegment {
            owner_offset,
            payload: *payload,
            offset,
            nbytes,
        });
    }
    for pair in segments.windows(2) {
        if pair[0].owner_offset.checked_add(pair[0].nbytes).is_none_or(|end| end > pair[1].owner_offset) {
            return Vec::new();
        }
    }
    segments
}

/// Plans one derived rank against its canonical rank.
///
/// Sharing requires identical payload bytes, or agreement on the declared
/// packing and prepared sources followed by an exact candidate byte comparison.
/// Anything else stays independent; no numerical transform is attempted.
pub fn plan_rank(
    canonical: &PreparedRank,
    derived: &PreparedRank,
    limits: &derivation::Limits,
) -> Result<RankPlan> {
    plan_rank_against(std::slice::from_ref(canonical), derived, limits)
}

const MAX_PIECE_AXES: usize = 6;

fn piece_path_eligible(canonical: &PreparedOwner, derived: &PreparedOwner) -> bool {
    if canonical.sources.is_empty() || derived.sources.is_empty() {
        return false;
    }
    let mut matched = BTreeSet::new();
    for target in &derived.sources {
        if target.pieces.is_empty() || !matched.insert(target.tensor.as_str()) {
            return false;
        }
        let Some(base) = canonical.sources.iter().find(|source| {
            source.tensor == target.tensor
                && source.piece_encoding == target.piece_encoding
                && !source.pieces.is_empty()
        }) else {
            return false;
        };
        if base.concatenation_axis != target.concatenation_axis {
            return false;
        }
    }
    true
}

/// Packing identity needed to compare encoded piece coordinates.
///
/// `row_rotation` is destination geometry already expressed by each piece.
/// Encoded pieces with the same deterministic encoding can therefore be
/// compared across differing rotation even when the retained source layout
/// differs; the candidate still has to match the target bytes exactly.
fn piece_packing_compatible(
    canonical: &Option<wire::PackingParameters>,
    derived: &Option<wire::PackingParameters>,
) -> bool {
    match (canonical, derived) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.format == b.format
                && a.block_size == b.block_size
                && a.group_size == b.group_size
                && a.scale_atom_rows == b.scale_atom_rows
                && a.tile_order == b.tile_order
                && a.locality_tile == b.locality_tile
        }
        _ => false,
    }
}

fn piece_compatible(canonical: &PreparedOwner, derived: &PreparedOwner) -> bool {
    if !piece_packing_compatible(&canonical.packing, &derived.packing)
        || canonical.element_bytes != derived.element_bytes
        || !piece_path_eligible(canonical, derived)
    {
        return false;
    }
    // An absent declared domain stays compatible; two different declared domains do not.
    !(!canonical.statistics_domain.is_empty()
        && !derived.statistics_domain.is_empty()
        && canonical.statistics_domain != derived.statistics_domain)
}

#[derive(Clone, Copy)]
struct PieceRun {
    source: u64,
    destination: u64,
    nbytes: u64,
}

fn piece_runs(pieces: &[wire::WeightPiece], limits: &derivation::Limits) -> Option<Vec<PieceRun>> {
    let mut runs = Vec::new();
    for piece in pieces {
        let axes = piece.extents.len();
        if axes > MAX_PIECE_AXES
            || piece.source_strides.len() != axes
            || piece.destination_strides.len() != axes
            || piece.contiguous_bytes == 0
        {
            return None;
        }
        let mut count = 1u64;
        for extent in &piece.extents {
            if *extent == 0 {
                return None;
            }
            count = count.checked_mul(*extent)?;
        }
        if count > limits.max_runs as u64 {
            return None;
        }
        for combination in 0..count {
            let mut source = piece.source_offset;
            let mut destination = piece.destination_offset;
            let mut index = combination;
            for axis in 0..axes {
                let extent = piece.extents[axis];
                let coordinate = index % extent;
                index /= extent;
                source = source.checked_add(
                    coordinate.checked_mul(piece.source_strides[axis])?,
                )?;
                destination = destination.checked_add(
                    coordinate.checked_mul(piece.destination_strides[axis])?,
                )?;
            }
            let end = source.checked_add(piece.contiguous_bytes)?;
            if destination.checked_add(piece.contiguous_bytes).is_none() {
                return None;
            }
            let _ = end;
            runs.push(PieceRun {
                source,
                destination,
                nbytes: piece.contiguous_bytes,
            });
            if runs.len() > limits.max_runs {
                return None;
            }
        }
    }
    runs.sort_by_key(|run| run.source);
    for pair in runs.windows(2) {
        if pair[0].source.checked_add(pair[0].nbytes)? > pair[1].source {
            return None;
        }
    }
    Some(runs)
}

fn piece_source_pair<'a>(
    base_owner: &'a PreparedOwner,
    target: &'a wire::WeightSource,
) -> Option<&'a wire::WeightSource> {
    let mut found = None;
    for source in &base_owner.sources {
        if source.tensor == target.tensor && source.piece_encoding == target.piece_encoding {
            if found.is_some() {
                return None;
            }
            found = Some(source);
        }
    }
    found.filter(|source| !source.pieces.is_empty())
}

fn verify_candidate_operations(
    local: &BTreeMap<PayloadId, Vec<derivation::Operation>>,
    base_rank: &PreparedRank,
    derived: &PreparedRank,
) -> Option<()> {
    for (target, operations) in local {
        let target_payload = derived.payloads.get(target)?;
        for operation in operations {
            let derivation::Operation::Copy(copy) = operation else {
                return None;
            };
            let source_payload = base_rank.payloads.get(&copy.source)?;
            let source_slice = container::Slice::new(
                source_payload.file().clone(),
                source_payload.offset().checked_add(copy.source_offset)?,
                copy.contiguous,
            )
            .ok()?;
            let target_slice = container::Slice::new(
                target_payload.file().clone(),
                target_payload.offset().checked_add(copy.destination_offset)?,
                copy.contiguous,
            )
            .ok()?;
            if !same_bytes(&source_slice, &target_slice).ok()? {
                return None;
            }
        }
    }
    Some(())
}

fn piece_owner_relation(
    base_rank: &PreparedRank,
    base_owner: &PreparedOwner,
    derived: &PreparedRank,
    owner: &PreparedOwner,
    target_segments: &[OwnerSegment],
    limits: &derivation::Limits,
) -> Option<BTreeMap<PayloadId, Vec<derivation::Operation>>> {
    if !piece_compatible(base_owner, owner) {
        return None;
    }
    let base_segments = owner_segments(base_owner);
    if base_segments.is_empty() || target_segments.is_empty() {
        return None;
    }
    let mut local: BTreeMap<PayloadId, Vec<derivation::Operation>> = BTreeMap::new();
    let mut covered = vec![0u64; target_segments.len()];
    let mut operation_count = 0usize;
    for target_source in &owner.sources {
        let base_source = piece_source_pair(base_owner, target_source)?;
        let target_runs = piece_runs(&target_source.pieces, limits)?;
        let base_runs = piece_runs(&base_source.pieces, limits)?;
        let mut base_index = 0usize;
        for target_run in target_runs {
            let target_end = target_run.source.checked_add(target_run.nbytes)?;
            while base_index < base_runs.len()
                && base_runs[base_index]
                    .source
                    .checked_add(base_runs[base_index].nbytes)? <= target_run.source
            {
                base_index += 1;
            }
            let mut index = base_index;
            let mut cursor = target_run.source;
            while index < base_runs.len() && cursor < target_end {
                let base_run = base_runs[index];
                let base_end = base_run.source.checked_add(base_run.nbytes)?;
                if base_end <= cursor {
                    index += 1;
                    continue;
                }
                let from = cursor.max(base_run.source);
                let to = target_end.min(base_end);
                if from >= to {
                    index += 1;
                    continue;
                }
                if from != cursor {
                    return None;
                }
                let destination_from =
                    target_run.destination.checked_add(from - target_run.source)?;
                let destination_to = destination_from.checked_add(to - from)?;
                let base_destination_from =
                    base_run.destination.checked_add(from - base_run.source)?;
                let mut target_start = target_segments.partition_point(|segment| {
                    segment
                        .owner_offset
                        .checked_add(segment.nbytes)
                        .is_none_or(|end| end <= destination_from)
                });
                while target_start < target_segments.len()
                    && target_segments[target_start].owner_offset < destination_to
                {
                    let target_segment = &target_segments[target_start];
                    let target_segment_end = target_segment
                        .owner_offset
                        .checked_add(target_segment.nbytes)?;
                    let target_from = destination_from.max(target_segment.owner_offset);
                    let target_to = destination_to.min(target_segment_end);
                    if target_from < target_to {
                        let source_from = base_destination_from
                            .checked_add(target_from - destination_from)?;
                        let source_to = source_from.checked_add(target_to - target_from)?;
                        let source_start_index = base_segments.partition_point(|segment| {
                            segment
                                .owner_offset
                                .checked_add(segment.nbytes)
                                .is_none_or(|end| end <= source_from)
                        });
                        for source_segment in &base_segments[source_start_index..] {
                            let source_segment_end = source_segment
                                .owner_offset
                                .checked_add(source_segment.nbytes)?;
                            if source_segment.owner_offset >= source_to {
                                break;
                            }
                            let intersection_from = source_from.max(source_segment.owner_offset);
                            let intersection_to = source_to.min(source_segment_end);
                            if intersection_from >= intersection_to {
                                continue;
                            }
                            let destination_offset = target_segment
                                .offset
                                .checked_add(target_from - target_segment.owner_offset)?
                                .checked_add(intersection_from - source_from)?;
                            let source_offset = source_segment
                                .offset
                                .checked_add(intersection_from - source_segment.owner_offset)?;
                            local
                                .entry(target_segment.payload)
                                .or_default()
                                .push(derivation::Operation::Copy(derivation::Copy {
                                    source: source_segment.payload,
                                    source_offset,
                                    destination_offset,
                                    contiguous: intersection_to - intersection_from,
                                    outer: Vec::new(),
                                }));
                            operation_count += 1;
                            if operation_count > limits.max_operations {
                                return None;
                            }
                            covered[target_start] = covered[target_start]
                                .checked_add(intersection_to - intersection_from)?;
                        }
                    }
                    target_start += 1;
                }
                cursor = to;
                index += 1;
            }
            if cursor != target_end {
                return None;
            }
        }
    }
    if covered.iter().all(|covered| *covered == 0) {
        return None;
    }
    verify_candidate_operations(&local, base_rank, derived)?;
    for operations in local.values_mut() {
        *operations = coalesce_copies(std::mem::take(operations));
    }
    Some(local)
}

fn map_owner_relation(
    base_rank: &PreparedRank,
    base_owner: &PreparedOwner,
    derived: &PreparedRank,
    owner: &PreparedOwner,
    target_segments: &[OwnerSegment],
    limits: &derivation::Limits,
) -> Option<BTreeMap<PayloadId, Vec<derivation::Operation>>> {
    let base_has_pieces = base_owner
        .sources
        .iter()
        .any(|source| !source.pieces.is_empty());
    let target_has_pieces = owner
        .sources
        .iter()
        .any(|source| !source.pieces.is_empty());
    if base_has_pieces || target_has_pieces {
        // Nonempty provenance means the stored owner bytes are not necessarily
        // raw checkpoint bytes. Never use the legacy geometry relation for a
        // transformed layout; absent encoding fails closed until approved.
        if !piece_path_eligible(base_owner, owner) {
            return None;
        }
        return piece_owner_relation(base_rank, base_owner, derived, owner, target_segments, limits);
    }
    let base_segments = owner_segments(base_owner);
    if base_segments.is_empty() || target_segments.is_empty() {
        return None;
    }
    let base_payload = base_owner
        .source
        .map(|source| source.payload)
        .or_else(|| base_segments.first().map(|segment| segment.payload))?;
    let operations = owner_operations(base_owner, owner, base_payload, 0, owner.nbytes)?;
    let relation = derivation::Derivation {
        target: target_segments[0].payload,
        length: owner.nbytes,
        operations,
    };
    let runs = derivation::destination_runs(&relation, limits).ok()?;
    let mut local: BTreeMap<PayloadId, Vec<derivation::Operation>> = BTreeMap::new();
    let mut covered = vec![0u64; target_segments.len()];
    for run in runs {
        let Some((_, source_start)) = run.source else {
            return None;
        };
        let run_end = run.destination.checked_add(run.length)?;
        let target_start = target_segments.partition_point(|segment| {
            segment.owner_offset.checked_add(segment.nbytes).is_none_or(|end| end <= run.destination)
        });
        for (target_index, target_segment) in target_segments.iter().enumerate().skip(target_start) {
            let Some(target_end) = target_segment.owner_offset.checked_add(target_segment.nbytes) else {
                return None;
            };
            if target_segment.owner_offset >= run_end {
                break;
            }
            let destination_from = run.destination.max(target_segment.owner_offset);
            let destination_to = run_end.min(target_end);
            if destination_from >= destination_to {
                continue;
            }
            let source_from = source_start.checked_add(destination_from - run.destination)?;
            let source_to = source_from.checked_add(destination_to - destination_from)?;
            let source_start_index = base_segments.partition_point(|segment| {
                segment.owner_offset.checked_add(segment.nbytes).is_none_or(|end| end <= source_from)
            });
            for source_segment in &base_segments[source_start_index..] {
                let Some(source_segment_end) = source_segment.owner_offset.checked_add(source_segment.nbytes) else {
                    return None;
                };
                if source_segment.owner_offset >= source_to {
                    break;
                }
                let intersection_from = source_from.max(source_segment.owner_offset);
                let intersection_to = source_to.min(source_segment_end);
                if intersection_from >= intersection_to {
                    continue;
                }
                let destination_offset = target_segment
                    .offset
                    .checked_add(destination_from - target_segment.owner_offset)?
                    .checked_add(intersection_from - source_from)?;
                let source_offset = source_segment
                    .offset
                    .checked_add(intersection_from - source_segment.owner_offset)?;
                local
                    .entry(target_segment.payload)
                    .or_default()
                    .push(derivation::Operation::Copy(derivation::Copy {
                        source: source_segment.payload,
                        source_offset,
                        destination_offset,
                        contiguous: intersection_to - intersection_from,
                        outer: Vec::new(),
                    }));
                covered[target_index] = covered[target_index]
                    .checked_add(intersection_to - intersection_from)?;
            }
        }
    }
    for (segment, covered) in target_segments.iter().zip(&covered) {
        if *covered != segment.nbytes {
            return None;
        }
    }
    verify_candidate_operations(&local, base_rank, derived)?;
    for operations in local.values_mut() {
        *operations = coalesce_copies(std::mem::take(operations));
    }
    Some(local)
}

/// Plans one derived rank against every canonical rank.
///
/// TP1 has one rank holding all rows, while TP2 rank 1 holds the second half;
/// matching is by logical prepared-source intervals, never by rank number.
pub fn plan_rank_against(
    canonical: &[PreparedRank],
    derived: &PreparedRank,
    limits: &derivation::Limits,
) -> Result<RankPlan> {
    let mut owners = BTreeMap::new();
    let mut recipes = BTreeMap::new();
    let mut target_plans: BTreeMap<PayloadId, PlannedTarget> = BTreeMap::new();
    for owner in &derived.owners {
        let Some(target) = owner.source else {
            continue;
        };
        let target_len = derived.payloads.get(&target.payload).map(|slice| slice.len()).unwrap_or(0);
        if target_len != owner.nbytes
            || derived
                .owners
                .iter()
                .filter(|other| other.name != owner.name)
                .any(|other| other.segments.iter().any(|segment| segment.payload == target.payload))
        {
            owners.insert(owner.name.clone(), OwnerShare::Independent);
            continue;
        }
        let mut accepted = None;
        'candidate: for base_rank in canonical {
            for base_owner in base_rank
                .owners
                .iter()
                .filter(|candidate| candidate.name == owner.name)
            {
                let Some(base) = base_owner.source else {
                    continue;
                };
                if base.payload == target.payload
                    && base.offset == target.offset
                    && base.nbytes == target.nbytes
                {
                    accepted = Some((base, None));
                    break 'candidate;
                }
                let Some(recipe) = owner_recipe(base_owner, owner, base, target.nbytes) else {
                    continue;
                };
                let Ok(target_slice) = slice_of(derived, target) else {
                    continue;
                };
                let Ok(source_slice) = payload_slice(base_rank, base.payload) else {
                    continue;
                };
                let sources = BTreeMap::from([(base.payload, source_slice)]);
                if derivation::verify_output(&target_slice, &recipe, &sources, limits).is_ok() {
                    accepted = Some((base, Some(recipe)));
                    break 'candidate;
                }
            }
        }
        let Some((base, recipe)) = accepted else {
            owners.insert(owner.name.clone(), OwnerShare::Independent);
            continue;
        };
        let Some(recipe) = recipe else {
            owners.insert(
                owner.name.clone(),
                OwnerShare::Identical {
                    payload: base.payload,
                    offset: base.offset,
                },
            );
            continue;
        };
        if derivation_uses_target(&recipe) {
            owners.insert(owner.name.clone(), OwnerShare::Independent);
            continue;
        }
        let rectangular = matches!(
            recipe.operations.first(),
            Some(derivation::Operation::Copy(copy)) if !copy.outer.is_empty()
        );
        if rectangular {
            owners.insert(
                owner.name.clone(),
                OwnerShare::Rectangular {
                    source: base.payload,
                },
            );
        } else {
            owners.insert(
                owner.name.clone(),
                OwnerShare::Copy {
                    source: base.payload,
                    source_offset: candidate_source_offset(&recipe),
                    contiguous: target.nbytes,
                },
            );
        }
        recipes.insert(owner.name.clone(), recipe);
    }

    // Build one owner-relative byte relation, then translate both source and
    // destination coordinates through their initializer segment maps. This keeps
    // TP shard geometry and allocation offsets out of compatibility decisions.
    let claimed_targets: BTreeSet<PayloadId> = recipes.values().map(|recipe| recipe.target).collect();
    let mut target_ops: BTreeMap<PayloadId, Vec<derivation::Operation>> = BTreeMap::new();
    let mut mapped_owners: BTreeSet<String> = BTreeSet::new();
    for owner in &derived.owners {
        if owners
            .get(&owner.name)
            .is_some_and(|share| !matches!(share, OwnerShare::Independent))
        {
            continue;
        }
        let target_segments = owner_segments(owner);
        if target_segments.is_empty() {
            continue;
        }
        let targets: BTreeSet<PayloadId> =
            target_segments.iter().map(|segment| segment.payload).collect();
        if targets.iter().any(|target| claimed_targets.contains(target)) {
            continue;
        }
        let Some(local_ops) = canonical.iter().find_map(|rank| {
            rank.owners
                .iter()
                .filter(|candidate| candidate.name == owner.name)
                .find_map(|candidate| {
                    map_owner_relation(
                        rank,
                        candidate,
                        derived,
                        owner,
                        &target_segments,
                        limits,
                    )
                })
        }) else {
            continue;
        };
        for (target, operations) in local_ops {
            target_ops.entry(target).or_default().extend(operations);
        }
        mapped_owners.insert(owner.name.clone());
    }
    for (target, operations) in target_ops {
        let Some(target_slice) = derived.payloads.get(&target) else {
            continue;
        };
        let recipe = derivation::Derivation {
            target,
            length: target_slice.len(),
            operations,
        };
        if derivation_uses_target(&recipe) {
            continue;
        }
        if recipe.operations.len() > limits.max_operations
            || recipe.operations.iter().any(|operation| {
                matches!(operation, derivation::Operation::Copy(copy) if !copy.outer.is_empty())
            })
        {
            continue;
        }
        let mut source_map = BTreeMap::new();
        let mut sources_ok = true;
        for operation in &recipe.operations {
            let derivation::Operation::Copy(copy) = operation else {
                continue;
            };
            if !source_map.contains_key(&copy.source) {
                let Some(slice) = canonical
                    .iter()
                    .find_map(|rank| rank.payloads.get(&copy.source).cloned())
                else {
                    sources_ok = false;
                    break;
                };
                source_map.insert(copy.source, slice);
            }
        }
        if !sources_ok {
            continue;
        }
        let mut covered: Vec<(u64, u64)> = recipe
            .operations
            .iter()
            .filter_map(|operation| match operation {
                derivation::Operation::Copy(copy) => Some((
                    copy.destination_offset,
                    copy.destination_offset.saturating_add(copy.contiguous),
                )),
                derivation::Operation::Fill(fill) => Some((
                    fill.destination_offset,
                    fill.destination_offset.saturating_add(fill.length),
                )),
            })
            .collect();
        covered.sort_unstable();
        if covered.iter().any(|(start, end)| start >= end || *end > target_slice.len()) {
            continue;
        }
        if covered.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            continue;
        }
        let mut gaps = Vec::new();
        let mut cursor = 0u64;
        for (start, end) in covered {
            if start > cursor {
                gaps.push((cursor, start - cursor));
            }
            cursor = end;
        }
        if cursor < target_slice.len() {
            gaps.push((cursor, target_slice.len() - cursor));
        }
        if gaps.is_empty() {
            if derivation::verify_output(target_slice, &recipe, &source_map, limits).is_ok() {
                recipes.insert(format!("{target}"), recipe);
            }
        } else {
            target_plans.insert(target, PlannedTarget {
                operations: recipe.operations,
                gaps,
                length: target_slice.len(),
            });
        }
    }
    for owner in mapped_owners {
        let all_targets = owner_targets(&derived.owners, &owner);
        if !all_targets.is_empty()
            && all_targets
                .iter()
                .all(|target| recipes.values().any(|recipe| recipe.target == *target))
        {
            owners.insert(owner, OwnerShare::Mapped);
        }
    }
    Ok(RankPlan {
        rank: derived.rank,
        owners,
        recipes,
        targets: target_plans,
    })
}

fn owner_segments(owner: &PreparedOwner) -> Vec<OwnerSegment> {
    if !owner.segments.is_empty() {
        return owner.segments.clone();
    }
    owner
        .source
        .map(|source| {
            vec![OwnerSegment {
                owner_offset: 0,
                payload: source.payload,
                offset: source.offset,
                nbytes: source.nbytes,
            }]
        })
        .unwrap_or_default()
}

fn coalesce_copies(operations: Vec<derivation::Operation>) -> Vec<derivation::Operation> {
    let mut result: Vec<derivation::Operation> = Vec::new();
    for operation in operations {
        let derivation::Operation::Copy(copy) = operation else {
            result.push(operation);
            continue;
        };
        if let Some(derivation::Operation::Copy(previous)) = result.last_mut()
            && previous.source == copy.source
            && previous.source_offset.checked_add(previous.contiguous) == Some(copy.source_offset)
            && previous.destination_offset.checked_add(previous.contiguous) == Some(copy.destination_offset)
            && copy.outer.is_empty()
            && previous.outer.is_empty()
        {
            previous.contiguous = previous.contiguous.saturating_add(copy.contiguous);
            continue;
        }
        result.push(derivation::Operation::Copy(copy));
    }
    result
}

fn validate_planned_derivations(recipes: &[derivation::Derivation]) -> Result<()> {
    derivation::order(recipes, &derivation::Limits::default())
        .map(|_| ())
        .map_err(|error| invalid(format!("planned derivation set: {error}; {}", planned_derivation_context(recipes))))
}

fn planned_derivation_context(recipes: &[derivation::Derivation]) -> String {
    let mut edges = 0usize;
    let mut self_edges = 0usize;
    let mut sample = Vec::new();
    for recipe in recipes {
        for operation in &recipe.operations {
            let derivation::Operation::Copy(copy) = operation else {
                continue;
            };
            edges += 1;
            if copy.source == recipe.target {
                self_edges += 1;
            }
            if sample.len() < 8 {
                sample.push(format!("{}->{}", recipe.target, copy.source));
            }
        }
    }
    format!(
        "edges={edges}, self_edges={self_edges}, first={}",
        sample.join(",")
    )
}

fn derivation_uses_target(recipe: &derivation::Derivation) -> bool {
    recipe.operations.iter().any(|operation| {
        matches!(
            operation,
            derivation::Operation::Copy(copy) if copy.source == recipe.target
        )
    })
}

fn owner_targets(owners: &[PreparedOwner], name: &str) -> BTreeSet<PayloadId> {
    owners
        .iter()
        .filter(|owner| owner.name == name)
        .flat_map(|owner| owner_segments(owner).into_iter().map(|segment| segment.payload))
        .collect()
}

fn payload_slice(rank: &PreparedRank, id: PayloadId) -> Result<container::Slice> {
    rank.payloads
        .get(&id)
        .cloned()
        .ok_or_else(|| invalid("prepared owner names an absent payload"))
}

/// Plans every rank of a derived deployment against all canonical ranks.
pub fn plan_deployment(
    canonical: &[PreparedRank],
    derived: &[PreparedRank],
    limits: &derivation::Limits,
) -> Result<Vec<RankPlan>> {
    let mut plans = Vec::with_capacity(derived.len());
    for rank in derived {
        plans.push(plan_rank_against(canonical, rank, limits)?);
    }
    Ok(plans)
}

fn finish_plan_exceptions(
    plan: &mut RankPlan,
    derived: &PreparedRank,
    canonical: &[PreparedRank],
    source: &CombineSource,
    deployment_id: &str,
    used_paths: &mut BTreeSet<String>,
    total_operations: &mut usize,
    total_runs: &mut usize,
) -> Result<Vec<(PayloadId, crate::package::export::PublishPayload)>> {
    use crate::package::export::{PublishPayload, pin_file};
    let mut artifacts = Vec::new();
    let targets = std::mem::take(&mut plan.targets);
    for (target, planned) in targets {
        let Some(target_slice) = derived.payloads.get(&target) else {
            continue;
        };
        if target_slice.len() != planned.length {
            continue;
        }
        let limits = derivation::Limits::default();
        let target_operations = planned.operations.len().saturating_add(planned.gaps.len());
        if target_operations > limits.max_operations
            || planned.gaps.len() > limits.max_runs
            || total_operations.saturating_add(target_operations) > limits.max_operations
            || total_runs.saturating_add(planned.gaps.len()) > limits.max_runs
        {
            continue;
        }
        *total_operations = total_operations.saturating_add(target_operations);
        *total_runs = total_runs.saturating_add(planned.gaps.len());
        let raw = unique_staging(&source.scratch.join(format!("exception-{}", plan.rank)), "payload")?;
        let guard = TempFile::new(&raw);
        let mut exception = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&raw)
            .map_err(|e| invalid(format!("create exception payload: {e}")))?;
        let mut hasher = DigestContext::new(&SHA256);
        let mut buffer = vec![0u8; 1 << 20];
        let mut exception_offset = 0u64;
        let mut exception_ranges = Vec::new();
        let mut operations = planned.operations;
        for (destination, length) in &planned.gaps {
            let mut read = 0u64;
            while read < *length {
                let count = (*length - read).min(buffer.len() as u64) as usize;
                let offset = destination
                    .checked_add(read)
                    .ok_or_else(|| invalid("exception source offset overflow"))?;
                target_slice.read_at(offset, &mut buffer[..count])?;
                exception.write_all(&buffer[..count]).map_err(|e| invalid(format!("write exception payload: {e}")))?;
                hasher.update(&buffer[..count]);
                read += count as u64;
            }
            exception_ranges.push((exception_offset, *destination, *length));
            exception_offset = exception_offset
                .checked_add(*length)
                .ok_or_else(|| invalid("exception payload length overflow"))?;
        }
        exception.sync_all().map_err(|e| invalid(format!("sync exception payload: {e}")))?;
        drop(exception);
        let exception_id = PayloadId::from_digest(hasher.finish());
        if exception_id == target {
            continue;
        }
        let exception_slice = container::Slice::new(
            std::sync::Arc::new(File::open(&raw).map_err(|e| invalid(format!("open exception payload: {e}")))?),
            0,
            exception_offset,
        )
        .map_err(|e| invalid(e.to_string()))?;
        for (source_offset, destination_offset, contiguous) in exception_ranges {
            operations.push(derivation::Operation::Copy(derivation::Copy {
                source: exception_id,
                source_offset,
                destination_offset,
                contiguous,
                outer: Vec::new(),
            }));
        }
        let recipe = derivation::Derivation {
            target,
            length: planned.length,
            operations,
        };
        let mut sources = BTreeMap::new();
        for operation in &recipe.operations {
            let derivation::Operation::Copy(copy) = operation else {
                continue;
            };
            if copy.source == exception_id {
                sources.insert(exception_id, exception_slice.clone());
            } else if !sources.contains_key(&copy.source) {
                let Some(slice) = canonical
                    .iter()
                    .find_map(|rank| rank.payloads.get(&copy.source).cloned())
                else {
                    sources.clear();
                    break;
                };
                sources.insert(copy.source, slice);
            }
        }
        if sources.is_empty()
            || derivation::validate(&recipe, &limits).is_err()
            || derivation::verify_output(target_slice, &recipe, &sources, &limits).is_err()
        {
            continue;
        }
        let hint = semantic_relative(source, target.as_bytes())
            .and_then(|relative| {
                relative
                    .rsplit_once('/')
                    .map(|(_, name)| name.to_owned())
                    .or_else(|| Some(relative))
            })
            .unwrap_or_else(|| format!("exception-{}", artifacts.len()));
        let stem = hint.strip_suffix(".safetensors").unwrap_or(&hint);
        let relative = unique_relative(
            used_paths,
            format!("weights/exceptions/{deployment_id}/rank-{}/{stem}-exceptions.safetensors", plan.rank),
            deployment_id,
            plan.rank,
        );
        let artifact = PublishPayload {
            role: wire::PayloadRole::Weights,
            relative,
            source: pin_file(&raw, exception_offset).map_err(|e| invalid(e.to_string()))?,
            id: exception_id,
            tensor: None,
        };
        guard.keep();
        plan.recipes.insert(format!("{target}"), recipe);
        artifacts.push((target, artifact));
    }
    Ok(artifacts)
}

fn slice_of(rank: &PreparedRank, source: OwnerSource) -> Result<container::Slice> {
    let payload = payload_slice(rank, source.payload)?;
    container::Slice::new(
        payload.file().clone(),
        payload
            .offset()
            .checked_add(source.offset)
            .ok_or_else(|| invalid("prepared owner offset overflow"))?,
        source.nbytes,
    )
}

fn candidate_source_offset(candidate: &derivation::Derivation) -> u64 {
    match candidate.operations.first() {
        Some(derivation::Operation::Copy(copy)) => copy.source_offset,
        _ => 0,
    }
}

/// Builds a 1-3D copy candidate from declared logical ranges and layout geometry.
///
/// Only an exact rectangular subrange of one source tensor is proposed, with the
/// same packing, shape pieces, strides and no destination padding. The caller
/// still proves the candidate against real bytes before sharing.
fn owner_recipe(
    canonical: &PreparedOwner,
    derived: &PreparedOwner,
    base: OwnerSource,
    target_nbytes: u64,
) -> Option<derivation::Derivation> {
    let target = derived.source?;
    let operations = owner_operations(canonical, derived, base.payload, base.offset, target_nbytes)?;
    Some(derivation::Derivation {
        target: target.payload,
        length: target_nbytes,
        operations,
    })
}

fn owner_operations(
    canonical: &PreparedOwner,
    derived: &PreparedOwner,
    source_payload: PayloadId,
    source_base_offset: u64,
    target_nbytes: u64,
) -> Option<Vec<derivation::Operation>> {
    if canonical.packing != derived.packing
        // A declared domain difference (shard-local vs tensor) rules out an affine
        // candidate at identical geometry. An absent domain stays compatible, so
        // older catalogs remain derivable; byte identity is unaffected because the
        // `Identical` payload share is decided before this check.
        || (!canonical.statistics_domain.is_empty()
            && !derived.statistics_domain.is_empty()
            && canonical.statistics_domain != derived.statistics_domain)
        || canonical.element_bytes != derived.element_bytes
        || !derived.padding.is_empty()
        || canonical.sources.len() != 1
        || derived.sources.len() != 1
    {
        return None;
    }
    let source = &canonical.sources[0];
    let target = &derived.sources[0];
    if source.tensor != target.tensor
        || source.concatenation_axis != target.concatenation_axis
        || source.ranges.len() != target.ranges.len()
        || source.ranges.is_empty()
        || source.ranges.len() > 3
    {
        return None;
    }
    let axes = source.ranges.len();
    if canonical.shape.len() != axes
        || derived.shape.len() != axes
        || canonical.stride.len() != axes
    {
        return None;
    }
    let element = u64::from(derived.element_bytes);
    let mut delta = [0u64; 3];
    for index in 0..axes {
        let (base_begin, base_end) = (source.ranges[index].begin, source.ranges[index].end);
        let (begin, end) = (target.ranges[index].begin, target.ranges[index].end);
        if begin < base_begin || end > base_end || begin >= end {
            return None;
        }
        if canonical.shape[index] != base_end - base_begin || derived.shape[index] != end - begin {
            return None;
        }
        delta[index] = begin - base_begin;
    }
    let mut source_offset = source_base_offset;
    for index in 0..axes {
        source_offset = source_offset.checked_add(
            delta[index]
                .checked_mul(canonical.stride[index])?
                .checked_mul(element)?,
        )?;
    }
    let contiguous = derived.shape[axes - 1].checked_mul(element)?;
    let mut destination_stride = contiguous;
    let mut outer = Vec::new();
    for index in (0..axes - 1).rev() {
        outer.push(derivation::Axis {
            count: derived.shape[index],
            source_stride: canonical.stride[index].checked_mul(element)?,
            destination_stride,
        });
        destination_stride = destination_stride.checked_mul(derived.shape[index])?;
    }
    if destination_stride != target_nbytes {
        return None;
    }
    Some(vec![derivation::Operation::Copy(derivation::Copy {
        source: source_payload,
        source_offset,
        destination_offset: 0,
        contiguous,
        outer,
    })])
}

/// Adds deployment-scoped payload descriptors for Engram planes and every
/// derivation-copy source reachable from the deployment payloads, then re-signs
/// the affected envelope bodies.
///
/// `publish` installs deployment-scoped payloads once for the canonical
/// deployment. Every other deployment references those payload identities and
/// must authenticate the descriptors in its own signed body; a descriptor is
/// therefore added to each deployment that consumes it, not inferred from the
/// package-wide catalog or another deployment's signature.
fn inject_dependency_descriptors(
    staging: &Path,
    key: &ring::signature::Ed25519KeyPair,
    package_id: [u8; 32],
    public_key: [u8; 32],
    recipes: &[derivation::Derivation],
) -> Result<()> {
    use crate::package::protection::EnvelopeBuilder;
    let metadata = Metadata::open(staging)?;
    let envelope = protection::envelope(&metadata.envelope)?;
    let mut package = wire::PackageBody::decode(envelope.package_body.as_slice())
        .map_err(|e| invalid(format!("invalid package body: {e}")))?;
    let mut catalog = schema::validate_package(&package)?;
    // Source descriptors may live in any deployment or rank body. Build the complete published
    // catalog first so dependency injection cannot depend on body ordering.
    for signed in &envelope.deployments {
        let body = wire::DeploymentBody::decode(signed.body.as_slice())
            .map_err(|e| invalid(format!("invalid deployment body: {e}")))?;
        catalog.insert_all(&body.payloads)?;
    }
    for rank in &envelope.ranks {
        let body = crate::package::schema::wire::RankBody::decode(rank.body.as_slice())
            .map_err(|e| invalid(format!("invalid rank body: {e}")))?;
        catalog.insert_all(&body.payloads)?;
    }
    let mut bodies: Vec<(u32, Vec<u8>)> = Vec::new();
    for (position, signed) in envelope.deployments.iter().enumerate() {
        let mut body = wire::DeploymentBody::decode(signed.body.as_slice())
            .map_err(|e| invalid(format!("invalid deployment body: {e}")))?;
        catalog.insert_all(&body.payloads)?;
        if let Some(engram) = &body.engram {
            for id in engram_payload_ids(engram) {
                if body.payloads.iter().any(|descriptor| descriptor.id == id) {
                    continue;
                }
                let nbytes = catalog.length(&id)?;
                body.payloads.push(wire::PayloadDescriptor {
                    id: id.clone(),
                    nbytes,
                });
            }
        }
        if !recipes.is_empty() {
            let mut by_target: BTreeMap<PayloadId, &derivation::Derivation> = BTreeMap::new();
            for recipe in recipes {
                by_target.insert(recipe.target, recipe);
            }
            let mut targets: BTreeSet<Vec<u8>> = body.payloads.iter().map(|d| d.id.clone()).collect();
            for rank in envelope.ranks.iter().filter(|rank| rank.deployment == position as u32) {
                let rank_body = crate::package::schema::wire::RankBody::decode(rank.body.as_slice())
                    .map_err(|e| invalid(format!("invalid rank body: {e}")))?;
                targets.extend(rank_body.payloads.iter().map(|d| d.id.clone()));
            }
            let mut pending: Vec<Vec<u8>> = targets.into_iter().collect();
            let mut present: BTreeSet<Vec<u8>> = body.payloads.iter().map(|d| d.id.clone()).collect();
            while let Some(id) = pending.pop() {
                let target = PayloadId::from_bytes(id.as_slice().try_into().map_err(|_| invalid("payload id must be 32 bytes"))?);
                let Some(recipe) = by_target.get(&target) else { continue };
                for operation in &recipe.operations {
                    let derivation::Operation::Copy(copy) = operation else { continue };
                    let source = copy.source.as_bytes().to_vec();
                    if present.insert(source.clone()) {
                        body.payloads.push(wire::PayloadDescriptor { nbytes: catalog.length(&source)?, id: source.clone() });
                        pending.push(source);
                    }
                }
            }
        }
        schema::validate_deployment(&body, &mut catalog)?;
        let encoded = wire::DeploymentBody::encode_to_vec(&body);
        package.deployments[position].body_digest = protection::sha256(&encoded).to_vec();
        bodies.push((position as u32, encoded));
    }
    let mut builder = EnvelopeBuilder::new(
        wire::PackageBody::encode_to_vec(&package),
        public_key,
        package_id,
    )?;
    for (index, body) in &bodies {
        builder.deployment(*index, body.clone())?;
    }
    for rank in &envelope.ranks {
        builder.rank(rank.deployment, rank.rank, rank.body.clone())?;
    }
    let manifest = builder.sign(key)?;
    let primary = staging.join("model.safetensors");
    let temporary = unique_staging(&primary, "manifest")?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|e| invalid(format!("create rewritten manifest: {e}")))?;
    let guard = TempFile::new(&temporary);
    let mut writer = Writer::new(file, metadata.primary().format())?;
    for name in metadata.primary().tensor_names() {
        if name.starts_with(container::ALIGNMENT_PREFIX) {
            continue;
        }
        if name == reader::MANIFEST {
            writer.add(name, manifest.len() as u64, Source::Bytes(manifest.clone()))?;
            continue;
        }
        let slice = metadata.primary().tensor(name)?;
        writer.add(name, slice.len(), Source::Slice(slice))?;
    }
    writer.finish()?;
    crate::package::cache::sync_file(&temporary)
        .map_err(|e| invalid(format!("sync rewritten manifest: {e}")))?;
    fs::rename(&temporary, &primary)
        .map_err(|e| invalid(format!("publish rewritten manifest: {e}")))?;
    guard.keep();
    sync_parent(&primary)?;
    Ok(())
}

fn engram_payload_ids(engram: &wire::Engram) -> Vec<Vec<u8>> {
    let mut ids = Vec::new();
    for table in &engram.tables {
        for plane in [&table.weights, &table.scales] {
            if let Some(use_) = plane
                && let Some(slice) = &use_.slice
            {
                ids.push(slice.payload_id.clone());
            }
        }
        for shard in &table.shards {
            for plane in [&shard.weights, &shard.scales] {
                if let Some(use_) = plane
                    && let Some(slice) = &use_.slice
                {
                    ids.push(slice.payload_id.clone());
                }
            }
        }
    }
    ids
}

/// Replaces installed target payloads with signed-target derivation recipes.
///
/// Every recipe is proven byte-for-byte against the installed target before its
/// file and location are removed. The primary is rewritten in place so the
/// downstream acquisition step can materialize the target from canonical
/// payloads and verify it against the signed identity. Materialization yields
/// raw payload bytes; the acquisition layer installs that output as the payload
/// object (or companion payload tensor), not as a safetensors container.
pub fn emit_derivations(
    directory: &Path,
    recipes: &[derivation::Derivation],
    limits: &derivation::Limits,
) -> Result<usize> {
    if recipes.is_empty() {
        return Ok(0);
    }
    let _lock = lock_package(directory)?;
    let metadata = Metadata::open(directory)?;
    let verified = Verified::open(directory, None)?;
    // Emission may target payloads that no initializer consumes yet; inspect
    // every installed location rather than only the current consumer closure.
    let all_ids: Vec<&[u8]> = metadata
        .locations
        .locations
        .iter()
        .map(|location| location.payload_id.as_slice())
        .collect();
    let files = metadata.payloads(&verified.catalog, all_ids)?;
    files.verify()?;
    let mut merged: BTreeMap<[u8; 32], derivation::Derivation> = BTreeMap::new();
    if let Some(set) = &metadata.derivations {
        for recipe in derivation::from_wire(set, &verified.catalog, limits)? {
            merged.insert(*recipe.target.as_bytes(), recipe);
        }
    }
    let mut fresh: BTreeSet<[u8; 32]> = BTreeSet::new();
    for recipe in recipes {
        let key = *recipe.target.as_bytes();
        match merged.get(&key) {
            Some(previous) if previous != recipe => {
                return Err(invalid("conflicting derivations for one target"));
            }
            Some(_) => continue,
            None => {
                if files.get(recipe.target.as_bytes()).is_err() {
                    return Err(invalid("derivation target is not an installed payload"));
                }
                merged.insert(key, recipe.clone());
                fresh.insert(key);
            }
        }
    }
    if fresh.is_empty() {
        return Ok(0);
    }
    let merged_recipes: Vec<derivation::Derivation> = merged.values().cloned().collect();
    derivation::validate_set(&merged_recipes, limits)?;
    derivation::validate_against_catalog(&merged_recipes, &verified.catalog, limits)?;

    let primary_path = directory.join("model.safetensors");
    let scratch = unique_staging(&primary_path, "materialize")?;
    fs::create_dir(&scratch).map_err(|e| invalid(format!("create derivation scratch: {e}")))?;
    let _scratch_guard = TempDirectory(scratch.clone());
    let mut cache: BTreeMap<[u8; 32], container::Slice> = BTreeMap::new();
    let mut visiting: BTreeSet<[u8; 32]> = BTreeSet::new();
    for recipe in merged.values() {
        if !fresh.contains(recipe.target.as_bytes()) {
            continue;
        }
        let target = files
            .get(recipe.target.as_bytes())
            .map_err(|_| invalid("derivation target is not an installed payload"))?;
        if target.len() != recipe.length {
            return Err(invalid("derivation target length differs from its payload"));
        }
        let sources = recipe_sources(
            recipe,
            &scratch,
            &files,
            &merged,
            &verified.catalog,
            &mut cache,
            &mut visiting,
            limits,
        )?;
        derivation::verify_output(target, recipe, &sources, limits)?;
    }
    drop(cache);
    drop(_scratch_guard);

    let set = wire::DerivationSet {
        version: derivation::DERIVATION_VERSION,
        derivations: merged.values().map(wire_derivation).collect(),
    };
    let removed: BTreeSet<Vec<u8>> = fresh.iter().map(|id| id.to_vec()).collect();
    let remaining_locations: Vec<wire::PayloadLocation> = metadata
        .locations
        .locations
        .iter()
        .filter(|location| !removed.contains(&location.payload_id))
        .cloned()
        .collect();
    let locations = wire::LocationSet {
        version: schema::FORMAT_VERSION,
        locations: remaining_locations,
        available_deployments: metadata.locations.available_deployments.clone(),
    };
    schema::validate_locations(&locations)?;
    let encoded_locations = locations.encode_to_vec();

    let primary = directory.join("model.safetensors");
    let staging = unique_staging(&primary, "derivations")?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staging)
        .map_err(|e| invalid(format!("create rewritten primary: {e}")))?;
    let guard = TempFile::new(&staging);
    let mut writer = Writer::new(file, metadata.primary().format())?;
    for name in metadata.primary().tensor_names() {
        if name == reader::LOCATIONS
            || name == reader::DERIVATIONS
            || name.starts_with(container::ALIGNMENT_PREFIX)
        {
            continue;
        }
        let slice = metadata.primary().tensor(name)?;
        writer.add(name, slice.len(), Source::Slice(slice))?;
    }
    writer.add(
        reader::LOCATIONS,
        encoded_locations.len() as u64,
        Source::Bytes(encoded_locations),
    )?;
    let encoded_derivations = set.encode_to_vec();
    writer.add(
        reader::DERIVATIONS,
        encoded_derivations.len() as u64,
        Source::Bytes(encoded_derivations),
    )?;
    writer.finish()?;
    crate::package::cache::sync_file(&staging)
        .map_err(|e| invalid(format!("sync rewritten primary: {e}")))?;
    fs::rename(&staging, &primary)
        .map_err(|e| invalid(format!("publish rewritten primary: {e}")))?;
    guard.keep();
    sync_parent(&primary)?;
    // Only after the index no longer references them are the derived payload
    // files removed; a leftover file is unreferenced, never a dangling location.
    let still_referenced: BTreeSet<&String> = locations
        .locations
        .iter()
        .map(|location| &location.file)
        .filter(|file| !file.is_empty())
        .collect();
    let mut cleanup_error = None;
    for location in &metadata.locations.locations {
        if removed.contains(&location.payload_id)
            && !location.file.is_empty()
            && !still_referenced.contains(&location.file)
        {
            let path = directory.join(&location.file);
            if path.exists()
                && let Err(error) = fs::remove_file(&path)
                && cleanup_error.is_none()
            {
                cleanup_error = Some(invalid(format!("remove derived payload: {error}")));
            }
        }
    }
    // The rewritten package must still present a complete selected closure; a
    // leftover file is unreferenced and harmless, so validation runs either way.
    Verified::open(directory, None)?;
    if let Some(error) = cleanup_error {
        return Err(error);
    }
    Ok(fresh.len())
}

fn recipe_sources(
    recipe: &derivation::Derivation,
    scratch: &Path,
    files: &reader::PayloadFiles,
    merged: &BTreeMap<[u8; 32], derivation::Derivation>,
    catalog: &Catalog,
    cache: &mut BTreeMap<[u8; 32], container::Slice>,
    visiting: &mut BTreeSet<[u8; 32]>,
    limits: &derivation::Limits,
) -> Result<BTreeMap<PayloadId, container::Slice>> {
    let mut sources = BTreeMap::new();
    for operation in &recipe.operations {
        if let derivation::Operation::Copy(copy) = operation {
            let slice = resolve_source(
                copy.source,
                scratch,
                files,
                merged,
                catalog,
                cache,
                visiting,
                limits,
            )?;
            sources.insert(copy.source, slice);
        }
    }
    Ok(sources)
}

fn resolve_source(
    id: PayloadId,
    scratch: &Path,
    files: &reader::PayloadFiles,
    merged: &BTreeMap<[u8; 32], derivation::Derivation>,
    catalog: &Catalog,
    cache: &mut BTreeMap<[u8; 32], container::Slice>,
    visiting: &mut BTreeSet<[u8; 32]>,
    limits: &derivation::Limits,
) -> Result<container::Slice> {
    if let Some(slice) = cache.get(id.as_bytes()) {
        return Ok(slice.clone());
    }
    if let Ok(slice) = files.get(id.as_bytes()) {
        cache.insert(*id.as_bytes(), slice.clone());
        return Ok(slice.clone());
    }
    let recipe = merged
        .get(id.as_bytes())
        .ok_or_else(|| invalid("derivation source is neither located nor derivable"))?;
    if !visiting.insert(*id.as_bytes()) {
        return Err(invalid("derivation dependency cycle"));
    }
    if catalog.length(id.as_bytes())? != recipe.length {
        return Err(invalid(
            "derivation source length differs from its descriptor",
        ));
    }
    let sources = recipe_sources(
        recipe, scratch, files, merged, catalog, cache, visiting, limits,
    )?;
    let path = scratch.join(format!("{}.bin", hex(id.as_bytes())));
    let _ = fs::remove_file(&path);
    derivation::materialize(&path, recipe, &sources, limits)?;
    let file = std::sync::Arc::new(
        File::open(&path).map_err(|e| invalid(format!("open materialized payload: {e}")))?,
    );
    let slice = container::Slice::new(file, 0, recipe.length)?;
    visiting.remove(id.as_bytes());
    cache.insert(*id.as_bytes(), slice.clone());
    Ok(slice)
}

fn wire_derivation(recipe: &derivation::Derivation) -> wire::Derivation {
    wire::Derivation {
        target_payload_id: recipe.target.as_bytes().to_vec(),
        operations: recipe
            .operations
            .iter()
            .map(|operation| {
                let operation = match operation {
                    derivation::Operation::Copy(copy) => {
                        wire::derivation_operation::Operation::Copy(wire::DerivationCopy {
                            source_payload_id: copy.source.as_bytes().to_vec(),
                            source_offset: copy.source_offset,
                            destination_offset: copy.destination_offset,
                            contiguous_bytes: copy.contiguous,
                            extents: copy.outer.iter().map(|axis| axis.count).collect(),
                            source_strides: copy
                                .outer
                                .iter()
                                .map(|axis| axis.source_stride)
                                .collect(),
                            destination_strides: copy
                                .outer
                                .iter()
                                .map(|axis| axis.destination_stride)
                                .collect(),
                        })
                    }
                    derivation::Operation::Fill(fill) => {
                        wire::derivation_operation::Operation::Fill(wire::DerivationFill {
                            destination_offset: fill.destination_offset,
                            byte_length: fill.length,
                            pattern: fill.pattern.clone(),
                        })
                    }
                };
                wire::DerivationOperation {
                    operation: Some(operation),
                }
            })
            .collect(),
    }
}

struct TempDirectory(PathBuf);

impl Drop for TempDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn staging_token() -> Result<String> {
    use ring::rand::SecureRandom;
    let mut bytes = [0u8; 8];
    ring::rand::SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| invalid("generate staging token"))?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn unique_staging(destination: &Path, tag: &str) -> Result<PathBuf> {
    let parent = destination.parent().unwrap_or(Path::new("."));
    let name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid("destination requires a filename"))?;
    Ok(parent.join(format!(
        ".{name}.{tag}-{}-{}",
        std::process::id(),
        staging_token()?
    )))
}

struct TempFile {
    path: PathBuf,
    keep: bool,
}

impl TempFile {
    fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            keep: false,
        }
    }
    fn keep(mut self) {
        self.keep = true;
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_file(&self.path);
        }
    }
}

struct PackageLock {
    _file: File,
}

fn lock_package(directory: &Path) -> Result<PackageLock> {
    let path = directory.join(".flare-package.lock");
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|e| invalid(format!("open package lock: {e}")))?;
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return Err(invalid(format!(
                "lock package: {}",
                std::io::Error::last_os_error()
            )));
        }
    }
    // Without a lock here, concurrent emission reads the same metadata, appends its own
    // derivation and rewrites the package, so one writer's derivation is lost.
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::HANDLE;
        use windows_sys::Win32::Storage::FileSystem::{LOCKFILE_EXCLUSIVE_LOCK, LockFileEx};
        use windows_sys::Win32::System::IO::OVERLAPPED;
        let mut overlapped: OVERLAPPED = unsafe { std::mem::zeroed() };
        let locked = unsafe {
            LockFileEx(
                file.as_raw_handle() as HANDLE,
                LOCKFILE_EXCLUSIVE_LOCK,
                0,
                u32::MAX,
                u32::MAX,
                &mut overlapped,
            )
        };
        if locked == 0 {
            return Err(invalid(format!(
                "lock package: {}",
                std::io::Error::last_os_error()
            )));
        }
    }
    Ok(PackageLock { _file: file })
}

#[cfg(unix)]
fn sync_parent(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    File::open(parent)
        .and_then(|file| file.sync_all())
        .map_err(|e| invalid(format!("sync package directory: {e}")))?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_parent(_path: &Path) -> Result<()> {
    Ok(())
}

fn publish_staged_file(staging: &Path, destination: &Path) -> Result<()> {
    let result = crate::package::cache::sync_file(staging)
        .map_err(|e| invalid(format!("sync packed package: {e}")))
        .and_then(|()| {
            fs::hard_link(staging, destination)
                .map_err(|e| invalid(format!("publish packed package: {e}")))
        })
        .and_then(|()| sync_parent(destination));
    let _ = fs::remove_file(staging);
    result
}

/// Inputs for producing one signed package containing several prepared TP
/// deployments. The canonical deployment is chosen internally as the lowest
/// exported tensor-parallel size, never from caller order.
/// Publisher identity written into a promoted or combined package. The model
/// license describes the checkpoint, not the publisher, so promotion carries
/// it over from the source package instead of accepting one here.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct TargetPublisher {
    pub namespace: String,
    pub official_source: String,
    pub development: bool,
}

pub struct CombineRequest<'a> {
    pub sources: &'a [PathBuf],
    pub output: &'a Path,
    pub publishers: &'a [crate::runtime_package::Publisher],
    /// Accepted publisher namespace of the source packages.
    pub accepted: &'a str,
    /// Acceptance string and namespace of the produced package.
    pub target_accepted: &'a str,
    /// Optional rewritten publisher identity for cross-namespace promotion.
    pub target_publisher: Option<&'a TargetPublisher>,
    pub key: &'a ring::signature::Ed25519KeyPair,
}

/// Re-signs one payload package under a fresh package identity.
///
/// Source publisher acceptance and target publisher identity are separate. Executables
/// are authenticated, decoded, and recompressed; decoded bytes and execution semantics
/// are preserved.
pub fn promote(
    source: &Path,
    output: &Path,
    publishers: &[crate::runtime_package::Publisher],
    accepted: &str,
    target_accepted: &str,
    target_publisher: Option<&TargetPublisher>,
    key: &ring::signature::Ed25519KeyPair,
) -> Result<crate::package::export::Published> {
    let sources = [source.to_path_buf()];
    combine(&CombineRequest {
        sources: &sources,
        output,
        publishers,
        accepted,
        target_accepted,
        target_publisher,
        key,
    })
}

struct CombineDeployment {
    body: wire::DeploymentBody,
    ranks: Vec<wire::RankBody>,
}

struct CombineSource {
    metadata: Metadata,
    package: wire::PackageBody,
    deployments: Vec<CombineDeployment>,
    catalog: Catalog,
    authorized: Vec<protection::AuthorizedPackage>,
    scratch: PathBuf,
    /// Payloads materialized from an input package's own derivations.
    extra: BTreeMap<Vec<u8>, container::Slice>,
}

/// Compare the semantic identity of independently exported Engram catalogs.
///
/// Hash offsets and physical plane/shard placements depend on the selected
/// tensor-parallel layout. Shared tables must remain byte-identical; when
/// either deployment uses partitioned planes, their semantic table identity
/// must match while each deployment retains its own physical layout.
fn engram_semantics_compatible(
    canonical: Option<&wire::Engram>,
    source: Option<&wire::Engram>,
) -> Result<()> {
    let (Some(canonical), Some(source)) = (canonical, source) else {
        return if canonical.is_none() && source.is_none() {
            Ok(())
        } else {
            Err(invalid("source deployments disagree on engram semantics"))
        };
    };
    if canonical.token_map != source.token_map
        || canonical.pad_token_id != source.pad_token_id
        || canonical.compressed_vocabulary != source.compressed_vocabulary
        || canonical.tables.len() != source.tables.len()
    {
        return Err(invalid("source deployments disagree on engram semantics"));
    }
    for (canonical, source) in canonical.tables.iter().zip(&source.tables) {
        let (Some(canonical_hash), Some(source_hash)) = (&canonical.hash, &source.hash) else {
            return Err(invalid("source deployments disagree on engram semantics"));
        };
        if canonical.layer != source.layer
            || canonical.rows != source.rows
            || canonical.embedding_width != source.embedding_width
            || canonical.scale_block != source.scale_block
            || canonical.encoding != source.encoding
            || canonical.global_scale != source.global_scale
            || canonical_hash.method != source_hash.method
            || canonical_hash.orders != source_hash.orders
            || canonical_hash.multipliers != source_hash.multipliers
            || canonical_hash.primes != source_hash.primes
        {
            return Err(invalid("source deployments disagree on engram semantics"));
        }
        if canonical.shards.is_empty() && source.shards.is_empty() && canonical != source {
            return Err(invalid(
                "source deployments disagree on shared engram planes",
            ));
        }
    }
    Ok(())
}

/// Compares the semantic preparation inputs of two rebuild manifests.
///
/// Tensor-parallel size, capacity, concurrency, chunking, and speculative shape
/// settings legitimately differ between deployments; checkpoint identity,
/// recipe packing, quantization/preparation switches, and target architecture
/// must agree. Preparer source revisions may differ because each deployment
/// retains its own `source_identity`; only the declared semantics must match.
fn rebuild_compatible(
    a: &Option<runtime::RebuildManifest>,
    b: &Option<runtime::RebuildManifest>,
) -> bool {
    let (Some(a), Some(b)) = (a, b) else {
        return a.is_none() && b.is_none();
    };
    let speculation = |value: &Option<runtime::RebuildSpeculation>| {
        value.as_ref().map(|s| {
            (
                s.proposal_source.clone(),
                s.history_drafts,
                s.draft_head_quantization.clone(),
                s.markov_head_quantization.clone(),
                s.vocabulary_parallel_sampling,
            )
        })
    };
    a.recipe == b.recipe
        && a.checkpoint_repository == b.checkpoint_repository
        && a.checkpoint_revision == b.checkpoint_revision
        && a.local_checkpoint == b.local_checkpoint
        && a.expert_scale_compression == b.expert_scale_compression
        && a.online_mxfp8 == b.online_mxfp8
        && a.language_model_only == b.language_model_only
        && a.mla_compute_mode == b.mla_compute_mode
        && speculation(&a.speculation) == speculation(&b.speculation)
}

/// Combines independently prepared TP packages into one signed package.
///
/// Every deployment of every source is preserved. Sources must agree on model
/// identity, hardware identity, frontend settings, sampling defaults, publisher
/// identity, Engram semantics, and semantic rebuild inputs; each deployment
/// retains its own deployment-scoped payloads, while tensor-parallel size and
/// capacity choices may differ. Executables are authenticated, decoded, and
/// recompressed; bodies are signed under a fresh package identity. Source
/// acceptance must match the signed publisher namespace. Derived initializers keep their signed target
/// ids and derivations are emitted where the planner proved canonical bytes.
pub fn combine(request: &CombineRequest<'_>) -> Result<crate::package::export::Published> {
    use crate::package::export::{
        self, Destination, PublishDeployment, PublishPayload, PublishRequest,
    };
    use ring::signature::KeyPair;

    if request.sources.is_empty() {
        return Err(invalid("combine requires at least one prepared package"));
    }
    // Keep every created source scratch owned by an RAII guard. The guards are
    // declared before `sources` so Windows drops source slices before removing
    // their files, including on an early error in a later source.
    let mut source_scratch_guards = Vec::with_capacity(request.sources.len());
    let mut sources = Vec::with_capacity(request.sources.len());
    for (index, path) in request.sources.iter().enumerate() {
        let metadata = Metadata::open(path)?;
        let envelope = protection::envelope(&metadata.envelope)?;
        if envelope.deployments.is_empty() {
            return Err(invalid("source package has no prepared deployment"));
        }
        let mut deployments = Vec::with_capacity(envelope.deployments.len());
        let mut authorized = Vec::with_capacity(envelope.deployments.len());
        let mut package: Option<wire::PackageBody> = None;
        let mut catalog = Catalog::new();
        for deployment in 0..envelope.deployments.len() {
            let verified = protection::verify(
                &metadata.envelope,
                request.publishers,
                protection::Selection::Deployment {
                    deployment,
                    rank: None,
                },
            )?;
            if package.is_none() {
                package = Some(verified.package().clone());
                catalog.insert_all(&verified.package().payloads)?;
            }
            let body = verified
                .deployment()
                .cloned()
                .ok_or_else(|| invalid("source package has no prepared deployment"))?;
            catalog.insert_all(&body.payloads)?;
            let ranks = verified.ranks().to_vec();
            for rank in &ranks {
                catalog.insert_all(&rank.payloads)?;
            }
            authorized.push(verified.authorize(request.accepted)?);
            deployments.push(CombineDeployment { body, ranks });
        }
        let scratch = unique_staging(request.output, &format!("source-{index}"))?;
        fs::create_dir(&scratch).map_err(|e| invalid(format!("create combine staging: {e}")))?;
        let source_scratch_guard = TempDirectory(scratch.clone());
        let mut extra: BTreeMap<Vec<u8>, container::Slice> = BTreeMap::new();
        if let Some(set) = &metadata.derivations {
            let limits = derivation::Limits::default();
            let recipes = derivation::from_wire(set, &catalog, &limits)?;
            let order = derivation::order(&recipes, &limits)?;
            let located: BTreeSet<Vec<u8>> = metadata
                .locations
                .locations
                .iter()
                .map(|location| location.payload_id.clone())
                .collect();
            for position in order {
                let recipe = &recipes[position];
                if located.contains(&recipe.target.as_bytes().to_vec()) {
                    continue;
                }
                let mut sources = BTreeMap::new();
                let mut missing = BTreeSet::new();
                for operation in &recipe.operations {
                    let derivation::Operation::Copy(copy) = operation else {
                        continue;
                    };
                    if let Some(slice) = extra.get(&copy.source.as_bytes().to_vec()) {
                        sources.insert(copy.source, slice.clone());
                    } else {
                        missing.insert(copy.source);
                    }
                }
                if !missing.is_empty() {
                    let files = metadata.payloads(
                        &catalog,
                        missing.iter().map(|id| id.as_bytes().as_slice()),
                    )?;
                    files.verify()?;
                    for id in missing {
                        sources.insert(id, files.get(id.as_bytes())?.clone());
                    }
                }
                if catalog.length(recipe.target.as_bytes())? != recipe.length {
                    return Err(invalid(
                        "derivation target length differs from its descriptor",
                    ));
                }
                let path = scratch.join(format!("derived-{}.bin", hex(recipe.target.as_bytes())));
                derivation::materialize(&path, recipe, &sources, &limits)?;
                let file = std::sync::Arc::new(
                    File::open(&path)
                        .map_err(|e| invalid(format!("open materialized payload: {e}")))?,
                );
                extra.insert(
                    recipe.target.as_bytes().to_vec(),
                    container::Slice::new(file, 0, recipe.length)?,
                );
            }
        }
        sources.push(CombineSource {
            metadata,
            package: package.ok_or_else(|| invalid("source package has no body"))?,
            deployments,
            catalog,
            authorized,
            scratch,
            extra,
        });
        source_scratch_guards.push(source_scratch_guard);
    }
    let result = (|| {
        let mut deployment_ids = BTreeSet::new();
        let mut entries: Vec<(usize, usize, u32)> = Vec::new();
        for (source_index, source) in sources.iter().enumerate() {
            for (deployment_index, deployment) in source.deployments.iter().enumerate() {
                if !deployment_ids.insert(deployment.body.deployment_id.clone()) {
                    return Err(invalid("source packages repeat one deployment id"));
                }
                let tp = if deployment.body.tensor_parallel == 0 {
                    deployment.ranks.len() as u32
                } else {
                    deployment.body.tensor_parallel
                };
                entries.push((source_index, deployment_index, tp));
            }
        }
        entries.sort_by_key(|(source, deployment, tp)| (*tp, *source, *deployment));
        let (canonical_source_index, canonical_deployment_index, canonical_tp) = entries[0];
        if canonical_tp == 0 {
            return Err(invalid("source deployment has no tensor-parallel size"));
        }
        let canonical_source = &sources[canonical_source_index];
        let canonical_deployment = &canonical_source.deployments[canonical_deployment_index];
        for (source_index, source) in sources.iter().enumerate() {
            if source.package.model_identity != canonical_source.package.model_identity
                || source.package.hardware_identity != canonical_source.package.hardware_identity
                || source.package.frontend != canonical_source.package.frontend
                || source.package.sampling_defaults != canonical_source.package.sampling_defaults
                || source.package.publisher != canonical_source.package.publisher
                || source.package.assets != canonical_source.package.assets
            {
                return Err(invalid(
                    "source packages do not describe the same prepared model",
                ));
            }
            for (deployment_index, deployment) in source.deployments.iter().enumerate() {
                if !rebuild_compatible(&deployment.body.rebuild, &canonical_deployment.body.rebuild)
                {
                    return Err(invalid(
                        "source deployments disagree on checkpoint or preparation semantics",
                    ));
                }
                engram_semantics_compatible(
                    canonical_deployment.body.engram.as_ref(),
                    deployment.body.engram.as_ref(),
                )?;
                let _ = deployment_index;
            }
            let _ = source_index;
        }
        let canonical_prepared = prepared_ranks(
            &request.sources[canonical_source_index],
            canonical_deployment_index,
        )?;
        let canonical_entry = (canonical_source_index, canonical_deployment_index);
        let mut deployments = Vec::with_capacity(entries.len());
        let mut deployment_positions: BTreeMap<String, usize> = BTreeMap::new();
        let mut used_paths = BTreeSet::new();
        let mut emitted: BTreeMap<PayloadId, derivation::Derivation> = BTreeMap::new();
        let mut conflicting: BTreeSet<PayloadId> = BTreeSet::new();
        let mut total_exception_operations = 0usize;
        let mut total_exception_runs = 0usize;
        let mut exception_targets: BTreeMap<PayloadId, PayloadId> = BTreeMap::new();
        for source in &sources {
            if let Some(set) = &source.metadata.derivations {
                for recipe in
                    derivation::from_wire(set, &source.catalog, &derivation::Limits::default())?
                {
                    match emitted.get(&recipe.target) {
                        Some(previous) if previous != &recipe => {
                            conflicting.insert(recipe.target);
                        }
                        Some(_) => {}
                        None => {
                            emitted.insert(recipe.target, recipe);
                        }
                    }
                }
            }
        }
        for (source_index, deployment_index, _) in entries {
            let source = &sources[source_index];
            let deployment = &source.deployments[deployment_index];
            let canonical_ref = if (source_index, deployment_index) == canonical_entry {
                None
            } else {
                Some(canonical_source)
            };
            let independent = if canonical_ref.is_none() {
                None
            } else {
                Some(prepared_ranks(
                    &request.sources[source_index],
                    deployment_index,
                )?)
            };
            let prepared = independent.as_ref().unwrap_or(&canonical_prepared);
            let owner_names: BTreeMap<Vec<u8>, String> = prepared
                .iter()
                .flat_map(|rank| rank.owners.iter())
                .filter_map(|owner| {
                    owner
                        .source
                        .map(|source| (source.payload.as_bytes().to_vec(), owner.name.clone()))
                })
                .collect();
            let mut plans = if canonical_ref.is_none() {
                Vec::new()
            } else {
                plan_deployment(
                    &canonical_prepared,
                    &prepared,
                    &derivation::Limits::default(),
                )?
            };
            let mut exception_payloads: BTreeMap<u32, Vec<(PayloadId, crate::package::export::PublishPayload)>> = BTreeMap::new();
            for plan in &mut plans {
                let Some(derived_rank) = prepared.iter().find(|rank| rank.rank == plan.rank) else {
                    continue;
                };
                let exceptions = finish_plan_exceptions(
                    plan,
                    derived_rank,
                    &canonical_prepared,
                    source,
                    &deployment.body.deployment_id,
                    &mut used_paths,
                    &mut total_exception_operations,
                    &mut total_exception_runs,
                )?;
                if !exceptions.is_empty() {
                    for (target, payload) in &exceptions {
                        exception_targets.insert(payload.id, *target);
                    }
                    exception_payloads.insert(plan.rank, exceptions);
                }
            }
            for plan in &plans {
                for recipe in plan.recipes.values() {
                    let Some(length) = source.catalog.length(recipe.target.as_bytes()).ok() else {
                        continue;
                    };
                    if length != recipe.length {
                        continue;
                    }
                    if derivation_uses_target(recipe) {
                        continue;
                    }
                    match emitted.get(&recipe.target) {
                        Some(previous) if previous != recipe => {
                            conflicting.insert(recipe.target);
                        }
                        Some(_) => {}
                        None => {
                            emitted.insert(recipe.target, recipe.clone());
                        }
                    }
                }
            }
            let mut recipe_sources: BTreeSet<PayloadId> = BTreeSet::new();
            for plan in &plans {
                for recipe in plan.recipes.values() {
                    for operation in &recipe.operations {
                        if let derivation::Operation::Copy(copy) = operation {
                            recipe_sources.insert(copy.source);
                        }
                    }
                }
            }
            let mut publishes =
                Vec::with_capacity(source.deployments[deployment_index].ranks.len());
            for rank in &deployment.ranks {
                let plan = plans.iter().find(|plan| plan.rank == rank.rank);
                let plan_sources: BTreeSet<PayloadId> = plan
                    .map(|plan| {
                        plan.recipes
                            .values()
                            .flat_map(|recipe| &recipe.operations)
                            .filter_map(|operation| match operation {
                                derivation::Operation::Copy(copy) => Some(copy.source),
                                derivation::Operation::Fill(_) => None,
                            })
                            .collect()
                    })
                    .unwrap_or_else(|| recipe_sources.clone());
                let exceptions = exception_payloads
                    .remove(&rank.rank)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(_, payload)| payload)
                    .collect();
                publishes.push(publish_rank(
                    source,
                    canonical_ref,
                    &source.authorized[deployment_index],
                    rank,
                    &plan_sources,
                    &owner_names,
                    &mut used_paths,
                    exceptions,
                )?);
            }
            let mut deployment_payloads = Vec::new();
            let mut rank_payloads: BTreeSet<Vec<u8>> = BTreeSet::new();
            for rank in &deployment.ranks {
                rank_payloads.extend(rank.payloads.iter().map(|descriptor| descriptor.id.clone()));
            }
            for descriptor in &deployment.body.payloads {
                if rank_payloads.contains(&descriptor.id) {
                    continue;
                }
                // Each deployment must contribute its own signed payload occurrence.
                // The publisher deduplicates identical payload identities after verifying
                // every occurrence's bytes.
                let (payload_source, relative) = (
                    combine_payload_slice(source, &descriptor.id)?,
                    semantic_relative(source, &descriptor.id),
                );
                let relative = relative.unwrap_or_else(|| {
                    format!(
                        "weights/derived/{}/{:03}.safetensors",
                        deployment.body.deployment_id,
                        deployment_payloads.len()
                    )
                });
                deployment_payloads.push(PublishPayload {
                    role: wire::PayloadRole::Weights,
                    relative: unique_relative(
                        &mut used_paths,
                        relative,
                        &deployment.body.deployment_id,
                        u32::MAX,
                    ),
                    source: payload_source,
                    id: PayloadId::from_bytes(
                        descriptor
                            .id
                            .as_slice()
                            .try_into()
                            .map_err(|_| invalid("payload id must be 32 bytes"))?,
                    ),
                    tensor: None,
                });
            }
            deployment_positions.insert(
                deployment.body.deployment_id.clone(),
                deployments.len(),
            );
            deployments.push(PublishDeployment {
                body: deployment.body.clone(),
                ranks: publishes,
                payloads: deployment_payloads,
            });
        }
        emitted.retain(|target, _| !conflicting.contains(target));
        let planned: Vec<derivation::Derivation> = emitted.values().cloned().collect();
        validate_planned_derivations(&planned)?;
        let accepted_sources: BTreeSet<PayloadId> = emitted
            .values()
            .flat_map(|recipe| &recipe.operations)
            .filter_map(|operation| match operation {
                derivation::Operation::Copy(copy) => Some(copy.source),
                derivation::Operation::Fill(_) => None,
            })
            .collect();
        for deployment in &mut deployments {
            for rank in &mut deployment.ranks {
                rank.payloads.retain(|payload| {
                    exception_targets
                        .get(&payload.id)
                        .is_none_or(|target| {
                            !conflicting.contains(target) || accepted_sources.contains(&payload.id)
                        })
                });
            }
        }
        // Every reachable derivation target and copy source must be physically
        // represented in the signed output before `emit_derivations` can prove its
        // exact bytes. Resolve lengths and ownership from each source deployment's
        // verified catalog, never from the unsigned recipe.
        let mut authorized_lengths: BTreeMap<PayloadId, (usize, String, u64)> = BTreeMap::new();
        for (source_index, source) in sources.iter().enumerate() {
            for (deployment_index, deployment) in source.deployments.iter().enumerate() {
                let catalog = source.authorized[deployment_index].verified().catalog();
                for id in catalog.ids() {
                    let id = PayloadId::from_bytes(
                        id.try_into()
                            .map_err(|_| invalid("payload id must be 32 bytes"))?,
                    );
                    let length = catalog.length(id.as_bytes())?;
                    match authorized_lengths.entry(id) {
                        std::collections::btree_map::Entry::Vacant(slot) => {
                            slot.insert((source_index, deployment.body.deployment_id.clone(), length));
                        }
                        std::collections::btree_map::Entry::Occupied(slot)
                            if slot.get().2 != length =>
                        {
                            return Err(invalid(
                                "selected deployments disagree about a payload length",
                            ));
                        }
                        std::collections::btree_map::Entry::Occupied(_) => {}
                    }
                }
            }
        }
        let mut needed: BTreeSet<PayloadId> = BTreeSet::new();
        for (target, recipe) in &emitted {
            needed.insert(*target);
            for operation in &recipe.operations {
                if let derivation::Operation::Copy(copy) = operation {
                    needed.insert(copy.source);
                }
            }
        }
        let mut published: BTreeSet<PayloadId> = deployments
            .iter()
            .flat_map(|deployment| {
                deployment
                    .payloads
                    .iter()
                    .chain(deployment.ranks.iter().flat_map(|rank| &rank.payloads))
                    .map(|payload| payload.id)
            })
            .collect();
        let mut ordinal = 0usize;
        for target in needed {
            if published.contains(&target) {
                continue;
            }
            let Some((source_index, deployment_id, length)) = authorized_lengths.get(&target) else {
                return Err(invalid(
                    "derivation dependency is not authorized by a selected deployment",
                ));
            };
            if let Some(recipe) = emitted.get(&target)
                && recipe.length != *length
            {
                return Err(invalid(
                    "derivation target length differs from its authenticated descriptor",
                ));
            }
            let source = &sources[*source_index];
            let payload_source = combine_payload_slice(source, target.as_bytes())?;
            if payload_source.len() != *length {
                return Err(invalid(
                    "derivation dependency length differs from its authenticated descriptor",
                ));
            }
            let relative = semantic_relative(source, target.as_bytes()).unwrap_or_else(|| {
                format!("weights/derived/{deployment_id}/{ordinal:03}.safetensors")
            });
            let relative = unique_relative(
                &mut used_paths,
                relative,
                deployment_id,
                u32::MAX,
            );
            let position = *deployment_positions
                .get(deployment_id)
                .ok_or_else(|| invalid("derivation owner deployment is absent"))?;
            deployments[position].payloads.push(PublishPayload {
                role: wire::PayloadRole::Weights,
                relative,
                source: payload_source,
                id: target,
                tensor: None,
            });
            published.insert(target);
            ordinal += 1;
        }
        let mut payloads = Vec::new();
        for (index, descriptor) in canonical_source.package.payloads.iter().enumerate() {
            let payload_source = combine_payload_slice(canonical_source, &descriptor.id)?;
            let location = canonical_source
                .metadata
                .locations
                .locations
                .iter()
                .find(|location| location.payload_id == descriptor.id)
                .ok_or_else(|| invalid("package payload has no location"))?;
            let tensor = canonical_source
                .package
                .assets
                .iter()
                .find(|asset| {
                    asset
                        .payload
                        .as_ref()
                        .and_then(|use_| use_.slice.as_ref())
                        .is_some_and(|slice| slice.payload_id == descriptor.id)
                })
                .map(|asset| format!("frontend.{}", asset.name.replace('_', "-")))
                .or_else(|| {
                    location
                        .tensor
                        .starts_with("frontend.")
                        .then(|| location.tensor.clone())
                })
                .unwrap_or_else(|| format!("frontend.payload-{index}"));
            payloads.push(PublishPayload {
                role: wire::PayloadRole::Frontend,
                relative: String::new(),
                source: payload_source,
                id: PayloadId::from_bytes(
                    descriptor
                        .id
                        .as_slice()
                        .try_into()
                        .map_err(|_| invalid("payload id must be 32 bytes"))?,
                ),
                tensor: Some(tensor),
            });
        }
        let mut package = canonical_source.package.clone();
        package.payloads.clear();
        package.deployments.clear();
        if let Some(target) = request.target_publisher {
            // The license belongs to the checkpoint, so the identity rewrite
            // keeps the name and link the export recorded from the model card.
            let recorded = canonical_source.package.publisher.as_ref();
            package.publisher = Some(wire::PublisherMetadata {
                namespace: target.namespace.clone(),
                official_source: target.official_source.clone(),
                development: target.development,
                license_name: recorded.map_or_else(String::new, |p| p.license_name.clone()),
                license_link: recorded.map_or_else(String::new, |p| p.license_link.clone()),
                ranks: 0,
            });
        }
        let mut package_id = [0u8; 32];
        ring::rand::SecureRandom::fill(&ring::rand::SystemRandom::new(), &mut package_id)
            .map_err(|_| invalid("generate package identity"))?;
        let public_key: [u8; 32] = request
            .key
            .public_key()
            .as_ref()
            .try_into()
            .map_err(|_| invalid("invalid signing key"))?;
        let publish_request = PublishRequest {
            package,
            payloads,
            deployments,
            package_id,
            public_key,
            accepted: request.target_accepted.to_owned(),
            no_weights: false,
        };
        // Embedded package payloads have no companion path; they are unique by
        // primary tensor name. Companions are unique by relative path.
        let mut tensors = BTreeSet::new();
        for payload in &publish_request.payloads {
            let tensor = payload
                .tensor
                .clone()
                .ok_or_else(|| invalid("package payload requires a primary tensor name"))?;
            if !tensors.insert(tensor) {
                return Err(invalid("duplicate package payload tensor name"));
            }
        }
        let mut declared = BTreeSet::new();
        let mut per_deployment: BTreeSet<String> = BTreeSet::new();
        for deployment in &publish_request.deployments {
            for payload in &deployment.payloads {
                let key = format!("{}/{}", deployment.body.deployment_id, payload.relative);
                if !per_deployment.insert(key) {
                    return Err(invalid(format!(
                        "duplicate relative path {:?}",
                        payload.relative
                    )));
                }
            }
            for rank in &deployment.ranks {
                for payload in &rank.payloads {
                    if !declared.insert(payload.relative.clone()) {
                        return Err(invalid(format!(
                            "duplicate relative path {:?}",
                            payload.relative
                        )));
                    }
                }
            }
        }
        if request.output.exists() {
            return Err(invalid(format!(
                "destination exists: {}",
                request.output.display()
            )));
        }
        let staging = unique_staging(request.output, "combined")?;
        fs::create_dir(&staging).map_err(|e| invalid(format!("create combine output: {e}")))?;
        let result = export::publish(
            &publish_request,
            request.key,
            Destination::Directory(&staging),
        )
        .map_err(|error| invalid(error.to_string()))
        .and_then(|mut published| {
            let recipes: Vec<derivation::Derivation> = emitted
                .into_values()
                .filter(|recipe| !conflicting.contains(&recipe.target))
                .collect();
            inject_dependency_descriptors(&staging, request.key, package_id, public_key, &recipes)?;
            emit_derivations(&staging, &recipes, &derivation::Limits::default())?;
            fs::rename(&staging, request.output)
                .map_err(|e| invalid(format!("publish combined package: {e}")))?;
            sync_parent(request.output)?;
            published.directory = request.output.to_path_buf();
            Ok(published)
        });
        if result.is_err() {
            let _ = fs::remove_dir_all(&staging);
        }
        result
    })();
    drop(sources);
    drop(source_scratch_guards);
    result
}

fn normalized_relative(file: &str, tensor: &str, identity: PayloadId) -> Option<String> {
    let valid = |relative: &str| {
        schema::location_path(relative).is_ok()
            && container::validate_relative_path(relative).is_ok()
    };
    let relative = container::normalized_payload_companion(file, tensor, identity);
    if valid(&relative) {
        return Some(relative);
    }
    let fallback = container::normalized_payload_companion(
        "",
        &format!("packed.{}", identity.to_hex()),
        identity,
    );
    valid(&fallback).then_some(fallback)
}

fn semantic_relative(source: &CombineSource, id: &[u8]) -> Option<String> {
    let identity = PayloadId::from_bytes(id.try_into().ok()?);
    let location = source
        .metadata
        .locations
        .locations
        .iter()
        .find(|location| location.payload_id == id)?;
    normalized_relative(&location.file, &location.tensor, identity)
}

fn combine_payload_slice(source: &CombineSource, id: &[u8]) -> Result<container::Slice> {
    if let Some(slice) = source.extra.get(id) {
        return Ok(slice.clone());
    }
    let location = source
        .metadata
        .locations
        .locations
        .iter()
        .find(|location| location.payload_id == id)
        .ok_or_else(|| invalid("payload has no location in its source package"))?;
    let slice = if location.file.is_empty() {
        source.metadata.primary().tensor(&location.tensor)?
    } else {
        let root = source.metadata.path.parent().unwrap_or(Path::new("."));
        let (_, file) = container::open_relative(root, &location.file)?;
        container::Container::open_pinned(file)?.tensor(&location.tensor)?
    };
    if slice.len() != source.catalog.length(id)? {
        return Err(invalid(
            "payload tensor length differs from its signed descriptor",
        ));
    }
    Ok(slice)
}

fn unique_relative(
    used: &mut BTreeSet<String>,
    preferred: String,
    deployment: &str,
    rank: u32,
) -> String {
    if used.insert(preferred.clone()) {
        return preferred;
    }
    let stem = preferred.trim_start_matches("weights/");
    for index in 0.. {
        let scoped = format!("weights/shared/{deployment}/rank-{rank}/{index}-{stem}");
        if used.insert(scoped.clone()) {
            return scoped;
        }
    }
    unreachable!()
}

fn publish_rank(
    source: &CombineSource,
    canonical: Option<&CombineSource>,
    authorized: &protection::AuthorizedPackage,
    rank: &wire::RankBody,
    recipe_sources: &BTreeSet<PayloadId>,
    owner_names: &BTreeMap<Vec<u8>, String>,
    used_paths: &mut BTreeSet<String>,
    exceptions: Vec<crate::package::export::PublishPayload>,
) -> Result<crate::package::export::PublishRank> {
    use crate::package::export::{PublishPayload, PublishRank, PublishResource};

    let mut body = rank.clone();
    body.resources.clear();
    let mut roles: BTreeMap<Vec<u8>, i32> = BTreeMap::new();
    for initializer in &body.initializers {
        if let Some(source) = &initializer.source
            && let Some(slice) = &source.slice
        {
            roles.entry(slice.payload_id.clone()).or_insert(source.role);
        }
    }
    for module in &body.modules {
        for global in &module.globals {
            if let Some(use_) = &global.contents
                && let Some(slice) = &use_.slice
            {
                roles.entry(slice.payload_id.clone()).or_insert(use_.role);
            }
        }
    }
    let exception_ids: BTreeSet<PayloadId> = exceptions.iter().map(|payload| payload.id).collect();
    for id in recipe_sources {
        if exception_ids.contains(id) {
            continue;
        }
        roles
            .entry(id.as_bytes().to_vec())
            .or_insert(wire::PayloadRole::Weights as i32);
    }
    let mut payloads = Vec::new();
    for (id, role) in roles {
        let payload_id = PayloadId::from_bytes(
            id.as_slice()
                .try_into()
                .map_err(|_| invalid("payload id must be 32 bytes"))?,
        );
        let (payload_source, relative) = if let Some(canonical) = canonical {
            if canonical.catalog.length(&id).is_ok() {
                let relative =
                    semantic_relative(canonical, &id).or_else(|| semantic_relative(source, &id));
                (combine_payload_slice(canonical, &id)?, relative)
            } else {
                (
                    combine_payload_slice(source, &id)?,
                    semantic_relative(source, &id),
                )
            }
        } else {
            (
                combine_payload_slice(source, &id)?,
                semantic_relative(source, &id),
            )
        };
        let relative = relative
            .or_else(|| {
                owner_names.get(&id).and_then(|name| {
                    normalized_relative(
                        "",
                        &format!("weights.{name}"),
                        payload_id,
                    )
                })
            })
            .unwrap_or_else(|| {
                format!(
                    "weights/{}/rank-{}/payload-{}.safetensors",
                    body.deployment_id,
                    rank.rank,
                    payloads.len()
                )
            });
        let relative = unique_relative(used_paths, relative, &body.deployment_id, rank.rank);
        payloads.push(PublishPayload {
            role: wire::PayloadRole::try_from(role).unwrap_or(wire::PayloadRole::Weights),
            relative,
            source: payload_source,
            id: payload_id,
            tensor: None,
        });
    }
    payloads.extend(exceptions);

    let mut resources = Vec::with_capacity(rank.resources.len());
    for resource in &rank.resources {
        let stored = resource
            .stored
            .as_ref()
            .ok_or_else(|| invalid("executable resource has no stored payload"))?;
        let payload_source = combine_payload_slice(source, &stored.payload_id)?;
        let mut reader = SliceReader {
            slice: payload_source,
            offset: stored.offset,
            remaining: stored.nbytes,
        };
        let plaintext_path = source.scratch.join(format!(
            "resource-{}-{}.bin",
            rank.rank, resource.resource_id
        ));
        let mut output = File::create(&plaintext_path)
            .map_err(|e| invalid(format!("create executable plaintext: {e}")))?;
        authorized.write_plaintext(resource, &mut reader, &mut output)?;
        output
            .sync_all()
            .map_err(|e| invalid(format!("sync executable plaintext: {e}")))?;
        resources.push(PublishResource {
            kind: wire::ExecutableKind::try_from(resource.kind)
                .map_err(|_| invalid("unknown executable kind"))?,
            path: plaintext_path,
            nbytes: resource.decoded_nbytes,
        });
    }
    Ok(PublishRank {
        body,
        payloads,
        resources,
    })
}

struct SliceReader {
    slice: container::Slice,
    offset: u64,
    remaining: u64,
}

impl std::io::Read for SliceReader {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        if self.remaining == 0 {
            return Ok(0);
        }
        let count = output.len().min(self.remaining as usize);
        self.slice
            .read_at(self.offset, &mut output[..count])
            .map_err(std::io::Error::other)?;
        self.offset += count as u64;
        self.remaining -= count as u64;
        Ok(count)
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::super::PayloadId;
    use super::*;
    use crate::package::container::{Container, Slice, write_companion};
    use crate::package::protection::{self as protect, Selection};
    use crate::runtime_package::Publisher;
    use crate::runtime_package::wire as runtime;
    use ring::{rand::SystemRandom, signature, signature::KeyPair};
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    struct Scratch(PathBuf);
    impl Scratch {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "flare-conversion-{}-{}-{tag}",
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

    struct Fixture {
        directory: PathBuf,
        envelope: Vec<u8>,
        publishers: Vec<Publisher>,
        payload: Vec<u8>,
        stored: Vec<u8>,
        derived: Vec<u8>,
    }

    fn descriptor(id: PayloadId, bytes: &[u8]) -> wire::PayloadDescriptor {
        wire::PayloadDescriptor {
            id: id.as_bytes().to_vec(),
            nbytes: bytes.len() as u64,
        }
    }

    fn slice(id: PayloadId, nbytes: u64) -> wire::PayloadSlice {
        wire::PayloadSlice {
            payload_id: id.as_bytes().to_vec(),
            offset: 0,
            nbytes,
        }
    }

    fn use_of(id: PayloadId, nbytes: u64, role: wire::PayloadRole) -> wire::PayloadUse {
        wire::PayloadUse {
            slice: Some(slice(id, nbytes)),
            role: role as i32,
        }
    }

    fn write_companion_file(dir: &Path, relative: &str, bytes: &[u8]) {
        let path = dir.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let raw = path.with_extension("bin");
        fs::write(&raw, bytes).unwrap();
        let file = std::sync::Arc::new(File::open(&raw).unwrap());
        let payload = Slice::new(file, 0, bytes.len() as u64).unwrap();
        let mut output = File::create(&path).unwrap();
        write_companion(&mut output, &payload).unwrap();
        output.sync_all().unwrap();
    }

    fn fixture(scratch: &Scratch) -> Fixture {
        fixture_with(scratch, b"stable-weight-bytes", 19, 64)
    }

    fn fixture_with(scratch: &Scratch, payload: &[u8], owner_len: u64, locality: u64) -> Fixture {
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let package_id = [5u8; 32];

        let payload = payload.to_vec();
        let weights = PayloadId::of(&payload);
        let frontend = b"tokenizer-json".to_vec();
        let frontend_id = PayloadId::of(&frontend);
        let derived_len = ((owner_len as usize) / 2).max(1).min(payload.len());
        let derived = payload[..derived_len].to_vec();
        let derived_id = PayloadId::of(&derived);

        let graph_plaintext = b"graph-node-bytes".to_vec();
        let (stored, compressed) = protect::compress(&graph_plaintext).unwrap();
        let graph_id = PayloadId::of(&stored);

        let rank = wire::RankBody {
            rank: 0,
            deployment_id: "sm120-tp1".into(),
            context: Some(runtime::Context::default()),
            allocations: vec![wire::Allocation {
                name: "weights".into(),
                nbytes: 4096,
                alignment: 4096,
                address: Some(0x1000_0000),
                memory: runtime::Memory::Device as i32,
                sizing: None,
                relocations: vec![],
            }],
            initializers: vec![wire::Initializer {
                allocation: "weights".into(),
                destination_offset: 0,
                source: Some(use_of(weights, owner_len, wire::PayloadRole::Weights)),
            }],
            payloads: vec![descriptor(graph_id, &stored)],
            prepared_weights: Some(wire::PreparedWeightCatalog {
                version: 1,
                views: vec![wire::PreparedWeightView {
                    name: "layer-000".into(),
                    group: "layer".into(),
                    dtype: "uint8".into(),
                    element_bytes: 1,
                    shape: vec![owner_len],
                    stride: vec![1],
                    layout_type: "packed".into(),
                    allocation: 0,
                    offset: 0,
                    nbytes: owner_len,
                    packing: Some(wire::PackingParameters {
                        format: "uint8".into(),
                        locality_tile: locality,
                        ..Default::default()
                    }),
                    sources: vec![wire::WeightSource {
                pieces: vec![],
                        piece_encoding: String::new(),
                        tensor: "weight".into(),
                        ranges: vec![wire::TensorRange {
                            begin: 0,
                            end: owner_len,
                        }],
                        concatenation_axis: 0,
                    }],
                    padding: vec![],
                    statistics_domain: "tensor".into(),
                }],
                unavailable: vec![],
            }),
            resources: vec![wire::ExecutableResource {
                resource_id: 77,
                kind: wire::ExecutableKind::Graph as i32,
                stored: Some(slice(graph_id, stored.len() as u64)),
                decoded_nbytes: graph_plaintext.len() as u64,
                chunk_nbytes: protect::CHUNK as u32,
                chunk_sha256: compressed.chunk_sha256,
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
        };
        let rank_bytes = rank.encode_to_vec();
        let deployment = wire::DeploymentBody {
            deployment_id: "sm120-tp1".into(),
            architecture: vec![12, 0],
            tensor_parallel: 1,
            contract: Some(Default::default()),
            payloads: vec![
                descriptor(weights, &payload),
                descriptor(derived_id, &derived),
            ],
            ranks: vec![wire::RankCommitment {
                rank: 0,
                body_digest: protect::sha256(&rank_bytes).to_vec(),
            }],
            ..Default::default()
        };
        let deployment_bytes = deployment.encode_to_vec();
        let package = wire::PackageBody {
            format_version: schema::FORMAT_VERSION,
            package_id: package_id.to_vec(),
            model_identity: "fixture".into(),
            hardware_identity: "sm120".into(),
            publisher: Some(wire::PublisherMetadata {
                namespace: "example".into(),
                official_source: "https://huggingface.co/example/model".into(),
                development: false,
                license_name: "Example".into(),
                ..Default::default()
            }),
            frontend: Some(Default::default()),
            sampling_defaults: Some(Default::default()),
            payloads: vec![descriptor(frontend_id, &frontend)],
            assets: Vec::new(),
            deployments: vec![wire::DeploymentCommitment {
                deployment_id: "sm120-tp1".into(),
                body_digest: protect::sha256(&deployment_bytes).to_vec(),
            }],
        };
        let mut builder =
            protect::EnvelopeBuilder::new(package.encode_to_vec(), public_key, package_id).unwrap();
        builder.deployment(0, deployment_bytes).unwrap();
        builder.rank(0, 0, rank_bytes).unwrap();
        let envelope = builder.sign(&key).unwrap();
        let verified = protect::verify(
            &envelope,
            std::slice::from_ref(&publisher),
            Selection::Deployment {
                deployment: 0,
                rank: None,
            },
        )
        .unwrap();
        assert!(verified.is_executable());

        let locations = wire::LocationSet {
            version: schema::FORMAT_VERSION,
            locations: vec![
                wire::PayloadLocation {
                    payload_id: weights.as_bytes().to_vec(),
                    file: "weights/layer-000.safetensors".into(),
                    tensor: COMPANION_TENSOR.into(),
                },
                wire::PayloadLocation {
                    payload_id: graph_id.as_bytes().to_vec(),
                    file: "deployments/sm120-tp1/rank-0/graphs.safetensors".into(),
                    tensor: COMPANION_TENSOR.into(),
                },
                wire::PayloadLocation {
                    payload_id: frontend_id.as_bytes().to_vec(),
                    file: String::new(),
                    tensor: "frontend.tokenizer".into(),
                },
            ],
            available_deployments: vec!["sm120-tp1".into()],
        };
        let derivations = wire::DerivationSet {
            version: derivation::DERIVATION_VERSION,
            derivations: vec![wire::Derivation {
                target_payload_id: derived_id.as_bytes().to_vec(),
                operations: vec![wire::DerivationOperation {
                    operation: Some(wire::derivation_operation::Operation::Copy(
                        wire::DerivationCopy {
                            source_payload_id: weights.as_bytes().to_vec(),
                            source_offset: 0,
                            destination_offset: 0,
                            contiguous_bytes: derived.len() as u64,
                            extents: vec![],
                            source_strides: vec![],
                            destination_strides: vec![],
                        },
                    )),
                }],
            }],
        };

        let directory = scratch.join("directory");
        fs::create_dir_all(&directory).unwrap();
        write_companion_file(&directory, "weights/layer-000.safetensors", &payload);
        write_companion_file(
            &directory,
            "deployments/sm120-tp1/rank-0/graphs.safetensors",
            &stored,
        );
        let mut primary = Writer::new(
            File::create(directory.join("model.safetensors")).unwrap(),
            container::PAYLOAD_FORMAT,
        )
        .unwrap();
        primary
            .add(
                reader::MANIFEST,
                envelope.len() as u64,
                Source::Bytes(envelope.clone()),
            )
            .unwrap();
        let locations = locations.encode_to_vec();
        primary
            .add(
                reader::LOCATIONS,
                locations.len() as u64,
                Source::Bytes(locations),
            )
            .unwrap();
        let derivations = derivations.encode_to_vec();
        primary
            .add(
                reader::DERIVATIONS,
                derivations.len() as u64,
                Source::Bytes(derivations),
            )
            .unwrap();
        primary
            .add(
                "frontend.tokenizer",
                frontend.len() as u64,
                Source::Bytes(frontend),
            )
            .unwrap();
        primary.finish().unwrap();
        Fixture {
            directory,
            envelope,
            publishers: vec![publisher],
            payload,
            stored,
            derived,
        }
    }

    fn rewrite_derivations(directory: &Path, derivations: &wire::DerivationSet) {
        let metadata = Metadata::open(directory).unwrap();
        let manifest = metadata.primary().tensor(reader::MANIFEST).unwrap();
        let locations = metadata.primary().tensor(reader::LOCATIONS).unwrap();
        let path = directory.join("model.safetensors");
        let temporary = directory.join("model.rebuilt");
        let mut output = File::create(&temporary).unwrap();
        let mut writer =
            Writer::new(&mut output, container::PAYLOAD_FORMAT).unwrap();
        writer
            .add(reader::MANIFEST, manifest.len(), Source::Slice(manifest))
            .unwrap();
        writer
            .add(reader::LOCATIONS, locations.len(), Source::Slice(locations))
            .unwrap();
        let encoded = derivations.encode_to_vec();
        writer
            .add(reader::DERIVATIONS, encoded.len() as u64, Source::Bytes(encoded))
            .unwrap();
        for name in metadata.primary().tensor_names() {
            if name == reader::MANIFEST
                || name == reader::LOCATIONS
                || name == reader::DERIVATIONS
                || name.starts_with(container::ALIGNMENT_PREFIX)
            {
                continue;
            }
            let tensor = metadata.primary().tensor(name).unwrap();
            writer.add(name, tensor.len(), Source::Slice(tensor)).unwrap();
        }
        writer.finish().unwrap();
        output.sync_all().unwrap();
        fs::rename(temporary, path).unwrap();
    }

    fn tensor_bytes(container: &Container, name: &str) -> Vec<u8> {
        let slice = container.tensor(name).unwrap();
        let mut bytes = vec![0u8; slice.len() as usize];
        slice.read_at(0, &mut bytes).unwrap();
        bytes
    }

    fn verify(envelope: &[u8], fixture: &Fixture) {
        let verified = protect::verify(
            envelope,
            &fixture.publishers,
            Selection::Deployment {
                deployment: 0,
                rank: None,
            },
        )
        .unwrap();
        assert!(verified.is_executable());
    }

    #[test]
    fn unpack_never_replaces_a_concurrently_created_destination() {
        let scratch = Scratch::new("unpack-no-replace");
        let fixture = fixture(&scratch);
        let packed = scratch.join("packed.safetensors");
        pack(&fixture.directory, &packed).unwrap();

        let output = scratch.join("output");
        fs::create_dir(&output).unwrap();
        fs::write(output.join("keep"), b"keep").unwrap();
        let error = unpack(&packed, &output).unwrap_err();
        assert!(error.to_string().contains("destination exists"), "{error}");
        assert_eq!(fs::read(output.join("keep")).unwrap(), b"keep");

        // A successful publication is likewise never replaced by a second
        // unpack of the same packed package.
        let fresh = scratch.join("fresh");
        unpack(&packed, &fresh).unwrap();
        let before = fs::read(fresh.join("model.safetensors")).unwrap();
        let error = unpack(&packed, &fresh).unwrap_err();
        assert!(error.to_string().contains("destination exists"), "{error}");
        assert_eq!(fs::read(fresh.join("model.safetensors")).unwrap(), before);
        // No staging tree survives a refused publication.
        let leftovers: Vec<_> = fs::read_dir(&scratch.0)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains("unpack-") || name.starts_with('.'))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    fn multi_payload_container_fixture(
        scratch: &Scratch,
    ) -> (PathBuf, Vec<u8>, Vec<u8>, Vec<u8>) {
        let base = fixture(scratch);
        let metadata = Metadata::open(&base.directory).unwrap();
        let weights_id = PayloadId::of(&base.payload);
        let graph_id = PayloadId::of(&base.stored);
        let frontend = metadata
            .primary()
            .tensor("frontend.tokenizer")
            .unwrap();
        let mut frontend_bytes = vec![0; frontend.len() as usize];
        frontend.read_at(0, &mut frontend_bytes).unwrap();
        let multi_path = base.directory.join("weights/layer.v2.safetensors");
        fs::create_dir_all(multi_path.parent().unwrap()).unwrap();
        let mut multi =
            Writer::new(File::create(&multi_path).unwrap(), container::PAYLOAD_FORMAT).unwrap();
        multi
            .add("part0", base.payload.len() as u64, Source::Bytes(base.payload.clone()))
            .unwrap();
        multi
            .add("part1", base.stored.len() as u64, Source::Bytes(base.stored.clone()))
            .unwrap();
        multi.finish().unwrap();
        let locations = wire::LocationSet {
            version: schema::FORMAT_VERSION,
            locations: vec![
                wire::PayloadLocation {
                    payload_id: weights_id.as_bytes().to_vec(),
                    file: "weights/layer.v2.safetensors".into(),
                    tensor: "part0".into(),
                },
                wire::PayloadLocation {
                    payload_id: graph_id.as_bytes().to_vec(),
                    file: "weights/layer.v2.safetensors".into(),
                    tensor: "part1".into(),
                },
                wire::PayloadLocation {
                    payload_id: PayloadId::of(b"tokenizer-json").as_bytes().to_vec(),
                    file: String::new(),
                    tensor: "frontend.tokenizer".into(),
                },
            ],
            available_deployments: vec!["sm120-tp1".into()],
        };
        let primary = base.directory.join("model.safetensors");
        let mut writer =
            Writer::new(File::create(&primary).unwrap(), container::PAYLOAD_FORMAT).unwrap();
        writer
            .add(reader::MANIFEST, base.envelope.len() as u64, Source::Bytes(base.envelope.clone()))
            .unwrap();
        writer
            .add(
                reader::LOCATIONS,
                locations.encode_to_vec().len() as u64,
                Source::Bytes(locations.encode_to_vec()),
            )
            .unwrap();
        writer
            .add("frontend.tokenizer", frontend_bytes.len() as u64, Source::Bytes(frontend_bytes))
            .unwrap();
        writer.finish().unwrap();
        (base.directory, base.envelope, base.payload, base.stored)
    }

    #[test]
    fn normalization_names_are_canonical_collision_safe_and_ancestor_safe() {
        let first = PayloadId::of(b"first normalization payload");
        let second = PayloadId::of(b"second normalization payload");
        let mut used = BTreeSet::new();
        let mut reserved = BTreeSet::new();
        assert_eq!(
            unique_companion_path("weights/layer-000.safetensors", COMPANION_TENSOR, first, &mut used)
                .unwrap(),
            "weights/layer-000.safetensors"
        );
        assert_eq!(
            packed_tensor_for("weights/layer-000.safetensors", COMPANION_TENSOR, first, &mut used, &reserved)
                .unwrap(),
            "weights.layer-000"
        );
        let name = packed_tensor_for("", "part0", first, &mut used, &reserved).unwrap();
        assert_eq!(name, "packed.part0");
        used.insert(name);
        let name = packed_tensor_for("", "part0", second, &mut used, &reserved).unwrap();
        assert_eq!(name, format!("packed.{}", second.to_hex()));
        reserved.insert("packed.part1".into());
        let name = packed_tensor_for("", "part1", first, &mut used, &reserved).unwrap();
        assert_eq!(name, format!("packed.{}", first.to_hex()));
        let mut ancestor = BTreeSet::from(["a".to_owned()]);
        let fallback =
            unique_companion_path("", "a.b", second, &mut ancestor).unwrap();
        assert_eq!(fallback, format!("packed/{}.safetensors", second.to_hex()));
        let mut dotted_used = BTreeSet::new();
        let dotted = packed_tensor_for(
            "weights/layer.v2.safetensors",
            COMPANION_TENSOR,
            first,
            &mut dotted_used,
            &reserved,
        )
        .unwrap();
        assert_eq!(dotted, format!("packed.{}", first.to_hex()));
        let mut primary_used = BTreeSet::new();
        let primary_reserved = BTreeSet::from(["weights.layer".to_owned()]);
        let collision = packed_tensor_for(
            "",
            "weights.layer",
            second,
            &mut primary_used,
            &primary_reserved,
        )
        .unwrap();
        assert!(!primary_reserved.contains(&collision));
    }

    #[test]
    fn signed_multi_payload_container_pack_unpack_preserves_bytes_and_manifest() {
        let scratch = Scratch::new("multi-payload-container");
        let (directory, envelope, weights, stored) = multi_payload_container_fixture(&scratch);
        let packed = scratch.join("multi-packed.safetensors");
        pack(&directory, &packed).unwrap();
        let packed_metadata = Metadata::open(&packed).unwrap();
        assert_eq!(packed_metadata.envelope, envelope);
        assert_eq!(
            packed_metadata
                .locations
                .locations
                .iter()
                .filter(|location| {
                    location.payload_id == PayloadId::of(&weights).as_bytes().to_vec()
                        || location.payload_id == PayloadId::of(&stored).as_bytes().to_vec()
                })
                .count(),
            2
        );
        let unpacked = scratch.join("multi-unpacked");
        unpack(&packed, &unpacked).unwrap();
        let unpacked_metadata = Metadata::open(&unpacked).unwrap();
        assert_eq!(unpacked_metadata.envelope, envelope);

        let file_backed = directory.join("file-backed-packed.safetensors");
        fs::copy(directory.join("model.safetensors"), &file_backed).unwrap();
        let file_backed_output = scratch.join("file-backed-unpacked");
        assert_eq!(form(&file_backed).unwrap(), Some(Form::Directory));
        assert!(unpack(&file_backed, &file_backed_output).is_err());
        unpack(&packed, &file_backed_output).unwrap();
        let file_backed_metadata = Metadata::open(&file_backed_output).unwrap();
        assert_eq!(file_backed_metadata.envelope, envelope);
        for (id, expected) in [
            (PayloadId::of(&weights), weights.as_slice()),
            (PayloadId::of(&stored), stored.as_slice()),
        ] {
            let location = unpacked_metadata
                .locations
                .locations
                .iter()
                .find(|location| location.payload_id == id.as_bytes().to_vec())
                .unwrap();
            let slice = if location.file.is_empty() {
                unpacked_metadata.primary().tensor(&location.tensor).unwrap()
            } else {
                let container = container::Container::open(&unpacked.join(&location.file)).unwrap();
                container.tensor(COMPANION_TENSOR).unwrap()
            };
            let mut bytes = vec![0; slice.len() as usize];
            slice.read_at(0, &mut bytes).unwrap();
            assert_eq!(bytes, expected);
        }
    }

    #[test]
    fn primary_path_and_directory_select_the_same_external_companion_form() {
        let scratch = Scratch::new("primary-form");
        let (directory, envelope, _weights, _stored) =
            multi_payload_container_fixture(&scratch);
        let primary = directory.join("model.safetensors");
        assert_eq!(form(&directory).unwrap(), Some(Form::Directory));
        assert_eq!(form(&primary).unwrap(), Some(Form::Directory));

        let from_directory = scratch.join("from-directory.safetensors");
        let from_primary = scratch.join("from-primary.safetensors");
        pack(&directory, &from_directory).unwrap();
        pack(&primary, &from_primary).unwrap();
        let directory_metadata = Metadata::open(&from_directory).unwrap();
        let primary_metadata = Metadata::open(&from_primary).unwrap();
        assert_eq!(directory_metadata.envelope, envelope);
        assert_eq!(primary_metadata.envelope, envelope);
        assert_eq!(
            directory_metadata.locations.locations,
            primary_metadata.locations.locations
        );
        assert_eq!(form(&from_directory).unwrap(), Some(Form::Packed));
        assert_eq!(form(&from_primary).unwrap(), Some(Form::Packed));
        assert!(pack(&from_primary, &scratch.join("packed-again.safetensors")).is_err());

        let unpacked = scratch.join("unpacked-from-primary");
        unpack(&from_primary, &unpacked).unwrap();
        assert_eq!(form(&unpacked).unwrap(), Some(Form::Directory));
        assert_eq!(
            form(&unpacked.join("model.safetensors")).unwrap(),
            Some(Form::Directory)
        );
        let unpacked_primary = unpacked.join("model.safetensors");
        let repacked = scratch.join("repacked-from-primary.safetensors");
        pack(&unpacked_primary, &repacked).unwrap();
        assert_eq!(Metadata::open(&repacked).unwrap().envelope, envelope);
        let verified = Verified::open(&repacked, None).unwrap();
        verified.payloads().unwrap().verify().unwrap();

        // An explicit directory stays directory form even when every payload
        // is embedded in its primary; the primary file itself is true packed form.
        let embedded = scratch.join("embedded-directory");
        fs::create_dir(&embedded).unwrap();
        fs::copy(&from_primary, embedded.join("model.safetensors")).unwrap();
        assert_eq!(form(&embedded).unwrap(), Some(Form::Directory));
        assert_eq!(
            form(&embedded.join("model.safetensors")).unwrap(),
            Some(Form::Packed)
        );
        assert!(unpack(&embedded, &scratch.join("embedded-out")).is_err());
        pack(&embedded, &scratch.join("embedded-packed.safetensors")).unwrap();
        unpack(
            &embedded.join("model.safetensors"),
            &scratch.join("embedded-primary-out"),
        )
        .unwrap();
    }

    #[test]
    fn pack_unpack_round_trip_preserves_payload_identity_and_envelope() {
        let scratch = Scratch::new("identity-roundtrip");
        let fixture = fixture(&scratch);
        let packed = scratch.join("packed.safetensors");
        pack(&fixture.directory, &packed).unwrap();
        let manifest = Metadata::open(&packed).unwrap().envelope;
        let unpacked = scratch.join("unpacked");
        unpack(&packed, &unpacked).unwrap();
        let again = scratch.join("repacked.safetensors");
        pack(&unpacked, &again).unwrap();
        assert_eq!(
            Metadata::open(&again).unwrap().envelope,
            manifest,
            "signed envelope changed"
        );

        // Payload identities are identical under both canonical forms, and the
        // check needs no signing key.
        let first = Verified::open(&packed, None).unwrap();
        let second = Verified::open(&again, None).unwrap();
        first.payloads().unwrap().verify().unwrap();
        second.payloads().unwrap().verify().unwrap();
        let summary = |verified: &Verified| -> Vec<(Vec<u8>, u64)> {
            let mut payloads: Vec<(Vec<u8>, u64)> = verified
                .payloads()
                .unwrap()
                .iter()
                .map(|(id, slice)| (id.to_vec(), slice.len()))
                .collect();
            payloads.sort();
            payloads
        };
        assert_eq!(summary(&first), summary(&second));
    }

    #[test]
    fn out_of_bounds_recipe_is_rejected_before_any_publication() {
        let scratch = Scratch::new("recipe-bounds");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let directory = publish_chain_source(&scratch, &key, &publisher, true);
        let weights = PayloadId::of(&(0..16u8).collect::<Vec<u8>>());
        let middle = PayloadId::of(&(0..8u8).collect::<Vec<u8>>());
        let before = Metadata::open(&directory).unwrap();

        // The copy reaches past the signed source payload, so the recipe must be
        // refused without rewriting the primary or removing any companion.
        let bad = derivation::Derivation {
            target: middle,
            length: 8,
            operations: vec![derivation::Operation::Copy(derivation::Copy {
                source: weights,
                source_offset: 12,
                destination_offset: 0,
                contiguous: 8,
                outer: Vec::new(),
            })],
        };
        let error =
            emit_derivations(&directory, &[bad], &derivation::Limits::default()).unwrap_err();
        assert!(!error.to_string().is_empty());
        let after = Metadata::open(&directory).unwrap();
        assert_eq!(after.envelope, before.envelope);
        assert_eq!(after.locations.locations, before.locations.locations);
        for location in &before.locations.locations {
            assert!(directory.join(&location.file).exists(), "{}", location.file);
        }
        Verified::open(&directory, None).unwrap();
    }

    #[test]
    fn pack_then_unpack_preserves_signed_bytes_and_stored_executables() {
        let scratch = Scratch::new("roundtrip");
        let fixture = fixture(&scratch);
        let packed = scratch.join("packed.safetensors");
        pack(&fixture.directory, &packed).unwrap();

        let packed_container = Container::open(&packed).unwrap();
        let manifest = tensor_bytes(&packed_container, reader::MANIFEST);
        assert_eq!(manifest, fixture.envelope);
        verify(&manifest, &fixture);
        assert_eq!(
            tensor_bytes(&packed_container, "weights.layer-000"),
            fixture.payload
        );
        assert_eq!(
            tensor_bytes(&packed_container, "deployments.sm120-tp1.rank-0.graphs"),
            fixture.stored
        );
        let derivations = tensor_bytes(&packed_container, reader::DERIVATIONS);
        assert!(!derivations.is_empty());

        let out = scratch.join("unpacked");
        unpack(&packed, &out).unwrap();
        let metadata = Metadata::open(&out).unwrap();
        assert_eq!(metadata.envelope, fixture.envelope);
        verify(&metadata.envelope, &fixture);
        assert_eq!(
            tensor_bytes(metadata.primary(), reader::DERIVATIONS),
            derivations
        );
        assert_eq!(
            tensor_bytes(metadata.primary(), "frontend.tokenizer"),
            b"tokenizer-json".to_vec()
        );
        let weights = Container::open(&out.join("weights/layer-000.safetensors")).unwrap();
        assert_eq!(tensor_bytes(&weights, COMPANION_TENSOR), fixture.payload);
        let graphs =
            Container::open(&out.join("deployments/sm120-tp1/rank-0/graphs.safetensors")).unwrap();
        assert_eq!(tensor_bytes(&graphs, COMPANION_TENSOR), fixture.stored);
        // Source-complete derivation: the target has no companion.
        assert!(!out.join("weights/stable-weight.safetensors").exists());
        assert!(!out.join("weights").read_dir().unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("stable-weight")
        }));
    }

    fn drop_location(directory: &Path, prefix: &str) {
        let primary = directory.join("model.safetensors");
        let container = Container::open(&primary).unwrap();
        let mut tensors: Vec<(String, Vec<u8>)> = Vec::new();
        for name in container.tensor_names() {
            if !name.starts_with(container::ALIGNMENT_PREFIX) && name != reader::LOCATIONS {
                tensors.push((name.to_owned(), tensor_bytes(&container, name)));
            }
        }
        let mut set =
            wire::LocationSet::decode(tensor_bytes(&container, reader::LOCATIONS).as_slice())
                .unwrap();
        drop(container);
        for location in &set.locations {
            if location.file.starts_with(prefix) && !location.file.is_empty() {
                let _ = fs::remove_file(directory.join(&location.file));
            }
        }
        set.locations
            .retain(|location| !location.file.starts_with(prefix));
        let locations = set.encode_to_vec();
        let mut writer =
            Writer::new(File::create(&primary).unwrap(), container::PAYLOAD_FORMAT).unwrap();
        for (name, bytes) in tensors {
            writer
                .add(&name, bytes.len() as u64, Source::Bytes(bytes))
                .unwrap();
        }
        writer
            .add(
                reader::LOCATIONS,
                locations.len() as u64,
                Source::Bytes(locations),
            )
            .unwrap();
        writer.finish().unwrap();
    }

    #[test]
    fn combiner_sharing_requires_exact_bytes() {
        let scratch = Scratch::new("sharing");
        let source = fixture(&scratch);
        let weights = Container::open(&source.directory.join("weights/layer-000.safetensors"))
            .unwrap()
            .tensor(COMPANION_TENSOR)
            .unwrap();
        let copy = entries(&source.directory, "weights/layer-000.safetensors");
        let canonical = BTreeMap::from([(PayloadId::of(&source.payload), copy.clone())]);
        let identical =
            sharing(&weights, &canonical, None, &derivation::Limits::default()).unwrap();
        assert_eq!(
            identical,
            Sharing::Identical(PayloadId::of(&source.payload))
        );

        let other = scratch.join("other.bin");
        fs::write(&other, b"different-weight").unwrap();
        let other = std::sync::Arc::new(File::open(&other).unwrap());
        let other = Slice::new(other, 0, 16).unwrap();
        assert_eq!(
            sharing(&other, &canonical, None, &derivation::Limits::default()).unwrap(),
            Sharing::PerSize
        );

        let candidate = wire::Derivation {
            target_payload_id: PayloadId::of(&source.derived).as_bytes().to_vec(),
            operations: vec![wire::DerivationOperation {
                operation: Some(wire::derivation_operation::Operation::Copy(
                    wire::DerivationCopy {
                        source_payload_id: PayloadId::of(&source.payload).as_bytes().to_vec(),
                        source_offset: 0,
                        destination_offset: 0,
                        contiguous_bytes: source.derived.len() as u64,
                        extents: vec![],
                        source_strides: vec![],
                        destination_strides: vec![],
                    },
                )),
            }],
        };
        let set = wire::DerivationSet {
            version: derivation::DERIVATION_VERSION,
            derivations: vec![candidate],
        };
        let candidate =
            derivation::from_wire(&set, &catalog_with(&source), &derivation::Limits::default())
                .unwrap()
                .remove(0);
        let target = scratch.join("derived.bin");
        fs::write(&target, &source.derived).unwrap();
        let target = std::sync::Arc::new(File::open(&target).unwrap());
        let target = Slice::new(target, 0, source.derived.len() as u64).unwrap();
        let sources = BTreeMap::from([(PayloadId::of(&source.payload), copy)]);
        assert_eq!(
            sharing(
                &target,
                &BTreeMap::new(),
                Some((&candidate, &sources)),
                &derivation::Limits::default()
            )
            .unwrap(),
            Sharing::Corroborated
        );
    }

    fn catalog_with(fixture: &Fixture) -> Catalog {
        let mut catalog = Catalog::new();
        for (bytes, id) in [
            (fixture.payload.as_slice(), PayloadId::of(&fixture.payload)),
            (fixture.derived.as_slice(), PayloadId::of(&fixture.derived)),
        ] {
            catalog
                .insert(&wire::PayloadDescriptor {
                    id: id.as_bytes().to_vec(),
                    nbytes: bytes.len() as u64,
                })
                .unwrap();
        }
        catalog
    }

    fn entries(directory: &Path, relative: &str) -> Slice {
        Container::open(&directory.join(relative))
            .unwrap()
            .tensor(COMPANION_TENSOR)
            .unwrap()
    }

    fn publish_source(
        scratch: &Scratch,
        name: &str,
        tp: u32,
        halves: &[&[u8]],
        columns: &[&[u8]],
        key: &signature::Ed25519KeyPair,
        publisher: &Publisher,
        model: &str,
        engram: bool,
    ) -> PathBuf {
        publish_source_with(
            scratch, name, tp, halves, columns, key, publisher, model, engram, None,
        )
    }

    #[derive(Clone, Copy)]
    enum EngramFixture {
        None,
        Shared,
        Tp1Shared,
        Partitioned,
        PartitionedMismatch,
    }

    const ENGRAM_HEADS: usize = 24;
    const ENGRAM_ROWS: usize = 48;
    const ENGRAM_WIDTH: usize = 256;
    const ENGRAM_SCALE_BLOCK: usize = 32;
    const ENGRAM_SCALES_PER_ROW: usize = ENGRAM_WIDTH / ENGRAM_SCALE_BLOCK;
    const ENGRAM_ROWS_PER_HEAD: usize = 2;

    fn engram_orders() -> Vec<u32> {
        let mut orders = vec![2; 8];
        orders.extend(std::iter::repeat_n(3, 8));
        orders.extend(std::iter::repeat_n(4, 8));
        orders
    }

    fn engram_tp1_offsets() -> Vec<u64> {
        (0..ENGRAM_HEADS)
            .map(|head| (head * ENGRAM_ROWS_PER_HEAD) as u64)
            .collect()
    }

    fn engram_tp2_offsets() -> Vec<u64> {
        (0..ENGRAM_HEADS)
            .map(|head| {
                ((head % 2) * (ENGRAM_ROWS / 2) + (head / 2) * ENGRAM_ROWS_PER_HEAD) as u64
            })
            .collect()
    }

    fn engram_multipliers() -> Vec<u64> {
        vec![1, 3, 5, 7]
    }

    fn engram_hash(offsets: Vec<u64>) -> wire::EngramHash {
        wire::EngramHash {
            method: wire::engram_hash::Method::XorModulo as i32,
            orders: engram_orders(),
            multipliers: engram_multipliers(),
            offsets,
            primes: vec![2; ENGRAM_HEADS],
        }
    }

    fn engram_tp1_weights() -> Vec<u8> {
        let mut weights = Vec::with_capacity(ENGRAM_ROWS * ENGRAM_WIDTH);
        for row in 0..ENGRAM_ROWS {
            for column in 0..ENGRAM_WIDTH {
                weights.push((row as u8).wrapping_mul(7).wrapping_add(column as u8));
            }
        }
        weights
    }

    fn engram_tp1_scales() -> Vec<u8> {
        let mut scales = Vec::with_capacity(ENGRAM_ROWS * ENGRAM_SCALES_PER_ROW);
        for row in 0..ENGRAM_ROWS {
            for block in 0..ENGRAM_SCALES_PER_ROW {
                scales.push((row as u8).wrapping_mul(19).wrapping_add(block as u8));
            }
        }
        scales
    }

    fn engram_tp2_shard(tp1: &[u8], row_bytes: usize, rank: usize) -> Vec<u8> {
        let rows_per_shard = ENGRAM_ROWS / 2;
        let mut shard = Vec::with_capacity(rows_per_shard * row_bytes);
        for head in (rank..ENGRAM_HEADS).step_by(2) {
            let begin = head * ENGRAM_ROWS_PER_HEAD * row_bytes;
            let end = begin + ENGRAM_ROWS_PER_HEAD * row_bytes;
            shard.extend_from_slice(&tp1[begin..end]);
        }
        shard
    }

    fn publish_source_with(
        scratch: &Scratch,
        name: &str,
        tp: u32,
        halves: &[&[u8]],
        columns: &[&[u8]],
        key: &signature::Ed25519KeyPair,
        publisher: &Publisher,
        model: &str,
        engram: bool,
        rebuild: Option<runtime::RebuildManifest>,
    ) -> PathBuf {
        publish_source_impl(
            scratch,
            name,
            tp,
            halves,
            columns,
            key,
            publisher,
            model,
            if engram {
                EngramFixture::Shared
            } else {
                EngramFixture::None
            },
            rebuild,
        )
    }

    fn publish_source_impl(
        scratch: &Scratch,
        name: &str,
        tp: u32,
        halves: &[&[u8]],
        columns: &[&[u8]],
        key: &signature::Ed25519KeyPair,
        publisher: &Publisher,
        model: &str,
        engram: EngramFixture,
        rebuild: Option<runtime::RebuildManifest>,
    ) -> PathBuf {
        publish_source_impl_with_alias(
            scratch,
            name,
            tp,
            halves,
            columns,
            key,
            publisher,
            model,
            engram,
            rebuild,
            false,
        )
    }

    fn publish_source_impl_with_alias(
        scratch: &Scratch,
        name: &str,
        tp: u32,
        halves: &[&[u8]],
        columns: &[&[u8]],
        key: &signature::Ed25519KeyPair,
        publisher: &Publisher,
        model: &str,
        engram: EngramFixture,
        rebuild: Option<runtime::RebuildManifest>,
        aliased: bool,
    ) -> PathBuf {
        use crate::package::export::{
            Destination, PublishDeployment, PublishPayload, PublishRank, PublishRequest,
            PublishResource,
        };
        let root = scratch.join(name);
        fs::create_dir_all(&root).unwrap();
        let rows = halves[0].len() as u64 / 2;
        let mut ranks = Vec::new();
        for (index, half) in halves.iter().enumerate() {
            let weight = root.join(format!("weights-{index}.bin"));
            fs::write(&weight, half).unwrap();
            let graph = root.join(format!("graphs-{index}.bin"));
            fs::write(&graph, format!("graph-{tp}-{index}")).unwrap();
            let cubin = root.join(format!("cubins-{index}.bin"));
            fs::write(&cubin, format!("cubin-{tp}-{index}")).unwrap();
            let state = format!("state-{tp}-{index}").into_bytes();
            let begin = index as u64 * rows;
            let column = columns[index];
            let column_begin = if columns.len() == 1 { 0 } else { index as u64 };
            let total_rows = rows * columns.len() as u64;
            let weight_column = root.join(format!("columns-{index}.bin"));
            fs::write(&weight_column, column).unwrap();
            let rank = wire::RankBody {
                rank: index as u32,
                deployment_id: format!("sm120-tp{tp}"),
                context: Some(runtime::Context::default()),
                allocations: vec![wire::Allocation {
                    name: "weights".into(),
                    nbytes: 12288,
                    alignment: 4096,
                    address: Some(0x1000_0000),
                    memory: runtime::Memory::Device as i32,
                    sizing: None,
                    relocations: vec![],
                }],
                initializers: vec![
                    wire::Initializer {
                        allocation: "weights".into(),
                        destination_offset: 0,
                        source: Some(use_of(
                            PayloadId::of(&state),
                            state.len() as u64,
                            wire::PayloadRole::InitialState,
                        )),
                    },
                    wire::Initializer {
                        allocation: "weights".into(),
                        destination_offset: 4096,
                        source: Some(use_of(
                            PayloadId::of(half),
                            half.len() as u64,
                            wire::PayloadRole::Weights,
                        )),
                    },
                    wire::Initializer {
                        allocation: "weights".into(),
                        destination_offset: 8192,
                        source: Some(use_of(
                            PayloadId::of(column),
                            column.len() as u64,
                            wire::PayloadRole::Weights,
                        )),
                    },
                ],
                modules: vec![wire::Module {
                    name: "kernel".into(),
                    cubin: 0,
                    globals: vec![wire::ModuleGlobal {
                        symbol: "constant".into(),
                        contents: Some(use_of(
                            PayloadId::of(&state),
                            state.len() as u64,
                            wire::PayloadRole::InitialState,
                        )),
                        nbytes: state.len() as u64,
                        relocations: vec![],
                    }],
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
                    functions: vec![0],
                    ..Default::default()
                }],
                prepared_weights: Some(wire::PreparedWeightCatalog {
                    version: 1,
                    views: vec![
                        wire::PreparedWeightView {
                            name: if aliased {
                                "mtp.attention['qkv_proj']".into()
                            } else {
                                "layer-000".into()
                            },
                            group: "layer".into(),
                            dtype: "uint8".into(),
                            element_bytes: 1,
                            shape: vec![rows, 2],
                            stride: vec![2, 1],
                            layout_type: "packed".into(),
                            allocation: 0,
                            offset: 4096,
                            nbytes: half.len() as u64,
                            packing: Some(wire::PackingParameters {
                                format: "uint8".into(),
                                locality_tile: 64,
                                ..Default::default()
                            }),
                            sources: vec![wire::WeightSource {
                pieces: vec![],
                                piece_encoding: String::new(),
                                tensor: "weight".into(),
                                ranges: vec![
                                    wire::TensorRange {
                                        begin,
                                        end: begin + rows,
                                    },
                                    wire::TensorRange { begin: 0, end: 2 },
                                ],
                                concatenation_axis: 0,
                            }],
                            statistics_domain: "tensor".into(),
                            padding: vec![],
                        },
                        wire::PreparedWeightView {
                            name: if aliased {
                                "mtp.attention['o_proj']".into()
                            } else {
                                "col-000".into()
                            },
                            group: "layer".into(),
                            dtype: "uint8".into(),
                            element_bytes: 1,
                            shape: vec![total_rows, if columns.len() == 1 { 2 } else { 1 }],
                            stride: vec![if columns.len() == 1 { 2 } else { 1 }, 1],
                            layout_type: "packed".into(),
                            allocation: 0,
                            offset: 8192,
                            nbytes: column.len() as u64,
                            packing: Some(wire::PackingParameters {
                                format: "uint8".into(),
                                locality_tile: 64,
                                ..Default::default()
                            }),
                            sources: vec![wire::WeightSource {
                pieces: vec![],
                                piece_encoding: String::new(),
                                tensor: "colweight".into(),
                                ranges: vec![
                                    wire::TensorRange {
                                        begin: 0,
                                        end: total_rows,
                                    },
                                    wire::TensorRange {
                                        begin: column_begin,
                                        end: column_begin + if columns.len() == 1 { 2 } else { 1 },
                                    },
                                ],
                                concatenation_axis: 0,
                            }],
                            statistics_domain: "tensor".into(),
                            padding: vec![],
                        },
                    ],
                    unavailable: vec![],
                }),
                ..Default::default()
            };
            ranks.push(PublishRank {
                body: rank,
                payloads: vec![
                    PublishPayload {
                        role: wire::PayloadRole::Weights,
                        relative: if tp == 1 {
                            "weights/layer-000.safetensors".into()
                        } else {
                            format!("weights/layer-000/rank-{index}.safetensors")
                        },
                        source: crate::package::export::pin_file(&weight, half.len() as u64)
                            .unwrap(),
                        id: PayloadId::of(half),
                        tensor: None,
                    },
                    PublishPayload {
                        role: wire::PayloadRole::Weights,
                        relative: if tp == 1 {
                            "weights/col-000.safetensors".into()
                        } else {
                            format!("weights/col-000/rank-{index}.safetensors")
                        },
                        source: crate::package::export::pin_file(
                            &weight_column,
                            column.len() as u64,
                        )
                        .unwrap(),
                        id: PayloadId::of(column),
                        tensor: None,
                    },
                    PublishPayload {
                        role: wire::PayloadRole::InitialState,
                        relative: format!("state/{tp}/{index}.safetensors"),
                        source: {
                            let path = root.join(format!("state-{index}.bin"));
                            fs::write(&path, &state).unwrap();
                            crate::package::export::pin_file(&path, state.len() as u64).unwrap()
                        },
                        id: PayloadId::of(&state),
                        tensor: None,
                    },
                ],
                resources: vec![
                    PublishResource {
                        kind: wire::ExecutableKind::Cubin,
                        path: cubin,
                        nbytes: format!("cubin-{tp}-{index}").len() as u64,
                    },
                    PublishResource {
                        kind: wire::ExecutableKind::Graph,
                        path: graph,
                        nbytes: format!("graph-{tp}-{index}").len() as u64,
                    },
                ],
            });
        }
        let mut deployment_payloads = Vec::new();
        let mut engram_body = None;
        match engram {
            EngramFixture::None => {}
            EngramFixture::Shared => {
                let weights = b"engram-weights".to_vec();
                let scales = b"engram-scales!".to_vec();
                let weights_id = PayloadId::of(&weights);
                let scales_id = PayloadId::of(&scales);
                for (name, bytes, id, role) in [
                    (
                        "engram/weights.safetensors",
                        weights,
                        weights_id,
                        wire::PayloadRole::Weights,
                    ),
                    (
                        "engram/scales.safetensors",
                        scales,
                        scales_id,
                        wire::PayloadRole::TunedWeights,
                    ),
                ] {
                    let path = root.join(name.replace('/', "-"));
                    fs::write(&path, &bytes).unwrap();
                    deployment_payloads.push(PublishPayload {
                        role,
                        relative: name.into(),
                        source: crate::package::export::pin_file(&path, bytes.len() as u64)
                            .unwrap(),
                        id,
                        tensor: None,
                    });
                }
                engram_body = Some(wire::Engram {
                    token_map: vec![1, 2],
                    pad_token_id: 0,
                    compressed_vocabulary: 8,
                    tables: vec![wire::EngramTable {
                        layer: 1,
                        rows: 4,
                        embedding_width: 2,
                        scale_block: 16,
                        weights: Some(use_of(weights_id, 14, wire::PayloadRole::Weights)),
                        scales: Some(use_of(scales_id, 14, wire::PayloadRole::TunedWeights)),
                        hash: Some(wire::EngramHash {
                            method: wire::engram_hash::Method::XorModulo as i32,
                            ..Default::default()
                        }),
                        weight_buffer: "w".into(),
                        scale_buffer: "s".into(),
                        hash_buffer: "h".into(),
                        ready_event: "e".into(),
                        request_buffer: "r".into(),
                        encoding: wire::engram_table::Encoding::Fp8E4m3E8m0 as i32,
                        global_scale: 1.0,
                        shards: vec![],
                    }],
                });
            }
            EngramFixture::Tp1Shared => {
                let weights = engram_tp1_weights();
                let scales = engram_tp1_scales();
                let weights_id = PayloadId::of(&weights);
                let scales_id = PayloadId::of(&scales);
                for (name, bytes, id, role) in [
                    (
                        "engram/tp1-weights.safetensors",
                        weights,
                        weights_id,
                        wire::PayloadRole::Weights,
                    ),
                    (
                        "engram/tp1-scales.safetensors",
                        scales,
                        scales_id,
                        wire::PayloadRole::TunedWeights,
                    ),
                ] {
                    let path = root.join(name.replace('/', "-"));
                    fs::write(&path, &bytes).unwrap();
                    deployment_payloads.push(PublishPayload {
                        role,
                        relative: name.into(),
                        source: crate::package::export::pin_file(&path, bytes.len() as u64)
                            .unwrap(),
                        id,
                        tensor: None,
                    });
                }
                engram_body = Some(wire::Engram {
                    token_map: vec![1, 2],
                    pad_token_id: 0,
                    compressed_vocabulary: 8,
                    tables: vec![wire::EngramTable {
                        layer: 1,
                        rows: ENGRAM_ROWS as u64,
                        embedding_width: ENGRAM_WIDTH as u32,
                        scale_block: ENGRAM_SCALE_BLOCK as u32,
                        weights: Some(use_of(
                            weights_id,
                            (ENGRAM_ROWS * ENGRAM_WIDTH) as u64,
                            wire::PayloadRole::Weights,
                        )),
                        scales: Some(use_of(
                            scales_id,
                            (ENGRAM_ROWS * ENGRAM_SCALES_PER_ROW) as u64,
                            wire::PayloadRole::TunedWeights,
                        )),
                        hash: Some(engram_hash(engram_tp1_offsets())),
                        weight_buffer: "w".into(),
                        scale_buffer: "s".into(),
                        hash_buffer: "h".into(),
                        ready_event: "e".into(),
                        request_buffer: "r".into(),
                        encoding: wire::engram_table::Encoding::Fp8E4m3E8m0 as i32,
                        global_scale: 1.0,
                        shards: vec![],
                    }],
                });
            }
            EngramFixture::Partitioned | EngramFixture::PartitionedMismatch => {
                assert_eq!(halves.len(), 2, "partitioned Engram fixture requires TP2");
                let mismatch = matches!(engram, EngramFixture::PartitionedMismatch);
                let tp1_weights = engram_tp1_weights();
                let tp1_scales = engram_tp1_scales();
                let rows_per_shard = ENGRAM_ROWS / 2;
                let mut shards = Vec::new();
                for index in 0..halves.len() {
                    let weights = engram_tp2_shard(&tp1_weights, ENGRAM_WIDTH, index);
                    let scales = engram_tp2_shard(&tp1_scales, ENGRAM_SCALES_PER_ROW, index);
                    let weights_id = PayloadId::of(&weights);
                    let scales_id = PayloadId::of(&scales);
                    for (relative, bytes, id, role) in [
                        (
                            format!("engram/weights-rank-{index}.safetensors"),
                            weights,
                            weights_id,
                            wire::PayloadRole::Weights,
                        ),
                        (
                            format!("engram/scales-rank-{index}.safetensors"),
                            scales,
                            scales_id,
                            wire::PayloadRole::TunedWeights,
                        ),
                    ] {
                        let path = root.join(relative.replace('/', "-"));
                        fs::write(&path, &bytes).unwrap();
                        deployment_payloads.push(PublishPayload {
                            role,
                            relative: relative.into(),
                            source: crate::package::export::pin_file(&path, bytes.len() as u64)
                                .unwrap(),
                            id,
                            tensor: None,
                        });
                    }
                    shards.push(wire::EngramTableShard {
                        rank: index as u32,
                        first_row: (index * rows_per_shard) as u64,
                        rows: rows_per_shard as u64,
                        weights: Some(use_of(
                            weights_id,
                            (rows_per_shard * ENGRAM_WIDTH) as u64,
                            wire::PayloadRole::Weights,
                        )),
                        scales: Some(use_of(
                            scales_id,
                            (rows_per_shard * ENGRAM_SCALES_PER_ROW) as u64,
                            wire::PayloadRole::TunedWeights,
                        )),
                    });
                }
                engram_body = Some(wire::Engram {
                    token_map: vec![1, 2],
                    pad_token_id: 0,
                    compressed_vocabulary: if mismatch { 9 } else { 8 },
                    tables: vec![wire::EngramTable {
                        layer: 1,
                        rows: ENGRAM_ROWS as u64,
                        embedding_width: ENGRAM_WIDTH as u32,
                        scale_block: ENGRAM_SCALE_BLOCK as u32,
                        weights: None,
                        scales: None,
                        hash: Some(engram_hash(engram_tp2_offsets())),
                        weight_buffer: "w".into(),
                        scale_buffer: "s".into(),
                        hash_buffer: "h".into(),
                        ready_event: "e".into(),
                        request_buffer: "r".into(),
                        encoding: wire::engram_table::Encoding::Fp8E4m3E8m0 as i32,
                        global_scale: 1.0,
                        shards,
                    }],
                });
            }
        }
        let frontend = b"tokenizer-json".to_vec();
        let frontend_id = PayloadId::of(&frontend);
        let frontend_path = root.join("frontend-tokenizer.bin");
        fs::write(&frontend_path, &frontend).unwrap();
        let template = b"chat-template".to_vec();
        let template_id = PayloadId::of(&template);
        let template_path = root.join("frontend-template.bin");
        fs::write(&template_path, &template).unwrap();
        let package = wire::PackageBody {
            format_version: schema::FORMAT_VERSION,
            package_id: vec![7; 32],
            model_identity: model.into(),
            hardware_identity: "sm120".into(),
            publisher: Some(wire::PublisherMetadata {
                namespace: publisher.namespace.clone(),
                official_source: "https://huggingface.co/example/model".into(),
                license_name: "Example".into(),
                ..Default::default()
            }),
            frontend: Some(Default::default()),
            sampling_defaults: Some(Default::default()),
            payloads: vec![
                descriptor(frontend_id, &frontend),
                descriptor(template_id, &template),
            ],
            assets: vec![
                wire::FrontendAsset {
                    name: "tokenizer".into(),
                    payload: Some(use_of(
                        frontend_id,
                        frontend.len() as u64,
                        wire::PayloadRole::Frontend,
                    )),
                    sha256: frontend_id.as_bytes().to_vec(),
                },
                wire::FrontendAsset {
                    name: "chat_template".into(),
                    payload: Some(use_of(
                        template_id,
                        template.len() as u64,
                        wire::PayloadRole::Frontend,
                    )),
                    sha256: template_id.as_bytes().to_vec(),
                },
            ],
            deployments: vec![],
        };
        let package_id = [tp as u8; 32];
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let request = PublishRequest {
            package,
            payloads: vec![
                PublishPayload {
                    role: wire::PayloadRole::Frontend,
                    relative: String::new(),
                    source: crate::package::export::pin_file(&frontend_path, frontend.len() as u64)
                        .unwrap(),
                    id: frontend_id,
                    tensor: Some("frontend.tokenizer".into()),
                },
                PublishPayload {
                    role: wire::PayloadRole::Frontend,
                    relative: String::new(),
                    source: crate::package::export::pin_file(&template_path, template.len() as u64)
                        .unwrap(),
                    id: template_id,
                    tensor: Some("frontend.chat-template".into()),
                },
            ],
            deployments: vec![PublishDeployment {
                body: wire::DeploymentBody {
                    deployment_id: format!("sm120-tp{tp}"),
                    architecture: vec![12, 0],
                    tensor_parallel: tp,
                    contract: Some(Default::default()),
                    engram: engram_body,
                    rebuild,
                    ..Default::default()
                },
                ranks,
                payloads: deployment_payloads,
            }],
            package_id,
            public_key,
            accepted: publisher.namespace.clone(),
            no_weights: false,
        };
        crate::package::export::publish(&request, key, Destination::Directory(&root)).unwrap();
        root
    }

    #[test]
    fn promotes_payload_package_with_fresh_identity() {
        let scratch = Scratch::new("promote");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let matrix: Vec<u8> = (10..18).collect();
        let full: Vec<u8> = (0..8).collect();
        let source = publish_source(
            &scratch,
            "promote-source",
            1,
            &[full.as_slice()],
            &[matrix.as_slice()],
            &key,
            &publisher,
            "model",
            false,
        );
        let before = Metadata::open(&source).unwrap();
        let output = scratch.join("promoted");
        let promoted = promote(
            &source,
            &output,
            std::slice::from_ref(&publisher),
            "example",
            "example",
            None,
            &key,
        )
        .unwrap();
        assert!(!promoted.payloads.is_empty());
        let after = Metadata::open(&output).unwrap();
        let before_envelope = protect::envelope(&before.envelope).unwrap();
        let after_envelope = protect::envelope(&after.envelope).unwrap();
        assert_ne!(before_envelope.package_id, after_envelope.package_id);
        assert_eq!(
            before_envelope.package_body.len(),
            after_envelope.package_body.len()
        );
        let verified = protect::verify(
            &after.envelope,
            std::slice::from_ref(&publisher),
            Selection::Deployment {
                deployment: 0,
                rank: None,
            },
        )
        .unwrap();
        let authorized = verified.authorize("example").unwrap();
        // Payload bytes survive promotion unchanged.
        let canonical = PayloadId::of(&full);
        let catalog = authorized.verified().catalog().clone();
        let files = after
            .payloads(&catalog, [canonical.as_bytes().as_slice()])
            .unwrap();
        files.verify().unwrap();
        let mut bytes = vec![0u8; full.len()];
        files
            .get(canonical.as_bytes())
            .unwrap()
            .read_at(0, &mut bytes)
            .unwrap();
        assert_eq!(bytes, full);
        // Executables are re-compressed under the fresh identity and still decode.
        let mut before_ids = Vec::new();
        {
            let before_verified = protect::verify(
                &before.envelope,
                std::slice::from_ref(&publisher),
                Selection::Deployment {
                    deployment: 0,
                    rank: None,
                },
            )
            .unwrap();
            for resource in &before_verified.ranks()[0].resources {
                before_ids.push(resource.stored.as_ref().unwrap().payload_id.clone());
            }
        }
        let rank = &authorized.verified().ranks()[0];
        assert_eq!(rank.resources.len(), before_ids.len());
        for (resource, _before_id) in rank.resources.iter().zip(before_ids) {
            let slice = resource.stored.as_ref().unwrap();
            let payloads = after
                .payloads(&catalog, [slice.payload_id.as_slice()])
                .unwrap();
            let payload = payloads.get(&slice.payload_id).unwrap();
            let mut stored = vec![0u8; slice.nbytes as usize];
            payload.read_at(slice.offset, &mut stored).unwrap();
            let plaintext = authorized.open_resource(resource, &stored).unwrap();
            assert!(!plaintext.is_empty());
        }
    }

    fn publish_split_owner_source(
        scratch: &Scratch,
        key: &signature::Ed25519KeyPair,
        publisher: &Publisher,
        target_b_pieces: bool,
        alt_target: bool,
        interior_gap: bool,
        duplicate_target: bool,
    ) -> (PathBuf, PayloadId, PayloadId, Vec<u8>) {
        use crate::package::export::{
            Destination, PublishDeployment, PublishPayload, PublishRank, PublishRequest,
            PublishResource,
        };
        let root = scratch.join("split-owner-source");
        fs::create_dir_all(&root).unwrap();
        let source = [vec![0xff], (0..16u8).collect::<Vec<_>>()].concat();
        let mut target = vec![source[1], source[2], source[3], source[4], source[1], source[2], source[5], source[6]];
        if interior_gap {
            target[2] ^= 0x40;
            target[3] ^= 0x40;
        }
        let target_alt = {
            let mut bytes = target.clone();
            if alt_target {
                bytes[6] ^= 0x5a;
            }
            bytes
        };
        let source_id = PayloadId::of(&source);
        let target_id = PayloadId::of(&target);
        let target_alt_id = if duplicate_target {
            target_id
        } else {
            PayloadId::of(&target_alt)
        };
        let source_path = root.join("source.bin");
        let target_path = root.join("target.bin");
        let target_alt_path = root.join("target-alt.bin");
        fs::write(&source_path, &source).unwrap();
        fs::write(&target_path, &target).unwrap();
        fs::write(&target_alt_path, &target_alt).unwrap();
        let graph = root.join("graph.bin");
        fs::write(&graph, b"split-graph").unwrap();
        let payload = |id: PayloadId, path: &Path, nbytes: u64, relative: &str| PublishPayload {
            role: wire::PayloadRole::Weights,
            relative: relative.into(),
            source: crate::package::export::pin_file(path, nbytes).unwrap(),
            id,
            tensor: None,
        };
        let use_ = |id: PayloadId, offset: u64, nbytes: u64| {
            Some(wire::PayloadUse {
                slice: Some(wire::PayloadSlice {
                    payload_id: id.as_bytes().to_vec(),
                    offset,
                    nbytes,
                }),
                role: wire::PayloadRole::Weights as i32,
            })
        };
        let weight_source = |tensor: &str, pieces: Vec<wire::WeightPiece>| wire::WeightSource {
            pieces,
            piece_encoding: String::new(),
            tensor: tensor.into(),
            ranges: vec![],
            concatenation_axis: 0,
        };
        let view = |name: &str, offset: u64, shape: Vec<u64>, stride: Vec<u64>, sources: Vec<wire::WeightSource>| wire::PreparedWeightView {
            name: name.into(),
            group: "layer".into(),
            dtype: "uint8".into(),
            element_bytes: 1,
            nbytes: shape.iter().product(),
            shape,
            stride,
            layout_type: "packed".into(),
            allocation: 0,
            offset,
            packing: None,
            sources,
            statistics_domain: String::new(),
            padding: vec![],
        };
        let rank = |deployment: &str, number: u32, payload_id: PayloadId, offsets: &[u64], source_offsets: &[u64], nbytes: &[u64], allocation_nbytes: u64, views: Vec<wire::PreparedWeightView>| {
            let mut initializers = Vec::new();
            for ((offset, source_offset), nbytes) in offsets.iter().zip(source_offsets).zip(nbytes) {
                initializers.push(wire::Initializer {
                    allocation: "weights".into(),
                    destination_offset: *offset,
                    source: use_(payload_id, *source_offset, *nbytes),
                });
            }
            PublishRank {
                body: wire::RankBody {
                    rank: number,
                    deployment_id: deployment.into(),
                    context: Some(runtime::Context::default()),
                    allocations: vec![wire::Allocation {
                        name: "weights".into(),
                        nbytes: allocation_nbytes,
                        alignment: 4096,
                        ..Default::default()
                    }],
                    initializers,
                    graphs: vec![wire::Graph {
                        name: "decode".into(),
                        resource: 0,
                        node_count: 1,
                        ..Default::default()
                    }],
                    prepared_weights: Some(wire::PreparedWeightCatalog {
                        version: 1,
                        views,
                        unavailable: vec![],
                    }),
                    ..Default::default()
                },
                payloads: {
                    let relative = if payload_id == source_id {
                        "weights/source.safetensors".to_owned()
                    } else {
                        format!("weights/{deployment}/target-{number}.safetensors")
                    };
                    let path = if payload_id == source_id {
                        &source_path
                    } else if payload_id == target_alt_id {
                        &target_alt_path
                    } else {
                        &target_path
                    };
                    let nbytes = if payload_id == source_id {
                        source.len() as u64
                    } else if payload_id == target_alt_id {
                        target_alt.len() as u64
                    } else {
                        target.len() as u64
                    };
                    vec![payload(payload_id, path, nbytes, &relative)]
                },
                resources: vec![PublishResource {
                    kind: wire::ExecutableKind::Graph,
                    path: graph.clone(),
                    nbytes: 10,
                }],
            }
        };
        let contract = runtime::ServingContract {
            model_id: "split-owner".into(),
            identity: "split-owner".into(),
            model_contract: Some(runtime::ModelContract::default()),
            ..Default::default()
        };
        let request = PublishRequest {
            package: wire::PackageBody {
                model_identity: "split-owner".into(),
                hardware_identity: "sm120".into(),
                frontend: Some(Default::default()),
                sampling_defaults: Some(Default::default()),
                publisher: Some(wire::PublisherMetadata {
                    namespace: publisher.namespace.clone(),
                    official_source: "https://huggingface.co/example/split".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            payloads: vec![],
            deployments: {
                let mut deployments = vec![
                    PublishDeployment {
                    body: wire::DeploymentBody {
                        deployment_id: "sm120-tp1".into(),
                        architecture: vec![12, 0],
                        tensor_parallel: 1,
                        contract: Some(contract.clone()),
                        ..Default::default()
                    },
                    payloads: vec![],
                    ranks: vec![rank("sm120-tp1", 0, source_id, &[0, 8], &[1, 1], &[8, 8], 16, vec![
                            view("a", 0, vec![2, 4], vec![4, 1], vec![
                                weight_source("a0", vec![wire::WeightPiece { source_offset: 1, destination_offset: 0, contiguous_bytes: 4, extents: vec![], source_strides: vec![], destination_strides: vec![] }]),
                                weight_source("a1", vec![wire::WeightPiece { source_offset: 5, destination_offset: 4, contiguous_bytes: 4, extents: vec![], source_strides: vec![], destination_strides: vec![] }]),
                            ]),
                            view("b", 8, vec![2, 4], vec![4, 1], vec![
                                weight_source("b0", vec![wire::WeightPiece { source_offset: 1, destination_offset: 0, contiguous_bytes: 4, extents: vec![], source_strides: vec![], destination_strides: vec![] }]),
                                weight_source("b1", vec![wire::WeightPiece { source_offset: 5, destination_offset: 4, contiguous_bytes: 4, extents: vec![], source_strides: vec![], destination_strides: vec![] }]),
                            ]),
                        ])],
                },
                PublishDeployment {
                    body: wire::DeploymentBody {
                        deployment_id: "sm120-tp2".into(),
                        architecture: vec![12, 0],
                        tensor_parallel: 2,
                        contract: Some(contract.clone()),
                        ..Default::default()
                    },
                    payloads: vec![],
                    ranks: vec![
                        rank("sm120-tp2", 0, target_id, &[32, 36], &[0, 4], &[4, 4], 48, vec![
                            view("a", 32, vec![1, 4], vec![4, 1], if interior_gap {
                                vec![weight_source("a0", vec![wire::WeightPiece { source_offset: 1, destination_offset: 0, contiguous_bytes: 2, extents: vec![], source_strides: vec![], destination_strides: vec![] }])]
                            } else {
                                vec![
                                    weight_source("a0", vec![wire::WeightPiece { source_offset: 1, destination_offset: 0, contiguous_bytes: 4, extents: vec![], source_strides: vec![], destination_strides: vec![] }]),
                                ]
                            }),
                            view("b", 36, vec![2, 2], vec![2, 1], if interior_gap {
                                vec![weight_source("b0", vec![wire::WeightPiece { source_offset: 1, destination_offset: 0, contiguous_bytes: 2, extents: vec![], source_strides: vec![], destination_strides: vec![] }])]
                            } else if target_b_pieces {
                                vec![
                                    weight_source("b0", vec![wire::WeightPiece { source_offset: 1, destination_offset: 0, contiguous_bytes: 2, extents: vec![], source_strides: vec![], destination_strides: vec![] }]),
                                    weight_source("b1", vec![wire::WeightPiece { source_offset: 5, destination_offset: 2, contiguous_bytes: 2, extents: vec![], source_strides: vec![], destination_strides: vec![] }]),
                                ]
                            } else {
                                vec![]
                            }),
                        ]),
                        rank("sm120-tp2", 1, target_id, &[64, 68], &[0, 4], &[4, 4], 80, vec![
                            view("a", 64, vec![1, 4], vec![4, 1], if interior_gap {
                                vec![weight_source("a0", vec![wire::WeightPiece { source_offset: 1, destination_offset: 0, contiguous_bytes: 2, extents: vec![], source_strides: vec![], destination_strides: vec![] }])]
                            } else {
                                vec![
                                    weight_source("a0", vec![wire::WeightPiece { source_offset: 1, destination_offset: 0, contiguous_bytes: 4, extents: vec![], source_strides: vec![], destination_strides: vec![] }]),
                                ]
                            }),
                            view("b", 68, vec![2, 2], vec![2, 1], if interior_gap {
                                vec![weight_source("b0", vec![wire::WeightPiece { source_offset: 1, destination_offset: 0, contiguous_bytes: 2, extents: vec![], source_strides: vec![], destination_strides: vec![] }])]
                            } else if target_b_pieces {
                                vec![
                                    weight_source("b0", vec![wire::WeightPiece { source_offset: 1, destination_offset: 0, contiguous_bytes: 2, extents: vec![], source_strides: vec![], destination_strides: vec![] }]),
                                    weight_source("b1", vec![wire::WeightPiece { source_offset: 5, destination_offset: 2, contiguous_bytes: 2, extents: vec![], source_strides: vec![], destination_strides: vec![] }]),
                                ]
                            } else {
                                vec![]
                            }),
                        ]),
                    ],
                },
                ];
                if alt_target || duplicate_target {
                    deployments.push(PublishDeployment {
                        body: wire::DeploymentBody {
                            deployment_id: "sm120-tp2-alt".into(),
                            architecture: vec![12, 0],
                            tensor_parallel: 2,
                            contract: Some(contract),
                            ..Default::default()
                        },
                        payloads: vec![],
                        ranks: vec![
                            rank("sm120-tp2-alt", 0, target_alt_id, &[96, 100], &[0, 4], &[4, 4], 112, vec![
                                view("a", 96, vec![1, 4], vec![4, 1], vec![
                                    weight_source("a0", vec![wire::WeightPiece { source_offset: 1, destination_offset: 0, contiguous_bytes: 4, extents: vec![], source_strides: vec![], destination_strides: vec![] }]),
                                ]),
                                view("b", 100, vec![2, 2], vec![2, 1], if duplicate_target {
                                    vec![
                                        weight_source("b0", vec![wire::WeightPiece { source_offset: 1, destination_offset: 0, contiguous_bytes: 2, extents: vec![], source_strides: vec![], destination_strides: vec![] }]),
                                        weight_source("b1", vec![wire::WeightPiece { source_offset: 5, destination_offset: 2, contiguous_bytes: 2, extents: vec![], source_strides: vec![], destination_strides: vec![] }]),
                                    ]
                                } else {
                                    vec![]
                                }),
                            ]),
                            rank("sm120-tp2-alt", 1, target_alt_id, &[128, 132], &[0, 4], &[4, 4], 144, vec![
                                view("a", 128, vec![1, 4], vec![4, 1], vec![
                                    weight_source("a0", vec![wire::WeightPiece { source_offset: 1, destination_offset: 0, contiguous_bytes: 4, extents: vec![], source_strides: vec![], destination_strides: vec![] }]),
                                ]),
                                view("b", 132, vec![2, 2], vec![2, 1], if duplicate_target {
                                    vec![
                                        weight_source("b0", vec![wire::WeightPiece { source_offset: 1, destination_offset: 0, contiguous_bytes: 2, extents: vec![], source_strides: vec![], destination_strides: vec![] }]),
                                        weight_source("b1", vec![wire::WeightPiece { source_offset: 5, destination_offset: 2, contiguous_bytes: 2, extents: vec![], source_strides: vec![], destination_strides: vec![] }]),
                                    ]
                                } else {
                                    vec![]
                                }),
                            ]),
                        ],
                    });
                }
                deployments
            },
            package_id: [61u8; 32],
            public_key: key.public_key().as_ref().try_into().unwrap(),
            accepted: publisher.namespace.clone(),
            no_weights: false,
        };
        crate::package::export::publish(&request, key, Destination::Directory(&root)).unwrap();
        (root, source_id, target_id, target)
    }

    fn publish_chain_source(
        scratch: &Scratch,
        key: &signature::Ed25519KeyPair,
        publisher: &Publisher,
        weights_initializer: bool,
    ) -> PathBuf {
        use crate::package::export::{
            Destination, PublishDeployment, PublishPayload, PublishRank, PublishRequest,
            PublishResource,
        };
        let root = scratch.join("chain-source");
        fs::create_dir_all(&root).unwrap();
        let weights: Vec<u8> = (0..16).collect();
        let middle = weights[0..8].to_vec();
        let final_bytes = middle[0..4].to_vec();
        let graph = root.join("graph.bin");
        fs::write(&graph, b"graph-chain").unwrap();
        let mut payloads = Vec::new();
        for (index, (name, bytes, role)) in [
            (
                "weights/chain-w.safetensors",
                &weights,
                wire::PayloadRole::Weights,
            ),
            (
                "weights/chain-b.safetensors",
                &middle,
                wire::PayloadRole::Weights,
            ),
            (
                "weights/chain-c.safetensors",
                &final_bytes,
                wire::PayloadRole::Weights,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let path = root.join(format!("payload-{index}.bin"));
            fs::write(&path, bytes).unwrap();
            payloads.push(PublishPayload {
                role,
                relative: name.into(),
                source: crate::package::export::pin_file(&path, bytes.len() as u64).unwrap(),
                id: PayloadId::of(bytes),
                tensor: None,
            });
        }
        let rank = wire::RankBody {
            rank: 0,
            deployment_id: "sm120-tp1".into(),
            context: Some(runtime::Context::default()),
            allocations: vec![wire::Allocation {
                name: "weights".into(),
                nbytes: 16384,
                alignment: 4096,
                address: Some(0x1000_0000),
                memory: runtime::Memory::Device as i32,
                sizing: None,
                relocations: vec![],
            }],
            initializers: {
                let mut initializers = Vec::new();
                if weights_initializer {
                    initializers.push(wire::Initializer {
                        allocation: "weights".into(),
                        destination_offset: 0,
                        source: Some(use_of(
                            PayloadId::of(&weights),
                            weights.len() as u64,
                            wire::PayloadRole::Weights,
                        )),
                    });
                }
                initializers.push(wire::Initializer {
                    allocation: "weights".into(),
                    destination_offset: 12288,
                    source: Some(use_of(
                        PayloadId::of(&final_bytes),
                        final_bytes.len() as u64,
                        wire::PayloadRole::Weights,
                    )),
                });
                initializers
            },
            graphs: vec![wire::Graph {
                name: "decode".into(),
                resource: 0,
                node_count: 1,
                ..Default::default()
            }],
            ..Default::default()
        };
        let request = PublishRequest {
            package: wire::PackageBody {
                format_version: schema::FORMAT_VERSION,
                package_id: vec![11; 32],
                model_identity: "chain".into(),
                hardware_identity: "sm120".into(),
                publisher: Some(wire::PublisherMetadata {
                    namespace: publisher.namespace.clone(),
                    official_source: "https://huggingface.co/example/model".into(),
                    license_name: "Example".into(),
                    ..Default::default()
                }),
                frontend: Some(Default::default()),
                sampling_defaults: Some(Default::default()),
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
                    contract: Some(Default::default()),
                    ..Default::default()
                },
                ranks: vec![PublishRank {
                    body: rank,
                    payloads,
                    resources: vec![PublishResource {
                        kind: wire::ExecutableKind::Graph,
                        path: graph,
                        nbytes: "graph-chain".len() as u64,
                    }],
                }],
                payloads: vec![],
            }],
            package_id: [11; 32],
            public_key: key.public_key().as_ref().try_into().unwrap(),
            accepted: publisher.namespace.clone(),
            no_weights: false,
        };
        crate::package::export::publish(&request, key, Destination::Directory(&root)).unwrap();
        root
    }

    fn prepared_owner(statistics_domain: &str) -> PreparedOwner {
        // Canonical owner: two rows from the checkpoint, sharing one payload.

        let payload = PayloadId::of(b"owner bytes");
        PreparedOwner {
            name: "layer-000.q".into(),
            group: "layer-000".into(),
            allocation: "device-0".into(),
            allocation_offset: 0,
            nbytes: 16,
            element_bytes: 2,
            shape: vec![2, 4],
            stride: vec![4, 1],
            packing: Some(wire::PackingParameters {
                format: "nvfp4".into(),
                ..Default::default()
            }),
            statistics_domain: statistics_domain.into(),
            padding: Vec::new(),
            sources: vec![wire::WeightSource {
                pieces: vec![],
                piece_encoding: String::new(),
                tensor: "checkpoint.q".into(),
                ranges: vec![
                    wire::TensorRange { begin: 0, end: 2 },
                    wire::TensorRange { begin: 0, end: 4 },
                ],
                concatenation_axis: 0,
            }],
            source: Some(OwnerSource {
                payload,
                offset: 0,
                nbytes: 16,
            }),
            segments: Vec::new(),
        }
    }

    /// One row subset of `prepared_owner`, so the copy candidate path is used.
    fn row_subset(statistics_domain: &str) -> PreparedOwner {
        let mut owner = prepared_owner(statistics_domain);
        owner.shape = vec![1, 4];
        owner.nbytes = 8;
        owner.sources[0].ranges = vec![
            wire::TensorRange { begin: 1, end: 2 },
            wire::TensorRange { begin: 0, end: 4 },
        ];
        owner.source = Some(OwnerSource {
            payload: PayloadId::of(b"owner bytes"),
            offset: 8,
            nbytes: 8,
        });
        owner.segments = Vec::new();
        owner
    }

    fn single_rank(rank: u32, owner: PreparedOwner, slice: container::Slice) -> PreparedRank {
        let payload = owner.source.map(|source| source.payload).unwrap();
        PreparedRank {
            rank,
            owners: vec![owner],
            payloads: BTreeMap::from([(payload, slice)]),
        }
    }

    #[test]
    fn owner_source_requires_one_exact_indexed_segment() {
        let payload = PayloadId::of(b"owner-source");
        let view = |offset, nbytes| wire::PreparedWeightView {
            offset,
            nbytes,
            ..Default::default()
        };
        let source = |entries: &[(u64, u64)], offset, nbytes| {
            let entries: Vec<_> = entries
                .iter()
                .map(|(begin, end)| (*begin, *end, payload, 100))
                .collect();
            let segments = mapped_owner_segments(&view(offset, nbytes), Some(&entries));
            owner_source(&segments, nbytes)
        };

        assert_eq!(
            source(&[(0, 16)], 0, 16),
            Some(OwnerSource {
                payload,
                offset: 100,
                nbytes: 16,
            })
        );
        assert_eq!(
            source(&[(0, 16)], 4, 8),
            Some(OwnerSource {
                payload,
                offset: 104,
                nbytes: 8,
            })
        );
        assert_eq!(source(&[(0, 8), (8, 16)], 0, 16), None);
        assert_eq!(source(&[(0, 4), (8, 16)], 0, 16), None);
        assert_eq!(source(&[(0, 8)], 4, 12), None);
        assert_eq!(source(&[], 0, 16), None);
    }

    #[test]
    fn shared_payload_with_distinct_owner_offsets_is_not_a_self_derivation() {
        let scratch = Scratch::new("shared-self-edge");
        let bytes = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let raw = scratch.0.join("shared.bin");
        fs::write(&raw, bytes).unwrap();
        let payload = PayloadId::of(&bytes);
        let owner = |offset| PreparedOwner {
            name: "vision.shared".into(),
            group: "vision".into(),
            allocation: "device-0".into(),
            allocation_offset: 0,
            nbytes: 4,
            element_bytes: 1,
            shape: vec![4],
            stride: vec![1],
            packing: Some(wire::PackingParameters {
                format: "uint8".into(),
                ..Default::default()
            }),
            statistics_domain: "tensor".into(),
            padding: Vec::new(),
            sources: vec![wire::WeightSource {
                pieces: Vec::new(),
                piece_encoding: String::new(),
                tensor: "vision".into(),
                ranges: vec![wire::TensorRange { begin: 0, end: 4 }],
                concatenation_axis: 0,
            }],
            source: Some(OwnerSource {
                payload,
                offset,
                nbytes: 4,
            }),
            segments: Vec::new(),
        };
        let slice = || {
            container::Slice::new(
                std::sync::Arc::new(fs::File::open(&raw).unwrap()),
                0,
                bytes.len() as u64,
            )
            .unwrap()
        };
        let canonical = single_rank(0, owner(0), slice());
        let derived = single_rank(1, owner(4), slice());
        let plan = plan_rank_against(
            std::slice::from_ref(&canonical),
            &derived,
            &derivation::Limits::default(),
        )
        .unwrap();
        assert_eq!(plan.owners["vision.shared"], OwnerShare::Independent);
        assert!(plan.recipes.is_empty());
    }

    #[test]
    fn planned_derivations_reject_genuine_cycles() {
        let a = PayloadId::of(b"cycle-a");
        let b = PayloadId::of(b"cycle-b");
        let recipe = |target, source| derivation::Derivation {
            target,
            length: 1,
            operations: vec![derivation::Operation::Copy(derivation::Copy {
                source,
                source_offset: 0,
                destination_offset: 0,
                contiguous: 1,
                outer: Vec::new(),
            })],
        };
        let error = validate_planned_derivations(&[recipe(a, b), recipe(b, a)]).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("planned derivation set"), "{message}");
        assert!(message.contains("edges=2"), "{message}");
        assert!(message.contains(&a.to_string()), "{message}");
        assert!(message.contains(&b.to_string()), "{message}");
    }

    #[test]
    fn differing_statistic_domains_block_affine_candidates_only() {
        let scratch = Scratch::new("statistic-domain");
        let bytes = (0..16u8).collect::<Vec<u8>>();
        let raw = scratch.0.join("owner.bin");
        fs::write(&raw, &bytes).unwrap();
        let slice = || {
            container::Slice::new(
                std::sync::Arc::new(fs::File::open(&raw).unwrap()),
                0,
                bytes.len() as u64,
            )
            .unwrap()
        };
        let canonical = single_rank(0, prepared_owner("tensor"), slice());
        let derived = single_rank(1, row_subset("shard"), slice());
        let plan = plan_rank(&canonical, &derived, &derivation::Limits::default()).unwrap();
        assert_eq!(plan.owners["layer-000.q"], OwnerShare::Independent);

        // The same relation with matching domains still plans a shared copy.
        let canonical = single_rank(0, prepared_owner("tensor"), slice());
        let derived = single_rank(1, row_subset("tensor"), slice());
        let plan = plan_rank(&canonical, &derived, &derivation::Limits::default()).unwrap();
        assert_eq!(plan.owners["layer-000.q"], OwnerShare::Independent);
        assert!(
            plan.recipes.is_empty(),
            "a partial owner must not claim a larger target payload"
        );

        // A catalog exported before the field existed has an empty domain. The
        // owner still occupies only part of the larger target payload, so it must
        // remain independently stored rather than claim an incomplete target.
        let canonical = single_rank(0, prepared_owner(""), slice());
        let derived = single_rank(1, row_subset("tensor"), slice());
        let plan = plan_rank(&canonical, &derived, &derivation::Limits::default()).unwrap();
        assert_eq!(plan.owners["layer-000.q"], OwnerShare::Independent);
        assert!(plan.recipes.is_empty());
    }

    fn combine_chain_case(
        scratch: &Scratch,
        second_operations: Vec<derivation::Operation>,
    ) {
        use crate::package::cache;
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let directory = publish_chain_source(scratch, &key, &publisher, false);
        let weights = PayloadId::of(&(0..16u8).collect::<Vec<u8>>());
        let middle = PayloadId::of(&(0..8u8).collect::<Vec<u8>>());
        let final_id = PayloadId::of(&(0..4u8).collect::<Vec<u8>>());
        let first = derivation::Derivation {
            target: middle,
            length: 8,
            operations: vec![derivation::Operation::Copy(derivation::Copy {
                source: weights,
                source_offset: 0,
                destination_offset: 0,
                contiguous: 8,
                outer: Vec::new(),
            })],
        };
        let second = derivation::Derivation {
            target: final_id,
            length: 4,
            operations: second_operations,
        };
        emit_derivations(&directory, &[first], &derivation::Limits::default()).unwrap();
        emit_derivations(&directory, &[second], &derivation::Limits::default()).unwrap();
        let first_output = scratch.join("combine-chain-output");
        combine(&CombineRequest {
            sources: std::slice::from_ref(&directory),
            output: &first_output,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap();
        let output = scratch.join("recombined-chain-output");
        combine(&CombineRequest {
            sources: std::slice::from_ref(&first_output),
            output: &output,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap();
        let metadata = Metadata::open(&output).unwrap();
        let set = wire::DerivationSet::decode(
            tensor_bytes(metadata.primary(), reader::DERIVATIONS).as_slice(),
        )
        .unwrap();
        assert_eq!(set.derivations.len(), 2);
        assert!(
            metadata
                .locations
                .locations
                .iter()
                .all(|location| location.payload_id != middle.as_bytes().to_vec()
                    && location.payload_id != final_id.as_bytes().to_vec()),
            "intermediate or final target remains physically installed"
        );

        let selection = protect::Selection::Deployment {
            deployment: 0,
            rank: None,
        };
        let verified = protect::verify(
            &metadata.envelope,
            std::slice::from_ref(&publisher),
            selection,
        )
        .unwrap();
        let initializer = verified
            .ranks()
            .first()
            .and_then(|rank| {
                rank.initializers.iter().find(|initializer| {
                    initializer
                        .source
                        .as_ref()
                        .and_then(|use_| use_.slice.as_ref())
                        .is_some_and(|slice| slice.payload_id == final_id.as_bytes().to_vec())
                })
            })
            .expect("combined deployment lacks the final initializer binding");
        let initializer_slice = initializer
            .source
            .as_ref()
            .and_then(|use_| use_.slice.as_ref())
            .unwrap();
        assert_eq!(initializer_slice.offset, 0);
        assert_eq!(initializer_slice.nbytes, 4);

        let store = cache::Root::open(scratch.join("chain-cache")).unwrap();
        let source = cache::PackageSource::open(
            &output,
            std::slice::from_ref(&publisher),
            selection,
        )
        .unwrap();
        let entry = store.install_package(source, "chain", "entry").unwrap();
        let installed = Metadata::open(entry.directory()).unwrap();
        let location = installed
            .locations
            .locations
            .iter()
            .find(|location| location.payload_id == final_id.as_bytes().to_vec())
            .expect("installed entry lacks the final target location");
        let slice = entry.payload(&location.file).unwrap();
        let mut bytes = vec![0; slice.len() as usize];
        slice.read_at(0, &mut bytes).unwrap();
        assert_eq!(
            bytes,
            (0..4u8).collect::<Vec<_>>(),
            "installed initializer bytes differ from the signed target"
        );
    }

    #[test]
    fn planner_aggregates_split_owner_segments_into_full_target_payloads() {
        let scratch = Scratch::new("split-owners");
        let source_bytes = (0..32u8).collect::<Vec<u8>>();
        let target_bytes = source_bytes[..16].to_vec();
        let source_path = scratch.0.join("source.bin");
        let target_path = scratch.0.join("target.bin");
        fs::write(&source_path, &source_bytes).unwrap();
        fs::write(&target_path, &target_bytes).unwrap();
        let source_id = PayloadId::of(&source_bytes);
        let target_id = PayloadId::of(&target_bytes);
        let slice = |path: &Path, nbytes: u64| {
            container::Slice::new(
                std::sync::Arc::new(fs::File::open(path).unwrap()),
                0,
                nbytes,
            )
            .unwrap()
        };
        let owner = |name: &str, offset: u64, nbytes: u64, segments: Vec<OwnerSegment>| PreparedOwner {
            name: name.into(),
            group: name.into(),
            allocation: "device-0".into(),
            allocation_offset: offset,
            nbytes,
            element_bytes: 1,
            shape: vec![nbytes],
            stride: vec![1],
            packing: None,
            statistics_domain: String::new(),
            padding: vec![],
            sources: vec![wire::WeightSource {
                pieces: vec![],
                piece_encoding: String::new(),
                tensor: "weight".into(),
                ranges: vec![wire::TensorRange { begin: 0, end: nbytes }],
                concatenation_axis: 0,
            }],
            source: Some(OwnerSource {
                payload: source_id,
                offset,
                nbytes,
            }),
            segments,
        };
        let canonical = PreparedRank {
            rank: 0,
            owners: vec![
                owner(
                    "a",
                    0,
                    8,
                    vec![
                        OwnerSegment { owner_offset: 0, payload: source_id, offset: 0, nbytes: 3 },
                        OwnerSegment { owner_offset: 3, payload: source_id, offset: 3, nbytes: 5 },
                    ],
                ),
                owner(
                    "b",
                    8,
                    8,
                    vec![OwnerSegment { owner_offset: 0, payload: source_id, offset: 8, nbytes: 8 }],
                ),
            ],
            payloads: BTreeMap::from([(source_id, slice(&source_path, source_bytes.len() as u64))]),
        };
        let mut target_a = owner(
            "a",
            0,
            8,
            vec![
                OwnerSegment { owner_offset: 0, payload: target_id, offset: 0, nbytes: 5 },
                OwnerSegment { owner_offset: 5, payload: target_id, offset: 5, nbytes: 3 },
            ],
        );
        target_a.source = None;
        let mut target_b = owner(
            "b",
            8,
            8,
            vec![OwnerSegment { owner_offset: 0, payload: target_id, offset: 8, nbytes: 8 }],
        );
        target_b.source = None;
        let derived = PreparedRank {
            rank: 0,
            owners: vec![target_a, target_b],
            payloads: BTreeMap::from([(target_id, slice(&target_path, target_bytes.len() as u64))]),
        };
        let mut wrong = canonical.clone();
        wrong.owners[0].shape = vec![4];
        wrong.owners[0].sources[0].ranges[0].end = 4;
        let plan = plan_rank_against(
            &[wrong, canonical.clone()],
            &derived,
            &derivation::Limits::default(),
        )
        .unwrap();
        assert_eq!(plan.owners["a"], OwnerShare::Mapped);
        assert_eq!(plan.owners["b"], OwnerShare::Mapped);
        assert_eq!(plan.recipes.len(), 1);
        let recipe = plan.recipes.values().next().unwrap();
        assert_eq!(recipe.target, target_id);
        assert_eq!(recipe.length, 16);
        assert_eq!(recipe.operations.len(), 2);

        let mut gap_a = owner(
            "a",
            0,
            8,
            vec![OwnerSegment { owner_offset: 0, payload: target_id, offset: 0, nbytes: 4 }],
        );
        gap_a.source = None;
        let mut gap_b = owner(
            "b",
            8,
            8,
            vec![OwnerSegment { owner_offset: 0, payload: target_id, offset: 8, nbytes: 8 }],
        );
        gap_b.source = None;
        let gap = PreparedRank {
            rank: 0,
            owners: vec![gap_a, gap_b],
            payloads: BTreeMap::from([(target_id, slice(&target_path, target_bytes.len() as u64))]),
        };
        let gap_plan = plan_rank_against(
            &[canonical.clone()],
            &gap,
            &derivation::Limits::default(),
        )
        .unwrap();
        assert!(gap_plan.recipes.is_empty(), "uncovered target bytes must remain independent");

        let mut overlap_a = owner(
            "a",
            0,
            8,
            vec![OwnerSegment { owner_offset: 0, payload: target_id, offset: 0, nbytes: 10 }],
        );
        overlap_a.source = None;
        let mut overlap_b = owner(
            "b",
            8,
            8,
            vec![OwnerSegment { owner_offset: 0, payload: target_id, offset: 4, nbytes: 8 }],
        );
        overlap_b.source = None;
        let overlap = PreparedRank {
            rank: 0,
            owners: vec![overlap_a, overlap_b],
            payloads: BTreeMap::from([(target_id, slice(&target_path, target_bytes.len() as u64))]),
        };
        let overlap_plan = plan_rank_against(
            &[canonical],
            &overlap,
            &derivation::Limits::default(),
        )
        .unwrap();
        assert!(overlap_plan.recipes.is_empty(), "overlapping target bytes must remain independent");
    }

    #[test]
    fn partial_owner_target_uses_interior_gaps_and_a_small_signed_exception_payload() {
        use crate::package::cache;
        let scratch = Scratch::new("exception-payload");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let (source, source_id, target_id, target) =
            publish_split_owner_source(&scratch, &key, &publisher, false, false, true, false);
        let output = scratch.join("exception-combined");
        combine(&CombineRequest {
            sources: std::slice::from_ref(&source),
            output: &output,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap();
        let metadata = Metadata::open(&output).unwrap();
        assert!(
            metadata
                .locations
                .locations
                .iter()
                .all(|location| location.payload_id != target_id.as_bytes().to_vec()),
            "target payload must be derived"
        );
        let set = metadata.derivations.as_ref().unwrap();
        let recipe = set
            .derivations
            .iter()
            .find(|recipe| recipe.target_payload_id == target_id.as_bytes())
            .expect("target recipe missing");
        assert!(recipe.operations.iter().any(|operation| {
            let wire::derivation_operation::Operation::Copy(copy) = operation.operation.as_ref().unwrap()
            else {
                return false;
            };
            copy.source_payload_id == source_id.as_bytes()
        }));
        let exception_locations: Vec<_> = metadata
            .locations
            .locations
            .iter()
            .filter(|location| location.file.contains("exceptions"))
            .collect();
        assert_eq!(exception_locations.len(), 1);
        let exception = &exception_locations[0];
        assert_ne!(exception.payload_id, target_id.as_bytes().to_vec());
        let exception_file = Container::open(&output.join(&exception.file)).unwrap();
        let exception_slice = exception_file.tensor(COMPANION_TENSOR).unwrap();
        assert!(exception_slice.len() < target.len() as u64);
        let mut exception_bytes = vec![0; exception_slice.len() as usize];
        exception_slice.read_at(0, &mut exception_bytes).unwrap();
        let mut expected_exception = target[2..4].to_vec();
        expected_exception.extend_from_slice(&target[6..8]);
        assert_eq!(exception_bytes, expected_exception);

        let selection = protect::Selection::Deployment { deployment: 1, rank: None };
        let store = cache::Root::open(scratch.join("exception-cache")).unwrap();
        let package = cache::PackageSource::open(&output, std::slice::from_ref(&publisher), selection).unwrap();
        let entry = store.install_package(package, "exception", "entry").unwrap();
        let installed = Metadata::open(entry.directory()).unwrap();
        let location = installed
            .locations
            .locations
            .iter()
            .find(|location| location.payload_id == target_id.as_bytes().to_vec())
            .expect("installed target missing");
        let slice = entry.payload(&location.file).unwrap();
        let mut bytes = vec![0; slice.len() as usize];
        slice.read_at(0, &mut bytes).unwrap();
        assert_eq!(bytes, target);
    }

    #[test]
    fn multi_deployment_exceptions_at_one_rank_keep_distinct_files() {
        let scratch = Scratch::new("exception-multi-deployment");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let (source, _, _, _) =
            publish_split_owner_source(&scratch, &key, &publisher, false, true, false, false);
        let output = scratch.join("exception-multi-combined");
        combine(&CombineRequest {
            sources: std::slice::from_ref(&source),
            output: &output,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap();
        let metadata = Metadata::open(&output).unwrap();
        let mut rank_zero: BTreeMap<String, (Vec<u8>, Vec<u8>)> = BTreeMap::new();
        for location in metadata
            .locations
            .locations
            .iter()
            .filter(|location| location.file.contains("/rank-0/") && location.file.contains("exceptions"))
        {
            let deployment = location
                .file
                .split('/')
                .nth(2)
                .expect("exception deployment path")
                .to_owned();
            let container = Container::open(&output.join(&location.file)).unwrap();
            let slice = container.tensor(COMPANION_TENSOR).unwrap();
            let mut bytes = vec![0; slice.len() as usize];
            slice.read_at(0, &mut bytes).unwrap();
            rank_zero.insert(deployment, (location.payload_id.clone(), bytes));
        }
        assert_eq!(rank_zero.len(), 2, "both deployments need an exception payload");
        let tp2 = rank_zero.get("sm120-tp2").expect("tp2 exception missing");
        let alt = rank_zero.get("sm120-tp2-alt").expect("alt exception missing");
        assert_ne!(tp2.0, alt.0);
        assert_ne!(tp2.1, alt.1);
    }

    #[test]
    fn conflicting_target_keeps_independent_target_and_prunes_exception() {
        let scratch = Scratch::new("exception-conflict");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let (source, _, target_id, _) =
            publish_split_owner_source(&scratch, &key, &publisher, false, false, true, true);
        let output = scratch.join("exception-conflict-combined");
        combine(&CombineRequest {
            sources: std::slice::from_ref(&source),
            output: &output,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap();
        let metadata = Metadata::open(&output).unwrap();
        assert!(
            metadata
                .locations
                .locations
                .iter()
                .any(|location| location.payload_id == target_id.as_bytes().to_vec()),
            "conflicting target must remain independently stored"
        );
        assert!(
            metadata
                .locations
                .locations
                .iter()
                .all(|location| !location.file.contains("exceptions")),
            "unused exception descriptor was published"
        );
        assert!(
            metadata
                .derivations
                .as_ref()
                .is_none_or(|set| set
                    .derivations
                    .iter()
                    .all(|recipe| recipe.target_payload_id != target_id.as_bytes())),
            "conflicting target recipe was published"
        );
    }

    #[test]
    fn weight_piece_maps_reject_mismatched_overlapping_and_incomplete_maps() {
        let scratch = Scratch::new("piece-invalid");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let (source, _, _, _) = publish_split_owner_source(&scratch, &key, &publisher, true, false, false, false);
        let base = prepared_ranks(&source, 0).unwrap();
        let target = prepared_ranks(&source, 1).unwrap();
        let reject = |target: PreparedRank| {
            let plan = plan_rank(&base[0], &target, &derivation::Limits::default()).unwrap();
            assert!(
                plan.recipes.is_empty(),
                "invalid piece mapping must not claim a target payload"
            );
            assert!(
                !plan.owners.values().any(|share| matches!(share, OwnerShare::Mapped)),
                "invalid piece mapping was marked shareable"
            );
            target
        };

        let mut mismatch = target[0].clone();
        mismatch.owners[1].sources[0].pieces[0].source_offset = 2;
        reject(mismatch);

        let mut encoding = target[0].clone();
        encoding.owners[1].sources[0].piece_encoding = "w4a16_mma_k16_n64_v1".into();
        reject(encoding);

        let mut overlapping = target[0].clone();
        overlapping.owners[1].sources[0].pieces[0].source_strides = vec![1];
        reject(overlapping);

        let mut incomplete = target[0].clone();
        let piece = &mut incomplete.owners[1].sources[0].pieces[0];
        piece.contiguous_bytes = 1;
        piece.extents.clear();
        piece.source_strides.clear();
        piece.destination_strides.clear();
        reject(incomplete);
    }

    #[test]
    fn signed_split_owner_package_combines_and_installs_exact_initializer_bytes() {
        use crate::package::cache;
        let scratch = Scratch::new("split-owner-combine");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let (source, _source_id, target_id, target) =
            publish_split_owner_source(&scratch, &key, &publisher, true, false, false, false);
        let canonical = prepared_ranks(&source, 0).unwrap();
        let target_rank = prepared_ranks(&source, 1).unwrap();
        let wrong_path = scratch.0.join("wrong-source.bin");
        fs::write(&wrong_path, vec![0xaa; 17]).unwrap();
        let wrong_slice = container::Slice::new(
            std::sync::Arc::new(fs::File::open(&wrong_path).unwrap()),
            0,
            17,
        )
        .unwrap();
        let wrong_id = PayloadId::of(&vec![0xaa; 17]);
        let mut wrong = canonical[0].clone();
        wrong.payloads.clear();
        wrong.payloads.insert(wrong_id, wrong_slice);
        for owner in &mut wrong.owners {
            for segment in &mut owner.segments {
                segment.payload = wrong_id;
            }
            if let Some(source) = &mut owner.source {
                source.payload = wrong_id;
            }
        }
        let plan = plan_rank_against(
            &[wrong, canonical[0].clone()],
            &target_rank[0],
            &derivation::Limits::default(),
        )
        .unwrap();
        assert_eq!(plan.owners["a"], OwnerShare::Mapped);
        assert_eq!(plan.owners["b"], OwnerShare::Mapped);
        let output = scratch.join("split-owner-combined");
        combine(&CombineRequest {
            sources: std::slice::from_ref(&source),
            output: &output,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap();
        let metadata = Metadata::open(&output).unwrap();
        assert!(
            metadata
                .locations
                .locations
                .iter()
                .all(|location| location.payload_id != target_id.as_bytes().to_vec()),
            "split target remained physically installed"
        );
        let selection = protect::Selection::Deployment { deployment: 1, rank: None };
        let verified = protect::verify(
            &metadata.envelope,
            std::slice::from_ref(&publisher),
            selection,
        )
        .unwrap();
        let initializer = verified
            .ranks()
            .first()
            .and_then(|rank| {
                rank.initializers.iter().find(|initializer| {
                    initializer
                        .source
                        .as_ref()
                        .and_then(|use_| use_.slice.as_ref())
                        .is_some_and(|slice| slice.payload_id == target_id.as_bytes().to_vec())
                })
            })
            .expect("split target initializer missing");
        assert_eq!(
            initializer.source.as_ref().unwrap().slice.as_ref().unwrap().nbytes,
            target.len() as u64
        );
        let store = cache::Root::open(scratch.join("split-owner-cache")).unwrap();
        let package = cache::PackageSource::open(&output, std::slice::from_ref(&publisher), selection).unwrap();
        let entry = store.install_package(package, "split-owner", "entry").unwrap();
        let installed = Metadata::open(entry.directory()).unwrap();
        let location = installed
            .locations
            .locations
            .iter()
            .find(|location| location.payload_id == target_id.as_bytes().to_vec())
            .expect("installed split target location missing");
        let slice = entry.payload(&location.file).unwrap();
        let mut bytes = vec![0; slice.len() as usize];
        slice.read_at(0, &mut bytes).unwrap();
        assert_eq!(bytes, target);
        assert_eq!(bytes.len(), 8);
    }

    #[test]
    fn combine_error_in_later_source_cleans_all_source_scratch_and_output() {
        let scratch = Scratch::new("combine-cleanup");
        let second_scratch = Scratch::new("combine-cleanup-second");
        let first = fixture(&scratch);
        let second = fixture(&second_scratch);
        // Keep the source packages valid and signed, but make the second source's
        // unsigned derivation set unparsable. Its scratch is created before this
        // failure, so both source guards must clean it.
        rewrite_derivations(
            &second.directory,
            &wire::DerivationSet {
                version: derivation::DERIVATION_VERSION.wrapping_add(1),
                derivations: Vec::new(),
            },
        );
        let first_primary = fs::read(first.directory.join("model.safetensors")).unwrap();
        let second_primary = fs::read(second.directory.join("model.safetensors")).unwrap();
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let output = scratch.join("cleanup-combined");
        let mut publishers = first.publishers.clone();
        publishers.extend(second.publishers.clone());
        let error = combine(&CombineRequest {
            sources: &[first.directory.clone(), second.directory.clone()],
            output: &output,
            publishers: &publishers,
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap_err();
        assert!(error.to_string().contains("unsupported derivation set version"), "{error}");
        assert!(!output.exists());
        assert_eq!(
            fs::read(first.directory.join("model.safetensors")).unwrap(),
            first_primary,
            "first source primary changed"
        );
        assert_eq!(
            fs::read(second.directory.join("model.safetensors")).unwrap(),
            second_primary,
            "second source primary changed"
        );
        for root in [&scratch.0, &second_scratch.0] {
            let leftovers: Vec<_> = fs::read_dir(root)
                .unwrap()
                .flatten()
                .map(|entry| entry.file_name().to_string_lossy().to_string())
                .filter(|name| name.contains("source-"))
                .collect();
            assert!(leftovers.is_empty(), "source scratch remains in {root:?}: {leftovers:?}");
        }
    }

    #[test]
    fn combine_preserves_a_multi_hop_source_chain() {
        let copy = |source, source_offset, destination_offset, contiguous| {
            derivation::Operation::Copy(derivation::Copy {
                source,
                source_offset,
                destination_offset,
                contiguous,
                outer: Vec::new(),
            })
        };
        let middle = PayloadId::of(&(0..8u8).collect::<Vec<u8>>());
        let weights = PayloadId::of(&(0..16u8).collect::<Vec<u8>>());
        for (name, operations) in [
            (
                "one-copy",
                vec![copy(middle, 0, 0, 4)],
            ),
            (
                "repeated-source",
                vec![copy(middle, 0, 0, 2), copy(middle, 2, 2, 2)],
            ),
            (
                "mixed-source",
                vec![copy(middle, 0, 0, 2), copy(weights, 2, 2, 2)],
            ),
        ] {
            let scratch = Scratch::new(name);
            combine_chain_case(&scratch, operations);
        }
    }

    #[test]
    fn multi_hop_derivations_are_emitted_and_materialized() {
        let scratch = Scratch::new("chain");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let directory = publish_chain_source(&scratch, &key, &publisher, true);
        let weights = PayloadId::of(&(0..16u8).collect::<Vec<u8>>());
        let middle = PayloadId::of(&(0..8u8).collect::<Vec<u8>>());
        let final_id = PayloadId::of(&(0..4u8).collect::<Vec<u8>>());
        let first = derivation::Derivation {
            target: middle,
            length: 8,
            operations: vec![derivation::Operation::Copy(derivation::Copy {
                source: weights,
                source_offset: 0,
                destination_offset: 0,
                contiguous: 8,
                outer: Vec::new(),
            })],
        };
        assert_eq!(
            emit_derivations(&directory, &[first.clone()], &derivation::Limits::default()).unwrap(),
            1
        );
        let second = derivation::Derivation {
            target: final_id,
            length: 4,
            operations: vec![derivation::Operation::Copy(derivation::Copy {
                source: middle,
                source_offset: 0,
                destination_offset: 0,
                contiguous: 4,
                outer: Vec::new(),
            })],
        };
        // The second hop resolves its source by materializing the first recipe.
        emit_derivations(&directory, &[second], &derivation::Limits::default()).unwrap();
        let metadata = Metadata::open(&directory).unwrap();
        let set = wire::DerivationSet::decode(
            tensor_bytes(metadata.primary(), reader::DERIVATIONS).as_slice(),
        )
        .unwrap();
        assert_eq!(set.derivations.len(), 2);
        // Packing proves the selected closure still resolves through the recipe.
        let packed = scratch.join("chain-packed.safetensors");
        pack(&directory, &packed).unwrap();
        let packed = Metadata::open(&packed).unwrap();
        assert!(!tensor_bytes(packed.primary(), reader::DERIVATIONS).is_empty());
    }

    fn move_package_payload_to_companion(directory: &Path, id: PayloadId) {
        let primary = directory.join("model.safetensors");
        let metadata = Metadata::open(directory).unwrap();
        let location = metadata
            .locations
            .locations
            .iter()
            .find(|location| location.payload_id == id.as_bytes().to_vec())
            .unwrap()
            .clone();
        assert!(location.file.is_empty());
        let slice = metadata.primary().tensor(&location.tensor).unwrap();
        let relative = "frontend/tokenizer.safetensors";
        let path = directory.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut file = File::create(&path).unwrap();
        write_companion(&mut file, &slice).unwrap();
        file.sync_all().unwrap();
        let mut tensors: Vec<(String, Vec<u8>)> = Vec::new();
        for name in metadata.primary().tensor_names() {
            if name == reader::LOCATIONS
                || name == location.tensor
                || name.starts_with(container::ALIGNMENT_PREFIX)
            {
                continue;
            }
            tensors.push((name.to_owned(), tensor_bytes(metadata.primary(), name)));
        }
        let mut set = wire::LocationSet::decode(
            tensor_bytes(metadata.primary(), reader::LOCATIONS).as_slice(),
        )
        .unwrap();
        for entry in &mut set.locations {
            if entry.payload_id == id.as_bytes().to_vec() {
                entry.file = relative.into();
                entry.tensor = container::COMPANION_TENSOR.into();
            }
        }
        let encoded = set.encode_to_vec();
        let mut writer =
            Writer::new(File::create(&primary).unwrap(), container::PAYLOAD_FORMAT).unwrap();
        for (name, bytes) in tensors {
            writer
                .add(&name, bytes.len() as u64, Source::Bytes(bytes))
                .unwrap();
        }
        writer
            .add(
                reader::LOCATIONS,
                encoded.len() as u64,
                Source::Bytes(encoded),
            )
            .unwrap();
        writer.finish().unwrap();
    }

    #[test]
    fn combine_embeds_package_payloads_that_live_in_companions() {
        let scratch = Scratch::new("frontend-companion");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let source = publish_source(
            &scratch,
            "frontend-source",
            1,
            &[(0..8u8).collect::<Vec<_>>().as_slice()],
            &[(10..18u8).collect::<Vec<_>>().as_slice()],
            &key,
            &publisher,
            "model",
            false,
        );
        let frontend = PayloadId::of(b"tokenizer-json");
        move_package_payload_to_companion(&source, frontend);
        let output = scratch.join("frontend-combined");
        combine(&CombineRequest {
            sources: std::slice::from_ref(&source),
            output: &output,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap();
        let metadata = Metadata::open(&output).unwrap();
        let location = metadata
            .locations
            .locations
            .iter()
            .find(|location| location.payload_id == frontend.as_bytes().to_vec())
            .unwrap();
        assert!(location.file.is_empty());
        let mut bytes = vec![0u8; 14];
        metadata
            .primary()
            .tensor(&location.tensor)
            .unwrap()
            .read_at(0, &mut bytes)
            .unwrap();
        assert_eq!(bytes, b"tokenizer-json");
        let template = PayloadId::of(b"chat-template");
        let template_location = metadata
            .locations
            .locations
            .iter()
            .find(|location| location.payload_id == template.as_bytes().to_vec())
            .unwrap();
        assert!(template_location.file.is_empty());
        protect::verify(
            &metadata.envelope,
            std::slice::from_ref(&publisher),
            Selection::Deployment {
                deployment: 0,
                rank: None,
            },
        )
        .unwrap();
    }

    #[test]
    fn combine_preserves_independent_source_identities_but_rejects_semantic_differences() {
        let scratch = Scratch::new("rebuild-identity");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let full: Vec<u8> = (0..8).collect();
        let matrix: Vec<u8> = (10..18).collect();
        let half0 = &full[0..4];
        let half1 = &full[4..8];
        let col0 = [10u8, 12, 14, 16];
        let col1 = [11u8, 13, 15, 17];
        let rebuild = |identity: &str| runtime::RebuildManifest {
            version: 1,
            recipe: "qwen38".into(),
            source_identity: identity.into(),
            checkpoint_repository: "owner/model".into(),
            checkpoint_revision: "revision".into(),
            local_checkpoint: false,
            expert_scale_compression: Some("csf".into()),
            online_mxfp8: true,
            language_model_only: false,
            mla_compute_mode: "dense".into(),
            speculation: Some(runtime::RebuildSpeculation::default()),
            ..Default::default()
        };
        let publish_pair = |tag: &str, first: runtime::RebuildManifest, second: runtime::RebuildManifest| {
            let tp1 = publish_source_with(
                &scratch,
                &format!("{tag}-tp1"),
                1,
                &[full.as_slice()],
                &[matrix.as_slice()],
                &key,
                &publisher,
                "model",
                false,
                Some(first),
            );
            let tp2 = publish_source_with(
                &scratch,
                &format!("{tag}-tp2"),
                2,
                &[half0, half1],
                &[col0.as_slice(), col1.as_slice()],
                &key,
                &publisher,
                "model",
                false,
                Some(second),
            );
            (tp1, tp2)
        };

        let first_identity = "source-code-a";
        let second_identity = "source-code-b";
        let (tp1, tp2) = publish_pair("same-semantics", rebuild(first_identity), rebuild(second_identity));
        let combined = scratch.join("same-semantics-combined");
        combine(&CombineRequest {
            sources: &[tp1.clone(), tp2],
            output: &combined,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap();
        let metadata = Metadata::open(&combined).unwrap();
        let envelope = protect::envelope(&metadata.envelope).unwrap();
        let mut identities = BTreeMap::new();
        for deployment in &envelope.deployments {
            let body = wire::DeploymentBody::decode(deployment.body.as_slice()).unwrap();
            if let Some(rebuild) = body.rebuild {
                identities.insert(body.deployment_id, rebuild.source_identity);
            }
        }
        assert_eq!(
            identities.get("sm120-tp1").map(String::as_str),
            Some(first_identity)
        );
        assert_eq!(
            identities.get("sm120-tp2").map(String::as_str),
            Some(second_identity)
        );

        let store = crate::package::cache::Root::open(scratch.join("same-semantics-cache")).unwrap();
        let source = crate::package::cache::PackageSource::open(
            &combined,
            std::slice::from_ref(&publisher),
            Selection::Deployment {
                deployment: 1,
                rank: None,
            },
        )
        .unwrap();
        let entry = store.install_package(source, "model", "tp2").unwrap();
        let installed = Metadata::open(entry.directory()).unwrap();
        let installed_envelope = protect::envelope(&installed.envelope).unwrap();
        let installed_body = wire::DeploymentBody::decode(
            installed_envelope.deployments[1].body.as_slice(),
        )
        .unwrap();
        assert_eq!(
            installed_body.rebuild.unwrap().source_identity,
            second_identity
        );

        let semantic_cases: [(&str, fn(&mut runtime::RebuildManifest)); 8] = [
            ("checkpoint-repository", |value| value.checkpoint_repository = "other/model".into()),
            ("checkpoint-revision", |value| value.checkpoint_revision = "other".into()),
            ("local-checkpoint", |value| value.local_checkpoint = true),
            ("recipe", |value| value.recipe = "other-recipe".into()),
            ("expert-scale-compression", |value| {
                value.expert_scale_compression = Some("native".into())
            }),
            ("online-mxfp8", |value| value.online_mxfp8 = false),
            ("language-model-only", |value| value.language_model_only = true),
            ("mla-compute-mode", |value| value.mla_compute_mode = "sparse".into()),
        ];
        for (label, change) in semantic_cases {
            let first = rebuild("same-code");
            let mut second = rebuild("same-code");
            change(&mut second);
            let (tp1, tp2) = publish_pair(label, first, second);
            let output = scratch.join(&format!("{label}-combined"));
            let error = combine(&CombineRequest {
                sources: &[tp1, tp2],
                output: &output,
                publishers: std::slice::from_ref(&publisher),
                accepted: "example",
                target_accepted: "example",
                target_publisher: None,
                key: &key,
            })
            .expect_err("semantic rebuild mismatch must reject");
            assert_eq!(
                error.to_string(),
                "source deployments disagree on checkpoint or preparation semantics",
                "{label}: {error}"
            );
        }

        let first = rebuild("same-code");
        let mut second = rebuild("same-code");
        second.speculation.as_mut().unwrap().history_drafts = true;
        let (tp1, tp2) = publish_pair("speculation", first, second);
        let output = scratch.join("speculation-combined");
        let speculation_error = combine(&CombineRequest {
            sources: &[tp1, tp2],
            output: &output,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .expect_err("speculation mismatch must reject");
        assert_eq!(
            speculation_error.to_string(),
            "source deployments disagree on checkpoint or preparation semantics",
            "speculation: {speculation_error}"
        );
    }

    #[test]
    fn multi_deployment_sources_are_preserved_and_recombine() {
        let scratch = Scratch::new("multi-source");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let full: Vec<u8> = (0..8).collect();
        let matrix: Vec<u8> = (10..18).collect();
        let half0 = &full[0..4];
        let half1 = &full[4..8];
        let col0 = [10u8, 12, 14, 16];
        let col1 = [11u8, 13, 15, 17];
        let tp1 = publish_source(
            &scratch,
            "multi-tp1",
            1,
            &[full.as_slice()],
            &[matrix.as_slice()],
            &key,
            &publisher,
            "model",
            false,
        );
        let tp2 = publish_source(
            &scratch,
            "multi-tp2",
            2,
            &[half0, half1],
            &[col0.as_slice(), col1.as_slice()],
            &key,
            &publisher,
            "model",
            false,
        );
        let combined = scratch.join("multi-combined");
        combine(&CombineRequest {
            sources: &[tp1, tp2],
            output: &combined,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap();
        // A multi-deployment package is a valid combine input and promotion
        // source; neither operation may silently drop a deployment.
        let again = scratch.join("multi-again");
        combine(&CombineRequest {
            sources: std::slice::from_ref(&combined),
            output: &again,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap();
        let promoted = scratch.join("multi-promoted");
        promote(
            &combined,
            &promoted,
            std::slice::from_ref(&publisher),
            "example",
            "example",
            None,
            &key,
        )
        .unwrap();
        for artifact in [&again, &promoted] {
            let metadata = Metadata::open(artifact).unwrap();
            let ids: Vec<String> = protect::verify(
                &metadata.envelope,
                std::slice::from_ref(&publisher),
                Selection::Deployment {
                    deployment: 0,
                    rank: None,
                },
            )
            .unwrap()
            .package()
            .deployments
            .iter()
            .map(|entry| entry.deployment_id.clone())
            .collect();
            assert_eq!(ids, vec!["sm120-tp1".to_string(), "sm120-tp2".to_string()]);
            for deployment in 0..2 {
                protect::verify(
                    &metadata.envelope,
                    std::slice::from_ref(&publisher),
                    Selection::Deployment {
                        deployment,
                        rank: None,
                    },
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn packing_rejects_corrupted_payloads_without_staging_leftovers() {
        let scratch = Scratch::new("corrupt-pack");
        let fixture = fixture(&scratch);
        let companion = fixture.directory.join("weights/layer-000.safetensors");
        let mut bytes = fs::read(&companion).unwrap();
        let end = bytes.len() - 1;
        bytes[end] ^= 0xff;
        fs::write(&companion, bytes).unwrap();
        let output = scratch.join("corrupt-packed.safetensors");
        let error = pack(&fixture.directory, &output).err().unwrap().to_string();
        assert!(error.contains("content does not match"), "{error}");
        assert!(!output.exists());
        let leftovers: Vec<String> = fs::read_dir(&scratch.0)
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .filter(|name| name.contains(".corrupt-packed.safetensors.pack-"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        // A stale staging name from an earlier process cannot block a new pack.
        fs::write(
            scratch.join(&format!(
                ".corrupt-packed.safetensors.pack-{}-0000000000000000",
                std::process::id()
            )),
            b"stale",
        )
        .unwrap();
    }

    #[test]
    fn concurrent_packs_and_emissions_stay_consistent() {
        let scratch = Scratch::new("concurrent-forms");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let full: Vec<u8> = (0..8).collect();
        let matrix: Vec<u8> = (10..18).collect();
        let half0 = &full[0..4];
        let half1 = &full[4..8];
        let col0 = [10u8, 12, 14, 16];
        let col1 = [11u8, 13, 15, 17];
        let tp1 = publish_source(
            &scratch,
            "concurrent-tp1",
            1,
            &[full.as_slice()],
            &[matrix.as_slice()],
            &key,
            &publisher,
            "model",
            false,
        );
        let tp2 = publish_source(
            &scratch,
            "concurrent-tp2",
            2,
            &[half0, half1],
            &[col0.as_slice(), col1.as_slice()],
            &key,
            &publisher,
            "model",
            false,
        );
        let combined = scratch.join("concurrent-combined");
        combine(&CombineRequest {
            sources: &[tp1, tp2],
            output: &combined,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap();

        let mut packs = Vec::new();
        for index in 0..2 {
            let source = combined.clone();
            let output = scratch.join(&format!("concurrent-{index}.safetensors"));
            packs.push(std::thread::spawn(move || {
                let result = pack(&source, &output);
                (result, output)
            }));
        }
        for pack in packs {
            let (result, output) = pack.join().unwrap();
            result.unwrap();
            assert!(output.is_file());
        }

        let mut emissions = Vec::new();
        for (_rank, state) in [(0u32, b"state-2-0".to_vec()), (1u32, b"state-2-1".to_vec())] {
            let source = combined.clone();
            let recipe = derivation::Derivation {
                target: PayloadId::of(&state),
                length: state.len() as u64,
                operations: vec![derivation::Operation::Fill(derivation::Fill {
                    destination_offset: 0,
                    length: state.len() as u64,
                    pattern: state,
                })],
            };
            emissions.push(std::thread::spawn(move || {
                emit_derivations(&source, &[recipe], &derivation::Limits::default())
            }));
        }
        for emission in emissions {
            emission.join().unwrap().unwrap();
        }
        let metadata = Metadata::open(&combined).unwrap();
        let set = wire::DerivationSet::decode(
            tensor_bytes(metadata.primary(), reader::DERIVATIONS).as_slice(),
        )
        .unwrap();
        assert_eq!(set.derivations.len(), 6);
        protect::verify(
            &metadata.envelope,
            std::slice::from_ref(&publisher),
            Selection::Deployment {
                deployment: 1,
                rank: None,
            },
        )
        .unwrap();
        let packed = scratch.join("concurrent-final.safetensors");
        pack(&combined, &packed).unwrap();
    }

    #[test]
    fn combine_refuses_existing_output_without_polluting_it() {
        let scratch = Scratch::new("combine-guard");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let full: Vec<u8> = (0..8).collect();
        let matrix: Vec<u8> = (10..18).collect();
        let source = publish_source(
            &scratch,
            "guard-source",
            1,
            &[full.as_slice()],
            &[matrix.as_slice()],
            &key,
            &publisher,
            "model",
            false,
        );
        let output = scratch.join("guard-output");
        combine(&CombineRequest {
            sources: std::slice::from_ref(&source),
            output: &output,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap();
        let primary = output.join("model.safetensors");
        let before = fs::read(&primary).unwrap();
        let error = combine(&CombineRequest {
            sources: &[source],
            output: &output,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .err()
        .expect("existing output must be refused")
        .to_string();
        assert!(error.contains("destination exists"), "{error}");
        assert_eq!(fs::read(&primary).unwrap(), before);
        let leftovers: Vec<String> = fs::read_dir(&scratch.0)
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .filter(|name| name.contains(".guard-output."))
            .collect();
        assert!(leftovers.is_empty(), "staging leftovers: {leftovers:?}");
    }

    #[test]
    fn combined_derivations_survive_keyless_container_forms() {
        let scratch = Scratch::new("derivation-forms");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let full: Vec<u8> = (0..8).collect();
        let matrix: Vec<u8> = (10..18).collect();
        let half0 = &full[0..4];
        let half1 = &full[4..8];
        let col0 = [10u8, 12, 14, 16];
        let col1 = [11u8, 13, 15, 17];
        let tp1 = publish_source(
            &scratch,
            "forms-tp1",
            1,
            &[full.as_slice()],
            &[matrix.as_slice()],
            &key,
            &publisher,
            "model",
            false,
        );
        let tp2 = publish_source(
            &scratch,
            "forms-tp2",
            2,
            &[half0, half1],
            &[col0.as_slice(), col1.as_slice()],
            &key,
            &publisher,
            "model",
            false,
        );
        let combined = scratch.join("forms-combined");
        combine(&CombineRequest {
            sources: &[tp1, tp2],
            output: &combined,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap();
        let packed = scratch.join("forms-packed.safetensors");
        pack(&combined, &packed).unwrap();
        let directory = scratch.join("forms-directory");
        unpack(&packed, &directory).unwrap();

        let original = Metadata::open(&combined).unwrap();
        let converted = Metadata::open(&directory).unwrap();
        let expected = tensor_bytes(original.primary(), reader::DERIVATIONS);
        assert!(!expected.is_empty());
        assert_eq!(
            tensor_bytes(converted.primary(), reader::DERIVATIONS),
            expected
        );
        for deployment in 0..2 {
            protect::verify(
                &converted.envelope,
                std::slice::from_ref(&publisher),
                Selection::Deployment {
                    deployment,
                    rank: None,
                },
            )
            .unwrap();
        }
        // Merging a new recipe must retain every existing recipe.
        {
            let state = b"state-2-0".to_vec();
            let recipe = derivation::Derivation {
                target: PayloadId::of(&state),
                length: state.len() as u64,
                operations: vec![derivation::Operation::Fill(derivation::Fill {
                    destination_offset: 0,
                    length: state.len() as u64,
                    pattern: state.clone(),
                })],
            };
            let added =
                emit_derivations(&directory, &[recipe], &derivation::Limits::default()).unwrap();
            assert_eq!(added, 1);
            let merged = Metadata::open(&directory).unwrap();
            let set = wire::DerivationSet::decode(
                tensor_bytes(merged.primary(), reader::DERIVATIONS).as_slice(),
            )
            .unwrap();
            assert_eq!(set.derivations.len(), 5);
            protect::verify(
                &merged.envelope,
                std::slice::from_ref(&publisher),
                Selection::Deployment {
                    deployment: 1,
                    rank: None,
                },
            )
            .unwrap();
        }

        // A selected subset ignores absent unselected payloads; the default
        // conversion still requires the complete closure.
        drop_location(&directory, "deployments/sm120-tp2/");
        pack_selected(
            &directory,
            &scratch.join("forms-subset.safetensors"),
            Some(&[0]),
        )
        .unwrap();
        assert!(pack(&directory, &scratch.join("forms-complete.safetensors")).is_err());

        // The converted artifact still declares, but does not ship, target payloads.
        let verified = protect::verify(
            &converted.envelope,
            std::slice::from_ref(&publisher),
            Selection::Deployment {
                deployment: 1,
                rank: None,
            },
        )
        .unwrap();
        let catalog = verified.catalog().clone();
        for rank in verified.ranks() {
            for initializer in &rank.initializers[1..] {
                let slice = initializer.source.as_ref().unwrap().slice.as_ref().unwrap();
                assert!(
                    converted
                        .locations
                        .locations
                        .iter()
                        .all(|location| location.payload_id != slice.payload_id),
                    "derived target must remain materialized on acquisition, not shipped"
                );
                let set = wire::DerivationSet::decode(expected.as_slice()).unwrap();
                let recipes =
                    derivation::from_wire(&set, &catalog, &derivation::Limits::default()).unwrap();
                let id = PayloadId::from_bytes(slice.payload_id.as_slice().try_into().unwrap());
                assert!(recipes.iter().any(|recipe| recipe.target == id));
            }
        }
    }

    #[test]
    fn promotes_across_namespaces_with_rewritten_identity() {
        let scratch = Scratch::new("cross-namespace");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let source_publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let target_publisher = Publisher {
            name: "Other".into(),
            namespace: "other".into(),
            public_key,
            development: false,
        };
        let matrix: Vec<u8> = (10..18).collect();
        let full: Vec<u8> = (0..8).collect();
        let source = publish_source(
            &scratch,
            "namespace-source",
            1,
            &[full.as_slice()],
            &[matrix.as_slice()],
            &key,
            &source_publisher,
            "model",
            false,
        );
        let output = scratch.join("namespace-promoted");
        let target = TargetPublisher {
            namespace: "other".into(),
            official_source: "https://huggingface.co/other/model".into(),
            development: false,
        };
        promote(
            &source,
            &output,
            &[source_publisher, target_publisher.clone()],
            "example",
            "other",
            Some(&target),
            &key,
        )
        .unwrap();
        let metadata = Metadata::open(&output).unwrap();
        let verified = protect::verify(
            &metadata.envelope,
            std::slice::from_ref(&target_publisher),
            Selection::Deployment {
                deployment: 0,
                rank: None,
            },
        )
        .unwrap();
        let publisher = verified.package().publisher.as_ref().unwrap();
        assert_eq!(publisher.namespace, "other");
        assert_eq!(
            publisher.official_source,
            "https://huggingface.co/other/model"
        );
        assert_eq!(publisher.license_name, "Example");
        assert_eq!(publisher.license_link, "");
        let authorized = verified.authorize("other").unwrap();
        let payloads = metadata
            .payloads(
                authorized.verified().catalog(),
                [PayloadId::of(&full).as_bytes().as_slice()],
            )
            .unwrap();
        payloads.verify().unwrap();
    }

    #[test]
    fn normalized_relative_falls_back_when_schema_rejects_container_paths() {
        let identity = PayloadId::of(b"normalized-relative");
        let long = "a".repeat(129);
        let deep = std::iter::repeat_n("segment", 17).collect::<Vec<_>>().join(".");
        for tensor in [format!("weights.{long}"), format!("weights.{deep}")] {
            let relative = normalized_relative("", &tensor, identity).unwrap();
            schema::location_path(&relative).unwrap();
            container::validate_relative_path(&relative).unwrap();
            assert!(relative.starts_with("packed/"));
        }
    }

    #[test]
    fn combine_packed_sources_use_readable_paths_for_bracketed_aliases() {
        let scratch = Scratch::new("combine-packed-aliases");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let full = vec![0x51; 8];
        let half0 = vec![0x52; 8];
        let half1 = vec![0x53; 8];
        let matrix = vec![0x54; 8];
        let column1 = vec![0x55; 8];
        let tp1_dir = publish_source_impl_with_alias(
            &scratch,
            "alias-tp1",
            1,
            &[full.as_slice()],
            &[matrix.as_slice()],
            &key,
            &publisher,
            "model",
            EngramFixture::None,
            None,
            true,
        );
        let tp2_dir = publish_source_impl_with_alias(
            &scratch,
            "alias-tp2",
            2,
            &[half0.as_slice(), half1.as_slice()],
            &[matrix.as_slice(), column1.as_slice()],
            &key,
            &publisher,
            "model",
            EngramFixture::None,
            None,
            true,
        );
        let tp1 = scratch.join("packed-alias-tp1.safetensors");
        let tp2 = scratch.join("packed-alias-tp2.safetensors");
        pack(&tp1_dir, &tp1).unwrap();
        pack(&tp2_dir, &tp2).unwrap();
        for source in [&tp1, &tp2] {
            let metadata = Metadata::open(source).unwrap();
            assert!(metadata.locations.locations.iter().all(|location| location.file.is_empty()));
        }

        let output = scratch.join("packed-alias-combined");
        combine(&CombineRequest {
            sources: &[tp1, tp2],
            output: &output,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap();
        let metadata = Metadata::open(&output).unwrap();
        let companions: Vec<_> = metadata
            .locations
            .locations
            .iter()
            .filter(|location| !location.file.is_empty())
            .collect();
        assert!(!companions.is_empty());
        for location in &companions {
            schema::location_path(&location.file).unwrap();
            container::validate_relative_path(&location.file).unwrap();
            assert!(!location.file.contains('['));
            assert!(!location.file.contains(']'));
        }
        let full_id = PayloadId::of(&full);
        let canonical = companions
            .iter()
            .find(|location| location.payload_id.as_slice() == full_id.as_bytes())
            .unwrap();
        assert_eq!(canonical.file, "weights/layer-000.safetensors");

        let selected = protect::verify(
            &metadata.envelope,
            std::slice::from_ref(&publisher),
            Selection::Deployment {
                deployment: 1,
                rank: None,
            },
        )
        .unwrap();
        assert_eq!(selected.deployment().unwrap().deployment_id, "sm120-tp2");
        assert_eq!(selected.ranks().len(), 2);
        let slice = selected.ranks()[0].initializers[1]
            .source
            .as_ref()
            .unwrap()
            .slice
            .as_ref()
            .unwrap();
        let catalog = selected.catalog().clone();
        let files = metadata
            .payloads(&catalog, [slice.payload_id.as_slice()])
            .unwrap();
        files.verify().unwrap();
        let mut bytes = vec![0u8; slice.nbytes as usize];
        files
            .get(&slice.payload_id)
            .unwrap()
            .read_at(slice.offset, &mut bytes)
            .unwrap();
        assert_eq!(bytes, half0);
    }

    #[test]
    fn combines_engram_deployments_with_shared_planes() {
        let scratch = Scratch::new("combine-engram");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let full: Vec<u8> = (0..8).collect();
        let matrix: Vec<u8> = (10..18).collect();
        let half0 = &full[0..4];
        let half1 = &full[4..8];
        let col0 = [10u8, 12, 14, 16];
        let col1 = [11u8, 13, 15, 17];
        let tp1 = publish_source(
            &scratch,
            "engram-tp1",
            1,
            &[full.as_slice()],
            &[matrix.as_slice()],
            &key,
            &publisher,
            "model",
            true,
        );
        let tp2 = publish_source(
            &scratch,
            "engram-tp2",
            2,
            &[half0, half1],
            &[col0.as_slice(), col1.as_slice()],
            &key,
            &publisher,
            "model",
            true,
        );
        let output = scratch.join("engram-combined");
        combine(&CombineRequest {
            sources: &[tp1, tp2],
            output: &output,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap();
        let metadata = Metadata::open(&output).unwrap();
        for deployment in 0..2 {
            let verified = protect::verify(
                &metadata.envelope,
                std::slice::from_ref(&publisher),
                Selection::Deployment {
                    deployment,
                    rank: None,
                },
            )
            .unwrap();
            let body = verified.deployment().unwrap();
            let table = &body.engram.as_ref().unwrap().tables[0];
            let weights = table.weights.as_ref().unwrap().slice.as_ref().unwrap();
            assert!(
                body.payloads
                    .iter()
                    .any(|descriptor| descriptor.id == weights.payload_id),
                "deployment {deployment} must authenticate the engram descriptor"
            );
            let catalog = verified.catalog().clone();
            let files = metadata
                .payloads(&catalog, [weights.payload_id.as_slice()])
                .unwrap();
            files.verify().unwrap();
            let mut bytes = vec![0u8; weights.nbytes as usize];
            files
                .get(&weights.payload_id)
                .unwrap()
                .read_at(weights.offset, &mut bytes)
                .unwrap();
            assert_eq!(bytes, b"engram-weights");
        }
    }

    #[test]
    fn combines_tp1_shared_and_tp2_partitioned_engram_with_exact_planes() {
        let scratch = Scratch::new("combine-partitioned-engram");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let full: Vec<u8> = (0..8).collect();
        let matrix: Vec<u8> = (10..18).collect();
        let half0 = &full[0..4];
        let half1 = &full[4..8];
        let col0 = [10u8, 12, 14, 16];
        let col1 = [11u8, 13, 15, 17];
        let tp1 = publish_source_impl(
            &scratch,
            "partitioned-tp1",
            1,
            &[full.as_slice()],
            &[matrix.as_slice()],
            &key,
            &publisher,
            "model",
            EngramFixture::Tp1Shared,
            None,
        );
        let tp2 = publish_source_impl(
            &scratch,
            "partitioned-tp2",
            2,
            &[half0, half1],
            &[col0.as_slice(), col1.as_slice()],
            &key,
            &publisher,
            "model",
            EngramFixture::Partitioned,
            None,
        );
        let output = scratch.join("partitioned-engram-combined");
        combine(&CombineRequest {
            sources: &[tp1.clone(), tp2],
            output: &output,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap();
        let metadata = Metadata::open(&output).unwrap();
        let tp1_selected = protect::verify(
            &metadata.envelope,
            std::slice::from_ref(&publisher),
            protect::Selection::Deployment {
                deployment: 0,
                rank: None,
            },
        )
        .unwrap();
        let tp1_table = &tp1_selected
            .deployment()
            .unwrap()
            .engram
            .as_ref()
            .unwrap()
            .tables[0];
        assert_eq!(tp1_table.rows, ENGRAM_ROWS as u64);
        assert_eq!(tp1_table.embedding_width, ENGRAM_WIDTH as u32);
        assert_eq!(tp1_table.scale_block, ENGRAM_SCALE_BLOCK as u32);
        assert_eq!(
            tp1_table.encoding,
            wire::engram_table::Encoding::Fp8E4m3E8m0 as i32
        );
        let tp1_hash = tp1_table.hash.as_ref().unwrap();
        assert_eq!(tp1_hash.offsets, engram_tp1_offsets());
        assert_eq!(tp1_hash.orders, engram_orders());
        assert_eq!(tp1_hash.multipliers, engram_multipliers());
        assert_eq!(tp1_hash.primes, vec![2; ENGRAM_HEADS]);
        for (prime, offset) in tp1_hash.primes.iter().zip(&tp1_hash.offsets) {
            assert!(*prime >= 2);
            assert!(offset
                .checked_add(*prime)
                .is_some_and(|end| end <= tp1_table.rows));
        }
        assert!(tp1_table.shards.is_empty());
        let tp1_weights_use = tp1_table.weights.as_ref().unwrap().slice.as_ref().unwrap();
        let tp1_scales_use = tp1_table.scales.as_ref().unwrap().slice.as_ref().unwrap();
        assert_eq!(
            tp1_weights_use.nbytes,
            (ENGRAM_ROWS * ENGRAM_WIDTH) as u64
        );
        assert_eq!(
            tp1_scales_use.nbytes,
            (ENGRAM_ROWS * ENGRAM_SCALES_PER_ROW) as u64
        );
        let tp1_catalog = tp1_selected.catalog().clone();
        let tp1_files = metadata
            .payloads(
                &tp1_catalog,
                [
                    tp1_weights_use.payload_id.as_slice(),
                    tp1_scales_use.payload_id.as_slice(),
                ],
            )
            .unwrap();
        tp1_files.verify().unwrap();
        let mut tp1_weights = vec![0u8; tp1_weights_use.nbytes as usize];
        tp1_files
            .get(&tp1_weights_use.payload_id)
            .unwrap()
            .read_at(tp1_weights_use.offset, &mut tp1_weights)
            .unwrap();
        assert_eq!(tp1_weights, engram_tp1_weights());
        let mut tp1_scales = vec![0u8; tp1_scales_use.nbytes as usize];
        tp1_files
            .get(&tp1_scales_use.payload_id)
            .unwrap()
            .read_at(tp1_scales_use.offset, &mut tp1_scales)
            .unwrap();
        assert_eq!(tp1_scales, engram_tp1_scales());

        let selected = protect::verify(
            &metadata.envelope,
            std::slice::from_ref(&publisher),
            protect::Selection::Deployment {
                deployment: 1,
                rank: None,
            },
        )
        .unwrap();
        let body = selected.deployment().unwrap();
        let table = &body.engram.as_ref().unwrap().tables[0];
        assert_eq!(table.rows, ENGRAM_ROWS as u64);
        assert_eq!(table.embedding_width, ENGRAM_WIDTH as u32);
        assert_eq!(table.scale_block, ENGRAM_SCALE_BLOCK as u32);
        assert_eq!(
            table.encoding,
            wire::engram_table::Encoding::Fp8E4m3E8m0 as i32
        );
        let hash = table.hash.as_ref().unwrap();
        assert_eq!(hash.offsets, engram_tp2_offsets());
        assert_eq!(hash.orders, engram_orders());
        assert_eq!(hash.multipliers, engram_multipliers());
        assert_eq!(hash.primes, vec![2; ENGRAM_HEADS]);
        assert_eq!(
            hash.orders,
            (2..=4).flat_map(|order| [order; 8]).collect::<Vec<_>>()
        );
        assert_eq!(hash.multipliers.len(), 4);
        assert!(hash.multipliers.iter().all(|multiplier| {
            *multiplier != 0
                && multiplier
                    .checked_mul(8)
                    .is_some_and(|scaled| scaled <= i64::MAX as u64)
        }));
        for (prime, offset) in hash.primes.iter().zip(&hash.offsets) {
            assert!(*prime >= 2);
            assert!(offset
                .checked_add(*prime)
                .is_some_and(|end| end <= table.rows));
        }
        assert!(table.weights.is_none());
        assert!(table.scales.is_none());
        assert_eq!(table.request_buffer, "r");
        assert_eq!(table.shards.len(), 2);
        let catalog = selected.catalog().clone();
        let files = metadata
            .payloads(
                &catalog,
                table.shards.iter().flat_map(|shard| {
                    [&shard.weights, &shard.scales]
                        .into_iter()
                        .flatten()
                        .filter_map(|use_| use_.slice.as_ref().map(|slice| slice.payload_id.as_slice()))
                }),
            )
            .unwrap();
        files.verify().unwrap();
        let rows_per_shard = ENGRAM_ROWS / 2;
        for (index, shard) in table.shards.iter().enumerate() {
            assert_eq!(shard.rank, index as u32);
            assert_eq!(shard.first_row, (index * rows_per_shard) as u64);
            assert_eq!(shard.rows, rows_per_shard as u64);
            let weights = shard.weights.as_ref().unwrap().slice.as_ref().unwrap();
            let scales = shard.scales.as_ref().unwrap().slice.as_ref().unwrap();
            assert_eq!(weights.nbytes, (rows_per_shard * ENGRAM_WIDTH) as u64);
            assert_eq!(scales.nbytes, (rows_per_shard * ENGRAM_SCALES_PER_ROW) as u64);
            let mut bytes = vec![0u8; weights.nbytes as usize];
            files.get(&weights.payload_id).unwrap().read_at(0, &mut bytes).unwrap();
            assert_eq!(
                bytes,
                engram_tp2_shard(&tp1_weights, ENGRAM_WIDTH, index)
            );
            let mut bytes = vec![0u8; scales.nbytes as usize];
            files.get(&scales.payload_id).unwrap().read_at(0, &mut bytes).unwrap();
            assert_eq!(
                bytes,
                engram_tp2_shard(&tp1_scales, ENGRAM_SCALES_PER_ROW, index)
            );
        }

        let store = crate::package::cache::Root::open(scratch.join("partitioned-engram-cache")).unwrap();
        let source = crate::package::cache::PackageSource::open(
            &output,
            std::slice::from_ref(&publisher),
            protect::Selection::Deployment {
                deployment: 1,
                rank: None,
            },
        )
        .unwrap();
        let entry = store.install_package(source, "model", "tp2").unwrap();
        let installed = Metadata::open(entry.directory()).unwrap();
        let installed_verified = protect::verify(
            &installed.envelope,
            std::slice::from_ref(&publisher),
            protect::Selection::Deployment {
                deployment: 1,
                rank: None,
            },
        )
        .unwrap();
        let installed_table = &installed_verified
            .deployment()
            .unwrap()
            .engram
            .as_ref()
            .unwrap()
            .tables[0];
        assert_eq!(installed_table.rows, ENGRAM_ROWS as u64);
        assert_eq!(installed_table.shards.len(), 2);
        assert_eq!(installed_table.hash.as_ref().unwrap().offsets, engram_tp2_offsets());
        assert_eq!(
            installed_table.hash.as_ref().unwrap().multipliers,
            engram_multipliers()
        );
        let installed_catalog = installed_verified.catalog().clone();
        let installed_files = installed
            .payloads(
                &installed_catalog,
                installed_table.shards.iter().flat_map(|shard| {
                    [&shard.weights, &shard.scales]
                        .into_iter()
                        .flatten()
                        .filter_map(|use_| use_.slice.as_ref().map(|slice| slice.payload_id.as_slice()))
                }),
            )
            .unwrap();
        installed_files.verify().unwrap();
        for (index, shard) in installed_table.shards.iter().enumerate() {
            assert_eq!(shard.rank, index as u32);
            assert_eq!(shard.first_row, (index * rows_per_shard) as u64);
            assert_eq!(shard.rows, rows_per_shard as u64);
            let weights = shard.weights.as_ref().unwrap().slice.as_ref().unwrap();
            let scales = shard.scales.as_ref().unwrap().slice.as_ref().unwrap();
            let mut bytes = vec![0u8; weights.nbytes as usize];
            installed_files
                .get(&weights.payload_id)
                .unwrap()
                .read_at(0, &mut bytes)
                .unwrap();
            assert_eq!(
                bytes,
                engram_tp2_shard(&tp1_weights, ENGRAM_WIDTH, index)
            );
            let mut bytes = vec![0u8; scales.nbytes as usize];
            installed_files
                .get(&scales.payload_id)
                .unwrap()
                .read_at(0, &mut bytes)
                .unwrap();
            assert_eq!(
                bytes,
                engram_tp2_shard(&tp1_scales, ENGRAM_SCALES_PER_ROW, index)
            );
        }

        let mismatch = publish_source_impl(
            &scratch,
            "partitioned-mismatch",
            2,
            &[half0, half1],
            &[col0.as_slice(), col1.as_slice()],
            &key,
            &publisher,
            "model",
            EngramFixture::PartitionedMismatch,
            None,
        );
        let error = combine(&CombineRequest {
            sources: &[tp1.clone(), mismatch],
            output: &scratch.join("partitioned-mismatch-combined"),
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("source deployments disagree on engram semantics"),
            "{error}"
        );
    }

    // Exercises existing selected-closure verification for a corrupted source
    // payload. It does not claim a new publish_rank occurrence-selection bypass.
    #[test]
    fn combine_rejects_corrupted_source_payload_in_selected_closure() {
        use std::io::{Seek, SeekFrom};

        let scratch = Scratch::new("combine-corrupt-source-occurrence");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let shared = vec![0x11; 8];
        let other = vec![0x22; 8];
        let matrix = vec![0x33; 8];
        let other_column = vec![0x44; 8];
        let tp1 = publish_source_impl(
            &scratch,
            "corrupt-occurrence-tp1",
            1,
            &[shared.as_slice()],
            &[matrix.as_slice()],
            &key,
            &publisher,
            "model",
            EngramFixture::None,
            None,
        );
        let tp2 = publish_source_impl(
            &scratch,
            "corrupt-occurrence-tp2",
            2,
            &[shared.as_slice(), other.as_slice()],
            &[matrix.as_slice(), other_column.as_slice()],
            &key,
            &publisher,
            "model",
            EngramFixture::None,
            None,
        );
        let shared_id = PayloadId::of(&shared);
        let metadata = Metadata::open(&tp2).unwrap();
        let location = metadata
            .locations
            .locations
            .iter()
            .find(|location| {
                location.payload_id.as_slice() == shared_id.as_bytes() && !location.file.is_empty()
            })
            .unwrap();
        let path = metadata
            .path
            .parent()
            .unwrap()
            .join(&location.file);
        let original_len = fs::metadata(&path).unwrap().len();
        let offset = {
            let container = Container::open(&path).unwrap();
            let slice = container.tensor(&location.tensor).unwrap();
            let mut bytes = vec![0u8; shared.len()];
            slice.read_at(0, &mut bytes).unwrap();
            assert_eq!(bytes, shared);
            slice.offset()
        };
        let mut corrupt = shared.clone();
        corrupt[0] ^= 0xff;
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(&corrupt).unwrap();
        file.sync_all().unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), original_len);

        let output = scratch.join("corrupt-occurrence-combined");
        let error = combine(&CombineRequest {
            sources: &[tp1, tp2],
            output: &output,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("payload content does not match its signed identity"),
            "{error}"
        );
        assert!(!output.exists());
        let output_name = output.file_name().unwrap().to_string_lossy();
        let prefix = format!(".{output_name}.");
        let leftovers: Vec<_> = fs::read_dir(&scratch.0)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name())
            .filter(|name| name.to_string_lossy().starts_with(&prefix))
            .collect();
        assert!(leftovers.is_empty(), "temporary paths remain: {leftovers:?}");
    }

    #[test]
    fn promotes_engram_payloads_unchanged() {
        let scratch = Scratch::new("promote-engram");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let matrix: Vec<u8> = (10..18).collect();
        let full: Vec<u8> = (0..8).collect();
        let source = publish_source(
            &scratch,
            "engram-source",
            1,
            &[full.as_slice()],
            &[matrix.as_slice()],
            &key,
            &publisher,
            "model",
            true,
        );
        let before = Metadata::open(&source).unwrap();
        let before_verified = protect::verify(
            &before.envelope,
            std::slice::from_ref(&publisher),
            Selection::Deployment {
                deployment: 0,
                rank: None,
            },
        )
        .unwrap();
        let table = &before_verified
            .deployment()
            .unwrap()
            .engram
            .as_ref()
            .unwrap()
            .tables[0];
        let weights = table.weights.as_ref().unwrap().slice.as_ref().unwrap();
        let scales = table.scales.as_ref().unwrap().slice.as_ref().unwrap();
        let weights_id = weights.payload_id.clone();
        let scales_id = scales.payload_id.clone();
        let mut weights_bytes = vec![0u8; weights.nbytes as usize];
        let mut scales_bytes = vec![0u8; scales.nbytes as usize];
        before
            .payloads(
                before_verified.catalog(),
                [weights_id.as_slice(), scales_id.as_slice()],
            )
            .unwrap()
            .get(&weights_id)
            .unwrap()
            .read_at(weights.offset, &mut weights_bytes)
            .unwrap();
        let before_files = before
            .payloads(before_verified.catalog(), [scales_id.as_slice()])
            .unwrap();
        before_files
            .get(&scales_id)
            .unwrap()
            .read_at(scales.offset, &mut scales_bytes)
            .unwrap();

        let output = scratch.join("engram-promoted");
        promote(
            &source,
            &output,
            std::slice::from_ref(&publisher),
            "example",
            "example",
            None,
            &key,
        )
        .unwrap();
        let after = Metadata::open(&output).unwrap();
        let after_verified = protect::verify(
            &after.envelope,
            std::slice::from_ref(&publisher),
            Selection::Deployment {
                deployment: 0,
                rank: None,
            },
        )
        .unwrap();
        let table = &after_verified
            .deployment()
            .unwrap()
            .engram
            .as_ref()
            .unwrap()
            .tables[0];
        let weights = table.weights.as_ref().unwrap().slice.as_ref().unwrap();
        let scales = table.scales.as_ref().unwrap().slice.as_ref().unwrap();
        assert_eq!(weights.payload_id, weights_id);
        assert_eq!(scales.payload_id, scales_id);
        let files = after
            .payloads(
                after_verified.catalog(),
                [weights_id.as_slice(), scales_id.as_slice()],
            )
            .unwrap();
        let mut promoted_weights = vec![0u8; weights.nbytes as usize];
        let mut promoted_scales = vec![0u8; scales.nbytes as usize];
        files
            .get(&weights_id)
            .unwrap()
            .read_at(weights.offset, &mut promoted_weights)
            .unwrap();
        files
            .get(&scales_id)
            .unwrap()
            .read_at(scales.offset, &mut promoted_scales)
            .unwrap();
        assert_eq!(promoted_weights, weights_bytes);
        assert_eq!(promoted_scales, scales_bytes);
    }

    #[test]
    fn combines_multi_tp_packages_into_one_signed_artifact() {
        let scratch = Scratch::new("combine");
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let full: Vec<u8> = (0..8).collect();
        let matrix: Vec<u8> = (10..18).collect();
        let half0 = &full[0..4];
        let half1 = &full[4..8];
        let col0 = [10u8, 12, 14, 16];
        let col1 = [11u8, 13, 15, 17];
        let tp1 = publish_source(
            &scratch,
            "tp1",
            1,
            &[full.as_slice()],
            &[matrix.as_slice()],
            &key,
            &publisher,
            "model",
            false,
        );
        let tp2 = publish_source(
            &scratch,
            "tp2",
            2,
            &[half0, half1],
            &[col0.as_slice(), col1.as_slice()],
            &key,
            &publisher,
            "model",
            false,
        );
        let output = scratch.join("combined");
        let combined = combine(&CombineRequest {
            sources: &[tp2.clone(), tp1.clone()],
            output: &output,
            publishers: std::slice::from_ref(&publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap();
        assert_eq!(combined.payloads.len() > 0, true);
        let metadata = Metadata::open(&output).unwrap();
        let tp1_view = protect::verify(
            &metadata.envelope,
            std::slice::from_ref(&publisher),
            Selection::Deployment {
                deployment: 0,
                rank: None,
            },
        )
        .unwrap();
        assert_eq!(tp1_view.deployment().unwrap().deployment_id, "sm120-tp1");
        let canonical_weights = PayloadId::of(&full);
        let canonical_location = metadata
            .locations
            .locations
            .iter()
            .find(|location| location.payload_id == canonical_weights.as_bytes().to_vec())
            .unwrap();
        assert_eq!(canonical_location.file, "weights/layer-000.safetensors");
        let tp2_view = protect::verify(
            &metadata.envelope,
            std::slice::from_ref(&publisher),
            Selection::Deployment {
                deployment: 1,
                rank: None,
            },
        )
        .unwrap();
        assert_eq!(tp2_view.deployment().unwrap().deployment_id, "sm120-tp2");
        assert_eq!(tp2_view.ranks().len(), 2);
        // Derived ranks keep their signed target payload ids; the combined
        // package omits those payloads and carries derivation recipes instead.
        let catalog = tp2_view.catalog().clone();
        assert!(
            metadata
                .primary()
                .tensor_names()
                .any(|name| name == reader::DERIVATIONS)
        );
        for rank in tp2_view.ranks() {
            let initializer = &rank.initializers[1];
            let slice = initializer.source.as_ref().unwrap().slice.as_ref().unwrap();
            assert_eq!(slice.offset, 0);
            assert_eq!(slice.nbytes, 4);
            assert!(
                metadata
                    .locations
                    .locations
                    .iter()
                    .all(|location| location.payload_id != slice.payload_id),
                "derived payload must not be installed"
            );
        }
        let set = wire::DerivationSet::decode(
            tensor_bytes(metadata.primary(), reader::DERIVATIONS).as_slice(),
        )
        .unwrap();
        let recipes =
            derivation::from_wire(&set, &catalog, &derivation::Limits::default()).unwrap();
        assert_eq!(recipes.len(), 4);
        assert!(recipes.iter().any(|recipe| matches!(
            recipe.operations.first(),
            Some(derivation::Operation::Copy(copy)) if !copy.outer.is_empty()
        )));
        let mut source_ids = BTreeSet::new();
        for recipe in &recipes {
            for operation in &recipe.operations {
                if let derivation::Operation::Copy(copy) = operation {
                    source_ids.insert(copy.source);
                }
            }
        }
        let sources = metadata
            .payloads(
                &catalog,
                source_ids.iter().map(|id| id.as_bytes().as_slice()),
            )
            .unwrap();
        let mut source_map = BTreeMap::new();
        for id in source_ids {
            source_map.insert(id, sources.get(id.as_bytes()).unwrap().clone());
        }
        for rank in tp2_view.ranks() {
            for source in source_map.keys() {
                assert!(
                    rank.payloads
                        .iter()
                        .any(|descriptor| descriptor.id == source.as_bytes().to_vec()),
                    "rank {} must authenticate shared source {:?}",
                    rank.rank,
                    source
                );
            }
            for (index, expected) in [
                (
                    1usize,
                    if rank.rank == 0 {
                        full[0..4].to_vec()
                    } else {
                        full[4..8].to_vec()
                    },
                ),
                (
                    2usize,
                    if rank.rank == 0 {
                        col0.to_vec()
                    } else {
                        col1.to_vec()
                    },
                ),
            ] {
                let id = PayloadId::from_bytes(
                    rank.initializers[index]
                        .source
                        .as_ref()
                        .unwrap()
                        .slice
                        .as_ref()
                        .unwrap()
                        .payload_id
                        .as_slice()
                        .try_into()
                        .unwrap(),
                );
                let recipe = recipes
                    .iter()
                    .find(|recipe| recipe.target == id)
                    .expect("derived target has a recipe");
                let path = scratch.join(&format!("materialized-{}-{index}.bin", rank.rank));
                derivation::materialize(&path, recipe, &source_map, &derivation::Limits::default())
                    .unwrap();
                assert_eq!(fs::read(&path).unwrap(), expected);
                assert_eq!(PayloadId::of(&expected), id);
            }
        }
        // Select only TP2 through the cache install path; materialized target payloads must be
        // resolved through each rank's signed initializer slice and match the exported shard.
        let cache_root = scratch.join("tp2-cache");
        let root = crate::package::cache::Root::open(&cache_root).unwrap();
        let source = crate::package::cache::PackageSource::open(
            &output,
            std::slice::from_ref(&publisher),
            Selection::Deployment {
                deployment: 1,
                rank: None,
            },
        )
        .unwrap();
        let entry = root.install_package(source, "model", "tp2").unwrap();
        let installed = Metadata::open(entry.directory()).unwrap();
        let view = protect::verify(
            &installed.envelope,
            std::slice::from_ref(&publisher),
            Selection::Deployment {
                deployment: 1,
                rank: None,
            },
        )
        .unwrap();
        assert_eq!(view.deployment().unwrap().deployment_id, "sm120-tp2");
        for rank in view.ranks() {
            for (index, expected) in [
                (
                    1usize,
                    if rank.rank == 0 { full[0..4].to_vec() } else { full[4..8].to_vec() },
                ),
                (
                    2usize,
                    if rank.rank == 0 { col0.to_vec() } else { col1.to_vec() },
                ),
            ] {
                let slice = rank.initializers[index]
                    .source.as_ref().unwrap().slice.as_ref().unwrap();
                let id = PayloadId::from_bytes(slice.payload_id.as_slice().try_into().unwrap());
                assert_eq!(id, PayloadId::of(&expected));
                let location = installed
                    .locations
                    .locations
                    .iter()
                    .find(|location| location.payload_id == slice.payload_id)
                    .unwrap_or_else(|| panic!("install lacks rank {} initializer {index}", rank.rank));
                let payload = if location.file.is_empty() {
                    installed.primary().tensor(&location.tensor).unwrap()
                } else {
                    entry.payload(&location.file).unwrap()
                };
                let mut bytes = vec![0u8; slice.nbytes as usize];
                payload.read_at(slice.offset, &mut bytes).unwrap();
                assert_eq!(
                    bytes, expected,
                    "rank {} initializer {index} must bind its exported shard",
                    rank.rank
                );
            }
        }
    }

    #[test]
    fn encoded_piece_packing_allows_rotation_but_not_other_semantics() {
        let mut base = prepared_owner("tensor");
        base.packing.as_mut().unwrap().row_rotation = 256;
        base.sources[0].piece_encoding = "w4a16_mma_k16_n64_v1".into();
        base.sources[0].pieces = vec![wire::WeightPiece {
            source_offset: 0,
            destination_offset: 0,
            contiguous_bytes: 16,
            extents: vec![],
            source_strides: vec![],
            destination_strides: vec![],
        }];
        let mut derived = base.clone();
        derived.packing.as_mut().unwrap().row_rotation = 128;
        assert!(piece_compatible(&base, &derived));
        derived.packing.as_mut().unwrap().group_size = 32;
        assert!(!piece_compatible(&base, &derived));
        derived.packing.as_mut().unwrap().group_size = 0;
        derived.packing.as_mut().unwrap().format = "other".into();
        assert!(!piece_compatible(&base, &derived));
        derived.packing.as_mut().unwrap().format = "nvfp4".into();
        derived.sources[0].piece_encoding = "other".into();
        assert!(!piece_compatible(&base, &derived));
    }

    #[test]
    fn combiner_plans_prepared_owners_across_deployments() {
        let base_scratch = Scratch::new("plan-base");
        let base = fixture(&base_scratch);
        let canonical = prepared_ranks(&base.directory, 0).unwrap();
        assert_eq!(canonical.len(), 1);
        assert_eq!(canonical[0].owners.len(), 1);

        let prefix_scratch = Scratch::new("plan-prefix");
        let prefix = fixture_with(&prefix_scratch, b"stable-weight-bytes-tail", 19, 64);
        let prefix = prepared_ranks(&prefix.directory, 0).unwrap();
        let plan = plan_rank(&canonical[0], &prefix[0], &derivation::Limits::default()).unwrap();
        assert_eq!(
            plan.owners["layer-000"],
            OwnerShare::Independent,
            "a partial owner must not claim a larger target payload"
        );
        assert!(plan.recipes.is_empty());

        let identical_scratch = Scratch::new("plan-identical");
        let identical = fixture(&identical_scratch);
        let identical = prepared_ranks(&identical.directory, 0).unwrap();
        let plan = plan_rank(&canonical[0], &identical[0], &derivation::Limits::default()).unwrap();
        assert!(matches!(
            plan.owners["layer-000"],
            OwnerShare::Identical { offset: 0, .. }
        ));

        let other_scratch = Scratch::new("plan-other");
        let other = fixture_with(&other_scratch, b"different-weight-xxx", 19, 64);
        let other = prepared_ranks(&other.directory, 0).unwrap();
        let plan = plan_rank(&canonical[0], &other[0], &derivation::Limits::default()).unwrap();
        assert_eq!(plan.owners["layer-000"], OwnerShare::Independent);

        let packing_scratch = Scratch::new("plan-packing");
        let packing = fixture_with(&packing_scratch, b"stable-weight-bytes-tail", 19, 128);
        let packing = prepared_ranks(&packing.directory, 0).unwrap();
        let plan = plan_rank(&canonical[0], &packing[0], &derivation::Limits::default()).unwrap();
        assert_eq!(plan.owners["layer-000"], OwnerShare::Independent);
    }

    #[test]
    fn selected_closures_do_not_require_unselected_payloads() {
        let scratch = Scratch::new("selection");
        let fixture = fixture(&scratch);
        drop_location(&fixture.directory, "deployments/");
        assert!(pack(&fixture.directory, &scratch.join("all.safetensors")).is_err());
        assert!(
            pack_selected(
                &fixture.directory,
                &scratch.join("dep0.safetensors"),
                Some(&[0])
            )
            .is_err()
        );
        let packed = scratch.join("selected.safetensors");
        pack_selected(&fixture.directory, &packed, Some(&[])).unwrap();
        let container = Container::open(&packed).unwrap();
        verify(&tensor_bytes(&container, reader::MANIFEST), &fixture);
        let unpacked = scratch.join("selected-directory");
        unpack_selected(&packed, &unpacked, Some(&[])).unwrap();
        assert!(Metadata::open(&unpacked).is_ok());
    }

    #[test]
    fn existing_outputs_are_never_removed_by_a_failed_conversion() {
        let scratch = Scratch::new("guard");
        let fixture = fixture(&scratch);
        let sentinel = b"existing user package".to_vec();

        let packed = scratch.join("packed.safetensors");
        fs::write(&packed, &sentinel).unwrap();
        assert!(pack(&fixture.directory, &packed).is_err());
        assert_eq!(fs::read(&packed).unwrap(), sentinel);

        let source = scratch.join("source-packed.safetensors");
        pack(&fixture.directory, &source).unwrap();
        let out = scratch.join("existing-directory");
        fs::create_dir_all(&out).unwrap();
        fs::write(out.join("model.safetensors"), &sentinel).unwrap();
        assert!(unpack(&source, &out).is_err());
        assert_eq!(fs::read(out.join("model.safetensors")).unwrap(), sentinel);
    }

    #[test]
    fn tampered_payload_and_incomplete_closure_fail_closed() {
        let scratch = Scratch::new("tamper");
        let tampered = fixture(&scratch);
        let companion = tampered.directory.join("weights/layer-000.safetensors");
        let mut bytes = fs::read(&companion).unwrap();
        let end = bytes.len() - 1;
        bytes[end] ^= 0xff;
        fs::write(&companion, bytes).unwrap();
        assert!(pack(&tampered.directory, &scratch.join("bad.safetensors")).is_err());
        assert!(!scratch.join("bad.safetensors").exists());

        // Drop the weights location but keep the derivation source needing it.
        let closure_scratch = Scratch::new("closure");
        let clean = fixture(&closure_scratch);
        let primary_path = clean.directory.join("model.safetensors");
        let container = Container::open(&primary_path).unwrap();
        let mut tensors: Vec<(String, Vec<u8>)> = Vec::new();
        for name in container.tensor_names() {
            if !name.starts_with(container::ALIGNMENT_PREFIX) && name != reader::LOCATIONS {
                tensors.push((name.to_owned(), tensor_bytes(&container, name)));
            }
        }
        drop(container);
        let mut set = wire::LocationSet::decode(
            wire::LocationSet::decode(
                tensor_bytes(&Container::open(&primary_path).unwrap(), reader::LOCATIONS)
                    .as_slice(),
            )
            .unwrap()
            .encode_to_vec()
            .as_slice(),
        )
        .unwrap();
        set.locations.retain(|location| {
            location.tensor != COMPANION_TENSOR || !location.file.starts_with("weights/")
        });
        let rewritten = set.encode_to_vec();
        let mut writer = Writer::new(
            File::create(&primary_path).unwrap(),
            container::PAYLOAD_FORMAT,
        )
        .unwrap();
        for (name, bytes) in tensors {
            if name == reader::LOCATIONS {
                continue;
            }
            writer
                .add(&name, bytes.len() as u64, Source::Bytes(bytes))
                .unwrap();
        }
        writer
            .add(
                reader::LOCATIONS,
                rewritten.len() as u64,
                Source::Bytes(rewritten),
            )
            .unwrap();
        writer.finish().unwrap();
        assert!(pack(&clean.directory, &scratch.join("closure.safetensors")).is_err());
    }
}
