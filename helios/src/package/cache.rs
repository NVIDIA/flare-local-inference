/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Immutable payload model cache: content-addressed objects, published directory
//! entries, process-owned leases, and crash-safe reclamation.
//!
//! ```text
//! blobs/<payload-sha256>.safetensors            payload-format companions
//! blobs/legacy/<container-sha256>.safetensors   legacy immutable containers
//! models/<model-key>/<entry-id>/...             published directory entries
//! refs/<model-key>                              mutable selection of one entry
//! sources/<selector-sha256>                    acquisition selectors to entries
//! staging/{trees,incoming,wrapped}/<txn>        unpublished work
//! locks/{publication,models,entries,blobs,staging}/...  stable lock inodes
//! ```
//!
//! A published entry is a complete directory canonical form whose companions are
//! hardlinks to immutable objects, so entries share one copy of each payload.
//! Leases are process-owned file locks released by the kernel on process death.
//! Reachability follows published entries and live staging trees, never a ref.
//!
//! Lock order is publication, then object, then model or entry: publishers and
//! object holders take `publication` shared, and collection takes it exclusive,
//! so a reachability snapshot cannot race an install or a staging link.

use prost::Message as _;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(windows)]
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use ring::digest::{Context as DigestContext, SHA256};

use super::{PayloadId, derivation, inverse, invalid, progress, protection, reader, schema::{self, wire}};

#[cfg(test)]
thread_local! {
    static PAYLOAD_OPENS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
use crate::Result;
use crate::package::container::{
    COMPANION_TENSOR, Container, PAYLOAD_FORMAT, Slice, Source, Writer, open_relative,
    validate_relative_path, write_companion,
};

const MAX_NAME: usize = 128;

static TRANSACTION: AtomicU64 = AtomicU64::new(0);

/// Lossless, filesystem-safe cache namespace for a model identity.
///
/// `_` is escaped first, then every byte outside `[A-Za-z0-9.-]` becomes `_XX`,
/// so distinct identities (including `a/b`, `a_2Fb`, and `a%2Fb`) never collide.
/// The encoded form fits the name limit the cache validator enforces.
pub fn model_key(identity: &str) -> Result<String> {
    if identity.is_empty() {
        return Err("model identity is empty".into());
    }
    if identity == "." || identity == ".." {
        return Err("model identity is not a usable cache name".into());
    }
    let mut key = String::with_capacity(identity.len());
    for byte in identity.bytes() {
        if byte.is_ascii_alphanumeric() || byte == b'.' || byte == b'-' {
            key.push(byte as char);
        } else {
            key.push('_');
            key.push(
                char::from_digit(u32::from(byte >> 4), 16)
                    .unwrap()
                    .to_ascii_uppercase(),
            );
            key.push(
                char::from_digit(u32::from(byte & 0xf), 16)
                    .unwrap()
                    .to_ascii_uppercase(),
            );
        }
    }
    validate_name(&key, "model key")?;
    Ok(key)
}

/// Summary for a legacy container entry, read from its unverified metadata document.
fn legacy_summary(root: &Root, entry: &Entry) -> Result<EntrySummary> {
    let value = crate::runtime_package::unverified_metadata(
        &root.primary(entry)?,
        crate::runtime_package::InspectionOptions::default(),
        None,
    )?;
    let model = value.get("model").cloned().unwrap_or_default();
    let model_identity = model
        .get("identity")
        .and_then(|value| value.as_str())
        .or_else(|| model.get("model_id").and_then(|value| value.as_str()))
        .unwrap_or("")
        .to_owned();
    let hardware_identity = value
        .get("package")
        .and_then(|package| package.get("architecture"))
        .map(|value| value.to_string())
        .unwrap_or_default();
    let ranks = value
        .get("ranks")
        .and_then(|ranks| ranks.as_array())
        .map_or(0, |ranks| ranks.len()) as u32;
    let cache = value.get("cache").cloned().unwrap_or_default();
    Ok(EntrySummary {
        form: "legacy",
        format_version: 4,
        model_identity,
        hardware_identity,
        deployments: vec![DeploymentSummary {
            id: "legacy".into(),
            tensor_parallel: ranks,
            ready: true,
            pages: cache.get("pages").and_then(|value| value.as_u64()),
            state_rows: cache.get("state_rows").and_then(|value| value.as_u64()),
            max_sequence_tokens: model
                .get("max_sequence_tokens")
                .and_then(|value| value.as_u64()),
            prefill_chunk_size: model
                .get("prefill_chunk_size")
                .and_then(|value| value.as_u64()),
        }],
        logical_bytes: 0,
        payloads: 0,
    })
}

/// Decode a cache model key back to its model identity.
///
/// The encoding is reversible (`_XX` per escaped byte); a key that does not round-trip through
/// [`model_key`] is rejected, so display never invents an identity.
pub fn model_identity(key: &str) -> Result<String> {
    let bytes = key.as_bytes();
    // Decode to bytes, not chars: `model_key` escapes UTF-8 bytes, so a multi-byte
    // identity must be reassembled before it is interpreted as text.
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte == b'_' {
            let hex = |value: u8| -> Option<u8> {
                match value {
                    b'0'..=b'9' => Some(value - b'0'),
                    b'A'..=b'F' => Some(value - b'A' + 10),
                    _ => None,
                }
            };
            let (Some(high), Some(low)) = (
                bytes.get(index + 1).copied().and_then(hex),
                bytes.get(index + 2).copied().and_then(hex),
            ) else {
                return Err(format!("cache key {key:?} is not a model key").into());
            };
            decoded.push(high << 4 | low);
            index += 3;
        } else {
            decoded.push(byte);
            index += 1;
        }
    }
    let identity = String::from_utf8(decoded)
        .map_err(|_| crate::Error::from(format!("cache key {key:?} is not a model key")))?;
    if model_key(&identity)? != key {
        return Err(format!("cache key {key:?} is not a model key").into());
    }
    Ok(identity)
}

/// An immutable payload object retained in the object store.
///
/// The shared object lock keeps the store entry reachable until the handle is
/// dropped, so collection cannot unlink a blob between installation and the
/// staging link that makes it part of an entry.
#[derive(Clone)]
pub struct PayloadObject {
    id: PayloadId,
    path: PathBuf,
    slice: Slice,
    _guard: Arc<Lock>,
}

impl PayloadObject {
    pub fn id(&self) -> PayloadId {
        self.id
    }

    pub fn length(&self) -> u64 {
        self.slice.len()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Pinned handle to the object file, valid while this object lives.
    pub fn file(&self) -> &Arc<File> {
        self.slice.file()
    }

    /// Payload bytes as a pinned slice; callers bound their own reads.
    pub fn slice(&self) -> &Slice {
        &self.slice
    }
}

/// The configured cache root.
#[derive(Clone)]
pub struct Root {
    root: PathBuf,
}

fn located_payload_ids(metadata: &reader::Metadata) -> BTreeSet<Vec<u8>> {
    metadata
        .locations
        .locations
        .iter()
        .map(|location| location.payload_id.clone())
        .collect()
}

fn selected_derivation_ids(set: &wire::DerivationSet) -> BTreeSet<Vec<u8>> {
    let mut ids = BTreeSet::new();
    for hint in &set.derivations {
        ids.insert(hint.target_payload_id.clone());
        for operation in &hint.operations {
            if let Some(wire::derivation_operation::Operation::Copy(copy)) = &operation.operation {
                ids.insert(copy.source_payload_id.clone());
            }
        }
    }
    ids
}

/// Keep only recipes reachable from missing selected targets. Recipes for other
/// deployments never enter the catalog, walk, or parser.
fn selected_derivation_set(
    set: &wire::DerivationSet,
    targets: &BTreeSet<Vec<u8>>,
    located: &BTreeSet<Vec<u8>>,
    limits: &derivation::Limits,
) -> Result<wire::DerivationSet> {
    if targets.is_empty() {
        return Ok(wire::DerivationSet {
            version: set.version,
            derivations: Vec::new(),
        });
    }
    if set.derivations.len() > limits.max_derivations {
        return Err(invalid("derivation set exceeds its entry limit"));
    }
    let mut hints: BTreeMap<Vec<u8>, Vec<&wire::Derivation>> = BTreeMap::new();
    for hint in &set.derivations {
        hints
            .entry(hint.target_payload_id.clone())
            .or_default()
            .push(hint);
    }
    let mut selected = BTreeMap::new();
    let mut visited = BTreeSet::new();
    let mut frontier: Vec<Vec<u8>> = targets.iter().cloned().collect();
    while let Some(target) = frontier.pop() {
        if located.contains(&target) || !visited.insert(target.clone()) {
            continue;
        }
        let candidates = hints.get(&target).ok_or_else(|| {
            invalid("payload is neither located nor derivable by a selected deployment")
        })?;
        if candidates.len() != 1 {
            return Err(invalid("duplicate derivation target"));
        }
        let hint = candidates[0];
        if hint.operations.len() > limits.max_operations {
            return Err(invalid("derivation exceeds its operation limit"));
        }
        selected.insert(target.clone(), hint);
        for operation in &hint.operations {
            if let Some(wire::derivation_operation::Operation::Copy(copy)) = &operation.operation {
                if !located.contains(&copy.source_payload_id) {
                    frontier.push(copy.source_payload_id.clone());
                }
            }
        }
    }
    Ok(wire::DerivationSet {
        version: set.version,
        derivations: selected.into_values().cloned().collect(),
    })
}

fn package_asset_payloads(verified: &protection::VerifiedPackage) -> BTreeSet<Vec<u8>> {
    verified
        .package()
        .assets
        .iter()
        .filter_map(|asset| asset.payload.as_ref())
        .filter_map(|use_| use_.slice.as_ref())
        .map(|slice| slice.payload_id.clone())
        .collect()
}

/// Choose one deterministic, unique companion path for a payload location.
///
/// Directory packages normally place one payload per companion. A size-sharded
/// container may place several payload tensors in one file; those locations keep
/// the container directory and name but gain a tensor-derived leaf. Packed
/// locations use their packed tensor mapping. Collisions append the payload
/// identity, so different payloads never target one entry path.
fn companion_name(
    file: &str,
    tensor: &str,
    identity: PayloadId,
    used: &mut BTreeSet<String>,
) -> Result<String> {
    let primary = "model.safetensors";
    let candidate = crate::package::container::normalized_payload_companion(file, tensor, identity);
    if candidate != primary
        && crate::package::container::validate_relative_path(&candidate).is_ok()
        && !crate::package::container::path_conflicts(&candidate, used)
    {
        used.insert(candidate.clone());
        return Ok(candidate);
    }
    let (parent, name) = match candidate.rsplit_once('/') {
        Some((parent, name)) => (parent, name),
        None => ("", candidate.as_str()),
    };
    let stem = name.strip_suffix(".safetensors").unwrap_or(name);
    let fallback = if parent.is_empty() {
        format!("{stem}-{}.safetensors", identity.to_hex())
    } else {
        format!("{parent}/{stem}-{}.safetensors", identity.to_hex())
    };
    if fallback != primary
        && crate::package::container::validate_relative_path(&fallback).is_ok()
        && !crate::package::container::path_conflicts(&fallback, used)
    {
        used.insert(fallback.clone());
        return Ok(fallback);
    }
    let fallback = format!("companions/{}", identity.file_name());
    if !crate::package::container::path_conflicts(&fallback, used) {
        used.insert(fallback.clone());
        return Ok(fallback);
    }
    Err(format!("cannot assign a unique companion path to payload {identity}").into())
}

impl Root {
    /// Valid optional derivation sources for already located selected targets.
    ///
    /// Required missing targets follow the strict closure path. This walk starts
    /// only from selected required targets that are present, follows unique,
    /// authenticated hints through missing intermediates, and returns available
    /// (supplied or cached) sources with their authenticated lengths. It is
    /// bounded by the derivation limits; malformed, ambiguous, cyclic, or
    /// unavailable optional branches are ignored and cannot fail the ready target.
    fn optional_derivation_sources(
        &self,
        source: &PackageSource<'_>,
        required: &BTreeSet<Vec<u8>>,
        placed: &BTreeMap<Vec<u8>, (String, String)>,
    ) -> BTreeMap<Vec<u8>, u64> {
        let mut retained = BTreeMap::new();
        let Some(set) = source.metadata.derivations.as_ref() else {
            return retained;
        };
        let limits = derivation::Limits::default();
        if set.derivations.len() > limits.max_derivations {
            return retained;
        }
        let mut hints: BTreeMap<Vec<u8>, Vec<&wire::Derivation>> = BTreeMap::new();
        for hint in &set.derivations {
            hints
                .entry(hint.target_payload_id.clone())
                .or_default()
                .push(hint);
        }
        let ids_limit = limits.max_operations.saturating_add(1);
        let mut visited = BTreeSet::new();
        let mut enqueued = BTreeSet::new();
        let mut stack: Vec<(Vec<u8>, usize)> = required
            .iter()
            .filter(|target| placed.contains_key(*target))
            .map(|target| (target.clone(), 0))
            .collect();
        for (target, _) in &stack {
            enqueued.insert(target.clone());
        }
        let mut availability: BTreeMap<Vec<u8>, bool> = BTreeMap::new();
        while let Some((target, depth)) = stack.pop() {
            if depth > limits.max_depth || !visited.insert(target.clone()) {
                continue;
            }
            if visited.len() > limits.max_derivations {
                break;
            }
            let Some(candidates) = hints.get(&target) else {
                continue;
            };
            if candidates.len() != 1 {
                continue;
            }
            let hint = candidates[0];
            if hint.operations.len() > limits.max_operations {
                continue;
            }
            let mut ids = BTreeSet::from([target.clone()]);
            for operation in &hint.operations {
                if let Some(wire::derivation_operation::Operation::Copy(copy)) = &operation.operation {
                    ids.insert(copy.source_payload_id.clone());
                    if ids.len() > ids_limit {
                        break;
                    }
                }
            }
            if ids.len() > ids_limit {
                continue;
            }
            let Ok(catalog) = Self::merged_catalog(&source.verified, ids.iter().cloned()) else {
                continue;
            };
            let subset = wire::DerivationSet {
                version: set.version,
                derivations: vec![hint.clone()],
            };
            let Ok(plans) = derivation::from_wire(&subset, &catalog, &limits) else {
                continue;
            };
            if derivation::validate_set(&plans, &limits).is_err() {
                continue;
            }
            for id in ids.iter().filter(|id| **id != target) {
                let Ok(bytes) = <[u8; 32]>::try_from(id.as_slice()) else {
                    continue;
                };
                let identity = PayloadId::from_bytes(bytes);
                let available = *availability.entry(id.clone()).or_insert_with(|| {
                    placed.contains_key(id) || self.payload(identity).ok().flatten().is_some()
                });
                if available
                    && let Ok(length) = catalog.length(id.as_slice())
                {
                    retained.insert(id.clone(), length);
                }
                if depth < limits.max_depth && enqueued.insert(id.clone()) {
                    stack.push((id.clone(), depth + 1));
                }
            }
        }
        retained
    }
}

/// Resolved acquisition work for one authenticated package selection.
///
/// The plan owns the verified source, retained cache-object leases, the selected
/// catalog, ordered derivations, and the minimal remote payload locations. Fields
/// are private so callers cannot present a plan built from different metadata.
pub struct AcquisitionPlan<'a> {
    source: PackageSource<'a>,
    required: BTreeMap<Vec<u8>, (PayloadId, u64)>,
    catalog: schema::Catalog,
    available: BTreeMap<PayloadId, PayloadObject>,
    derivations: Vec<derivation::Derivation>,
    remote: BTreeMap<Vec<u8>, (String, String)>,
    local: BTreeSet<Vec<u8>>,
    retained_available: BTreeSet<PayloadId>,
    progress: Option<progress::AcquisitionProgress>,
}

impl AcquisitionPlan<'_> {
    /// Minimal non-primary files that must be fetched before installation.
    pub fn remote_files(&self) -> Vec<String> {
        let files: BTreeSet<String> = self
            .remote
            .values()
            .filter(|(file, _)| !file.is_empty())
            .map(|(file, _)| file.clone())
            .collect();
        files.into_iter().collect()
    }

    /// Rebind the plan to a transport copy whose primary and relative companion
    /// layout were relocated. The signed envelope, unsigned locations, and
    /// derivations must match exactly before the plan can be installed.
    pub fn relocate_source(&mut self, primary: &Path) -> Result<()> {
        let relocated = reader::Metadata::open(primary)?;
        if relocated.envelope != self.source.metadata.envelope
            || relocated.locations != self.source.metadata.locations
            || relocated.derivations != self.source.metadata.derivations
        {
            return Err("relocated transport metadata differs from the planned package".into());
        }
        // Equal signed locations do not by themselves prove embedded frontend
        // asset bytes. Verify each distinct embedded payload against its whole
        // authenticated identity; asset uses may address a subrange, but the
        // payload digest always covers the complete stored tensor.
        let mut checked: BTreeSet<Vec<u8>> = BTreeSet::new();
        for verified in &self.source.verified {
            for asset in &verified.package().assets {
                let Some(slice) = asset.payload.as_ref().and_then(|use_| use_.slice.as_ref()) else {
                    continue;
                };
                if !checked.insert(slice.payload_id.clone()) {
                    continue;
                }
                let Some(location) = relocated
                    .locations
                    .locations
                    .iter()
                    .find(|location| location.payload_id == slice.payload_id)
                else {
                    continue;
                };
                if !location.file.is_empty() {
                    continue;
                }
                let length = verified.catalog().length(&slice.payload_id)?;
                let tensor = relocated.primary().tensor(&location.tensor)?;
                if tensor.len() != length {
                    return Err("relocated embedded asset has the wrong payload length".into());
                }
                let digest = tensor.digest_with(self.progress.as_ref())?;
                if let Some(progress) = &self.progress {
                    progress.record_local_source_bytes(tensor.len());
                }
                if digest
                    != PayloadId::from_bytes(
                        slice
                            .payload_id
                            .as_slice()
                            .try_into()
                            .map_err(|_| "embedded asset identity is not 32 bytes".to_string())?,
                    )
                {
                    return Err("relocated embedded asset bytes differ from their signed identity".into());
                }
            }
        }
        self.source.metadata = relocated;
        Ok(())
    }
}

fn local_forward_closure(
    target: PayloadId,
    forward: &BTreeMap<PayloadId, derivation::Derivation>,
    resolved: &BTreeSet<PayloadId>,
    memo: &mut BTreeMap<PayloadId, Option<Vec<derivation::Derivation>>>,
    visiting: &mut BTreeSet<PayloadId>,
) -> Option<Vec<derivation::Derivation>> {
    if resolved.contains(&target) {
        return Some(Vec::new());
    }
    if let Some(cached) = memo.get(&target) {
        return cached.clone();
    }
    if !visiting.insert(target) {
        return None;
    }
    let Some(plan) = forward.get(&target) else {
        visiting.remove(&target);
        memo.insert(target, None);
        return None;
    };
    let mut dependencies = Vec::new();
    for operation in &plan.operations {
        let derivation::Operation::Copy(copy) = operation else {
            continue;
        };
        if resolved.contains(&copy.source) {
            continue;
        }
        let Some(mut dependency) =
            local_forward_closure(copy.source, forward, resolved, memo, visiting)
        else {
            visiting.remove(&target);
            memo.insert(target, None);
            return None;
        };
        dependencies.append(&mut dependency);
    }
    dependencies.push(plan.clone());
    visiting.remove(&target);
    memo.insert(target, Some(dependencies.clone()));
    Some(dependencies)
}

impl Root {
    /// Resolve cache, remote, forward, and inverse acquisition work without
    /// opening any located source bytes.
    pub fn plan_acquisition<'a>(
        &self,
        source: PackageSource<'a>,
        limits: &derivation::Limits,
    ) -> Result<AcquisitionPlan<'a>> {
        self.plan_acquisition_with(source, limits, None)
    }

    /// Resolve acquisition work with a shared progress handle. The handle can
    /// begin before Hub transport fetch and remains observable through install.
    pub fn plan_acquisition_with<'a>(
        &self,
        source: PackageSource<'a>,
        limits: &derivation::Limits,
        progress: Option<&progress::AcquisitionProgress>,
    ) -> Result<AcquisitionPlan<'a>> {
        let progress = progress.cloned();
        let placed: BTreeMap<Vec<u8>, (String, String)> = source
            .metadata
            .locations
            .locations
            .iter()
            .map(|location| {
                (
                    location.payload_id.clone(),
                    (location.file.clone(), location.tensor.clone()),
                )
            })
            .collect();
        let mut required: BTreeMap<Vec<u8>, (PayloadId, u64)> = BTreeMap::new();
        for verified in &source.verified {
            if !verified.is_executable() {
                return Err("installation requires a selected prepared deployment".into());
            }
            for id in reader::required_payloads(verified)? {
                let identity = PayloadId::from_bytes(
                    id.as_slice()
                        .try_into()
                        .map_err(|_| "payload identity is not 32 bytes".to_string())?,
                );
                let length = verified.catalog().length(&id)?;
                match required.entry(id) {
                    std::collections::btree_map::Entry::Vacant(slot) => {
                        slot.insert((identity, length));
                    }
                    std::collections::btree_map::Entry::Occupied(slot)
                        if slot.get().1 != length =>
                    {
                        return Err("selections disagree about a payload length".into());
                    }
                    std::collections::btree_map::Entry::Occupied(_) => {}
                }
            }
        }
        // Selected deployment catalogs authorize execution. Unselected signed
        // deployments/ranks in the same package are authenticated only as
        // acquisition-source authorities for inverse hints; they are never
        // selected, authorized, loaded, linked as executable, or fetched.
        let mut catalog = schema::Catalog::new();
        for verified in &source.verified {
            for id in verified.catalog().ids() {
                let length = verified.catalog().length(id)?;
                catalog.insert(&wire::PayloadDescriptor {
                    id: id.to_vec(),
                    nbytes: length,
                })?;
            }
        }
        let mut forward: BTreeMap<PayloadId, derivation::Derivation> = BTreeMap::new();
        // An over-limit hint set is untrusted and optional: do not walk it. The
        // selected required targets are still resolved from authoritative
        // descriptors, cached bytes, or direct locations; a missing required
        // target without a usable recipe fails below.
        let hint_set_usable = source
            .metadata
            .derivations
            .as_ref()
            .is_none_or(|set| set.derivations.len() <= limits.max_derivations);
        if hint_set_usable && let Some(set) = &source.metadata.derivations {
            let selected_indices: BTreeSet<usize> = source
                .verified
                .iter()
                .filter_map(|verified| match verified.selection() {
                    protection::Selection::Deployment { deployment, .. } => Some(deployment),
                    protection::Selection::Catalog => None,
                })
                .collect();
            if let Some(publisher) = source.verified.first().map(|verified| verified.publisher().clone()) {
                for index in 0..source
                    .verified
                    .first()
                    .map(|verified| verified.deployment_count())
                    .unwrap_or(0)
                {
                    if selected_indices.contains(&index) {
                        continue;
                    }
                    let Ok(alternative) = protection::verify(
                        &source.metadata.envelope,
                        std::slice::from_ref(&publisher),
                        protection::Selection::Deployment { deployment: index, rank: None },
                    ) else {
                        continue;
                    };
                    for id in alternative.catalog().ids() {
                        let Ok(length) = alternative.catalog().length(id) else {
                            continue;
                        };
                        // A conflicting signed catalog is not needed for this
                        // acquisition; keep the selected authority intact.
                        let _ = catalog.insert(&wire::PayloadDescriptor {
                            id: id.to_vec(),
                            nbytes: length,
                        });
                    }
                }
            }
            let mut ambiguous: BTreeSet<PayloadId> = BTreeSet::new();
            for hint in &set.derivations {
                if hint.operations.len() > limits.max_operations {
                    continue;
                }
                let mut ids = BTreeSet::from([hint.target_payload_id.clone()]);
                for operation in &hint.operations {
                    if let Some(wire::derivation_operation::Operation::Copy(copy)) = &operation.operation {
                        ids.insert(copy.source_payload_id.clone());
                    }
                    if ids.len() > limits.max_operations.saturating_add(1) {
                        break;
                    }
                }
                if ids.len() > limits.max_operations.saturating_add(1) {
                    continue;
                }
                let subset = wire::DerivationSet {
                    version: set.version,
                    derivations: vec![hint.clone()],
                };
                let Ok(plans) = derivation::from_wire(&subset, &catalog, limits) else {
                    continue;
                };
                if derivation::validate_set(&plans, limits).is_err() {
                    continue;
                }
                let plan = plans.into_iter().next().unwrap();
                if let Some(previous) = forward.get(&plan.target) {
                    if previous != &plan {
                        forward.remove(&plan.target);
                        ambiguous.insert(plan.target);
                        continue;
                    }
                }
                if !ambiguous.contains(&plan.target) {
                    forward.insert(plan.target, plan);
                }
            }
        }

        // Scope authenticated candidate work to the selected required closure and
        // the inverse cone of derivations that can recover it. Unrelated cached
        // deployment payloads must not be opened or hashed.
        let mut relevant: BTreeSet<PayloadId> = required
            .values()
            .map(|(identity, _)| *identity)
            .collect();
        let mut frontier: Vec<PayloadId> = relevant.iter().copied().collect();
        loop {
            let before = relevant.len();
            // Forward dependencies of selected targets are part of the selected
            // acquisition closure.
            for plan in forward.values() {
                if relevant.contains(&plan.target) {
                    for operation in &plan.operations {
                        if let derivation::Operation::Copy(copy) = operation
                            && relevant.insert(copy.source)
                        {
                            frontier.push(copy.source);
                        }
                    }
                }
            }
            // Alternative targets that consume any relevant source can recover
            // a missing canonical dependency through the inverse planner.
            while let Some(source) = frontier.pop() {
                for plan in forward.values() {
                    if !relevant.contains(&plan.target)
                        && plan.operations.iter().any(|operation| {
                            matches!(operation, derivation::Operation::Copy(copy) if copy.source == source)
                        })
                    {
                        relevant.insert(plan.target);
                        frontier.push(plan.target);
                    }
                }
            }
            if relevant.len() == before {
                break;
            }
        }
        forward.retain(|target, _| relevant.contains(target));
        let required_keys: BTreeSet<Vec<u8>> = required.keys().cloned().collect();
        let retained_sources = if hint_set_usable {
            self.optional_derivation_sources(&source, &required_keys, &placed)
        } else {
            BTreeMap::new()
        };
        let mut candidates: BTreeSet<Vec<u8>> = relevant
            .iter()
            .map(|identity| identity.as_bytes().to_vec())
            .collect();
        candidates.extend(retained_sources.keys().cloned());
        for plan in forward.values() {
            candidates.insert(plan.target.as_bytes().to_vec());
            for operation in &plan.operations {
                if let derivation::Operation::Copy(copy) = operation {
                    candidates.insert(copy.source.as_bytes().to_vec());
                }
            }
        }
        let mut available: BTreeMap<PayloadId, PayloadObject> = BTreeMap::new();
        for id in &candidates {
            let identity = PayloadId::from_bytes(
                id.as_slice()
                    .try_into()
                    .map_err(|_| "payload identity is not 32 bytes".to_string())?,
            );
            let Ok(length) = catalog.length(id) else {
                continue;
            };
            if let Ok(Some(object)) = self.payload(identity) {
                let compared = object.length() == length;
                let valid = validate_hit(&object, identity, length, progress.as_ref()).is_ok();
                if compared && let Some(progress) = &progress {
                    progress.record_local_source_bytes(length);
                }
                if valid {
                    available.insert(identity, object);
                }
            }
        }
        let available_ids: BTreeSet<PayloadId> = available.keys().copied().collect();
        let missing_required: BTreeSet<PayloadId> = required
            .values()
            .map(|(id, _)| *id)
            .filter(|id| !available_ids.contains(id))
            .collect();
        // Inverse planning needs every missing canonical dependency of the
        // selected forward closure, not only the directly required targets.
        // A selected TP4 target may need canonical C, with C recoverable from a
        // cached TP2 alternative rather than being directly required itself.
        let mut inverse_missing = missing_required.clone();
        let mut missing_frontier: Vec<PayloadId> = inverse_missing.iter().copied().collect();
        while let Some(target) = missing_frontier.pop() {
            let Some(plan) = forward.get(&target) else {
                continue;
            };
            for operation in &plan.operations {
                if let derivation::Operation::Copy(copy) = operation
                    && !available_ids.contains(&copy.source)
                    && inverse_missing.insert(copy.source)
                {
                    missing_frontier.push(copy.source);
                }
            }
        }
        let forward_values: Vec<derivation::Derivation> = forward.values().cloned().collect();
        let inverse_plan = inverse::plan(
            &catalog,
            &forward_values,
            &inverse_missing,
            &available_ids,
            limits,
        )?;
        let mut derivations: BTreeMap<PayloadId, derivation::Derivation> = inverse_plan
            .derivations
            .into_iter()
            .map(|plan| (plan.target, plan))
            .collect();
        let mut resolved: BTreeSet<PayloadId> = available_ids.clone();
        resolved.extend(derivations.keys().copied());
        let mut remote: BTreeMap<Vec<u8>, (String, String)> = BTreeMap::new();
        let mut local: BTreeSet<Vec<u8>> = BTreeSet::new();
        let mut selected: BTreeMap<PayloadId, derivation::Derivation> = BTreeMap::new();
        let mut local_memo = BTreeMap::new();
        let required_missing: Vec<PayloadId> = required
            .values()
            .map(|(id, _)| *id)
            .filter(|id| !resolved.contains(id))
            .collect();
        // Prefer a complete locally satisfiable forward closure over fetching a
        // located target. This is what lets cached TP1 recover TP2 without a
        // network read of the TP2 target.
        for target in &required_missing {
            if resolved.contains(target) {
                continue;
            }
            if let Some(dependencies) = local_forward_closure(
                *target,
                &forward,
                &resolved,
                &mut local_memo,
                &mut BTreeSet::new(),
            ) {
                for plan in dependencies {
                    resolved.insert(plan.target);
                    selected.insert(plan.target, plan);
                }
            }
        }
        // Targets that could not be completed locally fall back to their
        // located target or a forward closure whose remaining sources are
        // fetched remotely.
        let mut stack: Vec<PayloadId> = required_missing
            .into_iter()
            .filter(|id| !resolved.contains(id))
            .collect();
        while let Some(target) = stack.pop() {
            if resolved.contains(&target) || selected.contains_key(&target) {
                continue;
            }
            if placed.contains_key(target.as_bytes().as_slice()) {
                remote.insert(
                    target.as_bytes().to_vec(),
                    placed.get(target.as_bytes().as_slice()).unwrap().clone(),
                );
                resolved.insert(target);
                continue;
            }
            let Some(plan) = forward.get(&target) else {
                return Err(format!(
                    "payload {target} is neither cached, located, nor derivable by a selected deployment"
                )
                .into());
            };
            selected.insert(target, plan.clone());
            for operation in &plan.operations {
                if let derivation::Operation::Copy(copy) = operation {
                    let source = copy.source;
                    if resolved.contains(&source) {
                        continue;
                    }
                    if let Some(dependencies) = local_forward_closure(
                        source,
                        &forward,
                        &resolved,
                        &mut local_memo,
                        &mut BTreeSet::new(),
                    ) {
                        for plan in dependencies {
                            resolved.insert(plan.target);
                            selected.insert(plan.target, plan);
                        }
                    } else if placed.contains_key(source.as_bytes().as_slice()) {
                        remote.insert(
                            source.as_bytes().to_vec(),
                            placed.get(source.as_bytes().as_slice()).unwrap().clone(),
                        );
                        resolved.insert(source);
                    } else {
                        stack.push(source);
                    }
                }
            }
        }
        // Optional sources for ready selected targets use the reviewed source
        // traversal. Cached sources stay in the available map; located sources
        // are read individually and never expand the remote fetch set.
        let mut retained_available = BTreeSet::new();
        for (id, _) in &retained_sources {
            let identity = PayloadId::from_bytes(
                id.as_slice()
                    .try_into()
                    .map_err(|_| "payload identity is not 32 bytes".to_string())?,
            );
            if available.contains_key(&identity) {
                retained_available.insert(identity);
            } else if !resolved.contains(&identity) && placed.contains_key(id.as_slice()) {
                local.insert(id.clone());
            }
        }
        for plan in selected.values() {
            derivations.insert(plan.target, plan.clone());
        }
        let mut ordered: Vec<derivation::Derivation> = derivations.into_values().collect();
        let order = derivation::validate_set(&ordered, limits)?;
        ordered = order.into_iter().map(|index| ordered[index].clone()).collect();
        for id in required.keys() {
            let identity = required.get(id).unwrap().0;
            if !resolved.contains(&identity)
                && !ordered.iter().any(|plan| plan.target == identity)
            {
                return Err(format!("ready closure lacks payload {identity}").into());
            }
        }
        Ok(AcquisitionPlan {
            source,
            required,
            catalog,
            available,
            derivations: ordered,
            remote,
            local,
            retained_available,
            progress,
        })
    }

    /// Install one resolved acquisition plan. No cached payload's source file is
    /// opened; remote and local locations are read only when installation needs
    /// their bytes, and cached objects carry their verified leases into the entry.
    pub fn install_planned(
        &self,
        plan: AcquisitionPlan<'_>,
        model_key: &str,
        entry: &str,
    ) -> Result<Entry> {
        let staging = self.stage(model_key, entry)?;
        self.install_planned_into(plan, staging)
    }

    /// Install one plan into a caller-reserved staging tree. This is the shared
    /// publication boundary for Hub naming: the reservation from `stage_named`
    /// is retained through commit, so no caller-chosen entry string can race.
    pub fn install_planned_named(
        &self,
        plan: AcquisitionPlan<'_>,
        staging: Staging,
    ) -> Result<Entry> {
        if staging.root.path() != self.path() {
            return Err("staging tree belongs to another cache".into());
        }
        self.install_planned_into(plan, staging)
    }

    fn install_planned_into(
        &self,
        plan: AcquisitionPlan<'_>,
        mut staging: Staging,
    ) -> Result<Entry> {
        let AcquisitionPlan {
            source,
            required,
            catalog,
            available,
            derivations,
            remote,
            local,
            retained_available,
            progress,
        } = plan;
        let placed: BTreeMap<Vec<u8>, (String, String)> = source
            .metadata
            .locations
            .locations
            .iter()
            .map(|location| {
                (
                    location.payload_id.clone(),
                    (location.file.clone(), location.tensor.clone()),
                )
            })
            .collect();
        let mut sources: BTreeMap<PayloadId, Slice> = BTreeMap::new();
        let mut present: BTreeMap<Vec<u8>, Slice> = BTreeMap::new();
        let mut locations = Vec::new();
        let mut used = BTreeSet::from(["model.safetensors".to_owned()]);
        let mut needed: BTreeSet<PayloadId> = required
            .values()
            .map(|(identity, _)| *identity)
            .collect();
        needed.extend(retained_available.iter().copied());
        for derivation in &derivations {
            needed.insert(derivation.target);
            for operation in &derivation.operations {
                if let derivation::Operation::Copy(copy) = operation {
                    needed.insert(copy.source);
                }
            }
        }
        for id in remote.keys().chain(local.iter()) {
            let identity = PayloadId::from_bytes(
                id.as_slice()
                    .try_into()
                    .map_err(|_| "payload identity is not 32 bytes".to_string())?,
            );
            needed.insert(identity);
        }
        let retained_ids: BTreeSet<PayloadId> = required
            .values()
            .map(|(identity, _)| *identity)
            .chain(retained_available.iter().copied())
            .collect();
        for (identity, object) in available {
            if !needed.contains(&identity) {
                continue;
            }
            let slice = object.slice().clone();
            sources.insert(identity, slice.clone());
            // Alternative deployment targets may supply derivation bytes without
            // being retained in this entry. Only the selected closure and the
            // reviewed optional source closure become entry locations.
            if !retained_ids.contains(&identity) {
                continue;
            }
            let id = identity.as_bytes().to_vec();
            let (file, tensor) = placed.get(&id).cloned().unwrap_or_else(|| {
                (
                    format!("companions/{}", identity.file_name()),
                    COMPANION_TENSOR.to_string(),
                )
            });
            if file.is_empty() && source.asset_payloads.contains(&id) {
                present.insert(id.clone(), slice.clone());
                locations.push(wire::PayloadLocation {
                    payload_id: id,
                    file,
                    tensor,
                });
                continue;
            }
            let companion = companion_name(&file, &tensor, identity, &mut used)?;
            staging.link(&object, &companion)?;
            present.insert(id.clone(), slice.clone());
            locations.push(wire::PayloadLocation {
                payload_id: id,
                file: companion,
                tensor: COMPANION_TENSOR.to_string(),
            });
        }
        let remote_ids: BTreeSet<Vec<u8>> = remote.keys().cloned().collect();
        if !remote_ids.is_empty() {
            let payloads =
                source.metadata.payloads(&catalog, remote_ids.iter().map(|id| id.as_slice()))?;
            for id in remote_ids {
                if present.contains_key(&id) {
                    continue;
                }
                let identity = PayloadId::from_bytes(
                    id.as_slice()
                        .try_into()
                        .map_err(|_| "payload identity is not 32 bytes".to_string())?,
                );
                let slice = payloads.get(&id)?.clone();
                let (file, tensor) = placed.get(&id).cloned().unwrap_or_else(|| {
                    (
                        format!("companions/{}", identity.file_name()),
                        COMPANION_TENSOR.to_string(),
                    )
                });
                let object = if remote.contains_key(&id) {
                    self.insert_payload(
                        identity,
                        slice.len(),
                        ProgressReader {
                            inner: SliceReader { slice: slice.clone(), offset: 0 },
                            progress: progress.clone(),
                        },
                    )?
                } else {
                    match self.insert_optional_payload(
                        identity,
                        slice.len(),
                        SliceReader { slice: slice.clone(), offset: 0 },
                        progress.as_ref(),
                    )? {
                        Some((object, _)) => object,
                        None => continue,
                    }
                };
                if file.is_empty() && source.asset_payloads.contains(&id) {
                    present.insert(id.clone(), slice.clone());
                    sources.insert(identity, slice.clone());
                    locations.push(wire::PayloadLocation { payload_id: id, file, tensor });
                    continue;
                }
                let companion = companion_name(&file, &tensor, identity, &mut used)?;
                staging.link(&object, &companion)?;
                present.insert(id.clone(), slice.clone());
                sources.insert(identity, slice.clone());
                locations.push(wire::PayloadLocation {
                    payload_id: id,
                    file: companion,
                    tensor: COMPANION_TENSOR.to_string(),
                });
            }
        }
        for id in local {
            if present.contains_key(&id) {
                continue;
            }
            let identity = PayloadId::from_bytes(
                id.as_slice()
                    .try_into()
                    .map_err(|_| "payload identity is not 32 bytes".to_string())?,
            );
            let Ok(payloads) = source.metadata.payloads(&catalog, [id.as_slice()]) else {
                continue;
            };
            let Ok(slice) = payloads.get(&id) else {
                continue;
            };
            let (object, _source_read) = match self.insert_optional_payload(
                identity,
                slice.len(),
                SliceReader { slice: slice.clone(), offset: 0 },
                progress.as_ref(),
            ) {
                Ok(Some(value)) => value,
                Ok(None) => continue,
                Err(error) => return Err(error),
            };
            let (file, tensor) = placed.get(&id).cloned().unwrap_or_else(|| {
                (
                    format!("companions/{}", identity.file_name()),
                    COMPANION_TENSOR.to_string(),
                )
            });
            if file.is_empty() && source.asset_payloads.contains(&id) {
                present.insert(id.clone(), slice.clone());
                sources.insert(identity, slice.clone());
                locations.push(wire::PayloadLocation { payload_id: id, file, tensor });
                continue;
            }
            let Ok(companion) = companion_name(&file, &tensor, identity, &mut used) else {
                continue;
            };
            staging.link(&object, &companion)?;
            present.insert(id.clone(), slice.clone());
            sources.insert(identity, slice.clone());
            locations.push(wire::PayloadLocation {
                payload_id: id,
                file: companion,
                tensor: COMPANION_TENSOR.to_string(),
            });
        }

        let deployment = source
            .deployments
            .first()
            .cloned()
            .unwrap_or_else(|| "canonical".into());
        let mut ordinal = 0usize;
        for derivation in derivations {
            let identity = derivation.target;
            if present.contains_key(identity.as_bytes().as_slice()) {
                continue;
            }
            let directory = staging.directory().join(format!("weights/tuned/{deployment}"));
            fs::create_dir_all(&directory).map_err(failure)?;
            let target = directory.join(format!("derived-{ordinal:03}.safetensors"));
            ordinal += 1;
            derivation::materialize_with(
                &target,
                &derivation,
                &sources,
                &derivation::Limits::default(),
                progress.as_ref(),
            )?;
            if fs::metadata(&target).map_err(failure)?.len() != derivation.length {
                return Err("materialized target has the wrong length".into());
            }
            let file = File::open(&target).map_err(failure)?;
            let object = self.insert_payload(
                identity,
                derivation.length,
                ProgressReader {
                    inner: file,
                    progress: progress.clone(),
                },
            )?;
            let companion = format!("weights/tuned/{deployment}/derived-{:03}.safetensors", ordinal - 1);
            fs::remove_file(&target).map_err(failure)?;
            staging.link(&object, &companion)?;
            present.insert(identity.as_bytes().to_vec(), object.slice().clone());
            sources.insert(identity, object.slice().clone());
            locations.push(wire::PayloadLocation {
                payload_id: identity.as_bytes().to_vec(),
                file: companion,
                tensor: COMPANION_TENSOR.to_string(),
            });
        }
        for id in required.keys() {
            if !present.contains_key(id) {
                return Err(format!("ready closure lacks payload {}", required.get(id).unwrap().0).into());
            }
        }
        write_primary(&staging, &source, locations, progress.as_ref())?;
        staging.commit()
    }

    /// Open or create the cache directory structure under `root`.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        for directory in [
            "blobs",
            "blobs/legacy",
            "models",
            "refs",
            "sources",
            "origins",
            "staging/incoming",
            "staging/wrapped",
            "staging/trees",
            "locks/models",
            "locks/entries",
            "locks/blobs",
            "locks/staging",
        ] {
            fs::create_dir_all(root.join(directory)).map_err(failure)?;
        }
        Ok(Self { root })
    }

    pub fn path(&self) -> &Path {
        &self.root
    }

    /// Published object for one payload identity, retained by its handle lock.
    pub fn payload(&self, id: PayloadId) -> Result<Option<PayloadObject>> {
        #[cfg(test)]
        PAYLOAD_OPENS.with(|count| count.set(count.get() + 1));
        let path = self.root.join("blobs").join(id.file_name());
        let guard = Lock::shared(self.lock_path("blobs", &id.to_hex()))?;
        if !path.exists() {
            return Ok(None);
        }
        let slice = Container::open(&path)?.tensor(COMPANION_TENSOR)?;
        Ok(Some(PayloadObject {
            id,
            path,
            slice,
            _guard: Arc::new(guard),
        }))
    }

    /// Legacy immutable container keyed by its own complete-file digest.
    pub fn legacy_container(&self, id: PayloadId) -> Result<Option<PayloadObject>> {
        let path = self.root.join("blobs/legacy").join(id.file_name());
        let guard = Lock::shared(self.lock_path("blobs", &id.to_hex()))?;
        if !path.exists() {
            return Ok(None);
        }
        let file = Arc::new(File::open(&path).map_err(failure)?);
        let length = file.metadata().map_err(failure)?.len();
        Ok(Some(PayloadObject {
            id,
            path,
            slice: Slice::new(file, 0, length)?,
            _guard: Arc::new(guard),
        }))
    }

    /// Ensure one payload object exists under a declared identity.
    ///
    /// A verified hit checks the installed payload digest and stream-checks the
    /// supplied source without writing. A miss writes the payload once and then
    /// its deterministic wrapper, hashing both as they are written.
    pub fn insert_payload(
        &self,
        declared: PayloadId,
        length: u64,
        source: impl Read,
    ) -> Result<PayloadObject> {
        if let Some(object) = self.payload(declared)? {
            validate_hit(&object, declared, length, None)?;
            verify_stream(declared, length, source)?;
            return Ok(object);
        }
        let (transaction, staged, actual) = self.stage_payload(length, source)?;
        if actual != declared {
            let _ = fs::remove_file(&staged);
            return Err(format!(
                "payload digest {actual} does not match its declared identity {declared}"
            )
            .into());
        }
        self.publish_payload(&transaction, &staged, declared, length)
    }

    /// Stream one payload whose identity is computed from its bytes.
    pub fn capture_payload(&self, length: u64, source: impl Read) -> Result<PayloadObject> {
        let (transaction, staged, id) = self.stage_payload(length, source)?;
        if let Some(object) = self.payload(id)? {
            let _ = fs::remove_file(&staged);
            validate_hit(&object, id, length, None)?;
            return Ok(object);
        }
        self.publish_payload(&transaction, &staged, id, length)
    }

    /// Store a legacy immutable container verbatim under its container digest.
    pub fn insert_legacy_container(
        &self,
        declared: PayloadId,
        source: impl Read,
    ) -> Result<PayloadObject> {
        let transaction = Transaction::begin(self)?;
        let staged = transaction.staging_file("incoming")?;
        let (actual, written) = match write_stream(&staged, source) {
            Ok(result) => result,
            Err(error) => {
                let _ = fs::remove_file(&staged);
                return Err(error);
            }
        };
        if written == 0 || actual != declared {
            let _ = fs::remove_file(&staged);
            return Err(format!(
                "legacy container does not match its declared identity {declared}"
            )
            .into());
        }
        let _publication = Lock::shared(self.publication_lock())?;
        let destination = self.root.join("blobs/legacy").join(declared.file_name());
        self.install_legacy(&staged, &destination, declared)?;
        syncdir(&self.root.join("blobs/legacy"))?;
        let guard = Lock::shared(self.lock_path("blobs", &declared.to_hex()))?;
        let file = Arc::new(File::open(&destination).map_err(failure)?);
        drop(transaction);
        Ok(PayloadObject {
            id: declared,
            path: destination,
            slice: Slice::new(file, 0, written)?,
            _guard: Arc::new(guard),
        })
    }

    /// Import a legacy package (primary plus companions) as a managed cache entry.
    ///
    /// Every file is stored verbatim under the container-digest namespace and
    /// hardlinked into the entry at its own relative name, so the legacy loader
    /// reads the same layout from the cache. The returned entry is leased.
    /// Import a legacy container tree without changing the model's selected entry.
    pub fn import_legacy_staged(
        &self,
        source: &Path,
        model_key: &str,
        entry: &str,
    ) -> Result<Entry> {
        let staging = self.stage(model_key, entry)?;
        self.import_legacy_into(source, staging)
    }

    /// Import a legacy container tree into a caller-reserved staging tree.
    /// Reserved names and model identity come from `stage_named`; the default
    /// ref is never published here.
    pub fn import_legacy_named(&self, source: &Path, staging: Staging) -> Result<Entry> {
        if staging.root.path() != self.path() {
            return Err("staging tree belongs to another cache".into());
        }
        self.import_legacy_into(source, staging)
    }

    fn import_legacy_into(&self, source: &Path, mut staging: Staging) -> Result<Entry> {
        let directory = source.parent().unwrap_or(Path::new("."));
        let primary_name = source
            .file_name()
            .ok_or_else(|| "legacy package has no file name".to_string())?
            .to_string_lossy()
            .into_owned();
        let mut files = vec![primary_name.clone()];
        files.extend(crate::runtime_package::package_companions(source)?);
        files.sort();
        files.dedup();
        for name in &files {
            let path = directory.join(name);
            // Hash the file by streaming, then store it from a pinned handle so a
            // companion far larger than memory is never buffered.
            let digest = hash_file(&path)?;
            let file = File::open(&path).map_err(failure)?;
            let object = self.insert_legacy_container(digest, file)?;
            // The primary is stored under the canonical name; signed bytes are unchanged and
            // companions keep their digest basenames.
            let placed = if name == &primary_name {
                "model.safetensors"
            } else {
                name.as_str()
            };
            staging.link(&object, placed)?;
        }
        staging.commit()
    }

    /// Import and select one legacy container tree.
    pub fn import_legacy(&self, source: &Path, model_key: &str, entry: &str) -> Result<Entry> {
        let entry = self.import_legacy_staged(source, model_key, entry)?;
        self.publish_entry(&entry)?;
        Ok(entry)
    }

    /// Store an existing payload slice, verifying the declared identity when given.
    ///
    /// Hashing and copying both stream through the pinned slice, so memory stays
    /// bounded for model-sized payloads. Installation reuses the publication
    /// transaction, which holds the publication lock that collection respects.
    pub fn store_slice(&self, declared: Option<PayloadId>, slice: &Slice) -> Result<PayloadObject> {
        let id = slice.digest()?;
        if let Some(expected) = declared
            && expected != id
        {
            return Err(format!(
                "payload digest {id} does not match its declared identity {expected}"
            )
            .into());
        }
        if let Some(object) = self.payload(id)? {
            if object.length() != slice.len() {
                return Err(format!("cached object {id} has a different length").into());
            }
            verify_slice(object.slice(), id, None)?;
            return Ok(object);
        }
        let transaction = Transaction::begin(self)?;
        let staged = transaction.staging_file("wrapped")?;
        let result = (|| {
            let mut output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&staged)
                .map_err(failure)?;
            write_companion(&mut output, slice)?;
            output.sync_all().map_err(failure)?;
            drop(output);
            self.publish_payload(&transaction, &staged, id, slice.len())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&staged);
        }
        result
    }

    /// Insert an optional retention source while distinguishing a corrupt
    /// source from publication failure. A missing/corrupt source yields
    /// `Ok(None)`; object-store or staging I/O errors are returned.
    fn insert_optional_payload(
        &self,
        declared: PayloadId,
        length: u64,
        source: impl Read,
        progress: Option<&progress::AcquisitionProgress>,
    ) -> Result<Option<(PayloadObject, bool)>> {
        if fs::symlink_metadata(self.root.join("blobs").join(declared.file_name()))
            .map(|metadata| metadata.is_dir())
            .unwrap_or(false)
        {
            return Err(format!(
                "payload object {} collides with a directory",
                declared
            )
            .into());
        }
        match self.payload(declared) {
            Ok(Some(object)) => {
                let valid = validate_hit(&object, declared, length, progress).is_ok();
                if valid {
                    return Ok(Some((object, false)));
                }
                return Ok(None);
            }
            Ok(None) => {}
            Err(_) => return Ok(None),
        }
        struct OptionalSource<R> {
            inner: ProgressReader<R>,
            length: u64,
            read: u64,
            failed: std::rc::Rc<std::cell::Cell<bool>>,
        }
        impl<R: Read> Read for OptionalSource<R> {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                match self.inner.read(buffer) {
                    Ok(count) => {
                        self.read = self.read.saturating_add(count as u64);
                        if self.read > self.length {
                            self.failed.set(true);
                        }
                        Ok(count)
                    }
                    Err(error) => {
                        self.failed.set(true);
                        Err(error)
                    }
                }
            }
        }
        let failed = std::rc::Rc::new(std::cell::Cell::new(false));
        let source = OptionalSource {
            inner: ProgressReader {
                inner: source,
                progress: progress.cloned(),
            },
            length,
            read: 0,
            failed: std::rc::Rc::clone(&failed),
        };
        let (transaction, staged, actual) = match self.stage_payload(length, source) {
            Ok(value) => value,
            Err(_) if failed.get() => return Ok(None),
            Err(error) => return Err(error),
        };
        if actual != declared {
            let _ = fs::remove_file(&staged);
            return Ok(None);
        }
        self.publish_payload(&transaction, &staged, declared, length)
            .map(|object| Some((object, true)))
    }

    /// Reserve a staging tree under a collision-free entry id derived from `candidate`.
    ///
    /// The naming policy belongs to the caller (for example `gb202-tp1-<UTC>` or
    /// `gb202-tp2-<UTC>` for one entry holding several deployments). Reservations are decided
    /// under the model coordination lock and keep the entry's exclusive lock until the returned
    /// staging tree is committed or dropped, so a published name and a live reservation are both
    /// excluded without retrying any payload work. Collisions use `-2`, `-3`, ... suffixes.
    pub fn stage_named(&self, model: &str, candidate: &str) -> Result<Staging> {
        validate_name(model, "model key")?;
        validate_name(candidate, "entry id")?;
        let _publication = Lock::shared(self.publication_lock())?;
        let _coordination = Lock::exclusive(self.lock_path("models", model))?;
        for suffix in 1..=9999u32 {
            let id = if suffix == 1 {
                candidate.to_owned()
            } else {
                format!("{candidate}-{suffix}")
            };
            validate_name(&id, "entry id")?;
            if fs::symlink_metadata(self.root.join("models").join(model).join(&id)).is_ok() {
                continue;
            }
            let Some(reservation) = Lock::try_exclusive(self.entry_lock(model, &id))? else {
                continue;
            };
            let mut staging = self.stage(model, &id)?;
            staging.reservation = Some(reservation);
            return Ok(staging);
        }
        Err("could not reserve a cache entry name".into())
    }

    /// Start one staging transaction for a published entry.
    pub fn stage(&self, model: &str, entry: &str) -> Result<Staging> {
        validate_name(model, "model key")?;
        validate_name(entry, "entry id")?;
        let transaction = Transaction::begin(self)?;
        let directory = transaction.staging_file("trees")?;
        fs::create_dir_all(&directory).map_err(failure)?;
        Ok(Staging {
            root: self.clone(),
            _transaction: transaction,
            model: model.to_owned(),
            entry: entry.to_owned(),
            directory,
            copied: false,
            published: false,
            reservation: None,
        })
    }

    /// Merge payload identities authenticated by the selected deployments only.
    fn merged_catalog(
        verified: &[protection::VerifiedPackage],
        ids: impl IntoIterator<Item = Vec<u8>>,
    ) -> Result<crate::package::schema::Catalog> {
        let mut catalog = crate::package::schema::Catalog::new();
        let mut seen: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
        for id in ids {
            let mut length = None;
            for package in verified {
                if let Ok(known) = package.catalog().length(&id) {
                    match length {
                        Some(existing) if existing != known => {
                            return Err(
                                "selected deployments disagree about a payload length".into()
                            );
                        }
                        _ => length = Some(known),
                    }
                }
            }
            let Some(length) = length else {
                return Err(format!(
                    "payload {} is not authorized by any selected deployment",
                    PayloadId::from_bytes(
                        id.as_slice()
                            .try_into()
                            .map_err(|_| "payload identity is not 32 bytes".to_string())?
                    )
                )
                .into());
            };
            if seen.insert(id.clone(), length).is_none() {
                catalog.insert(&wire::PayloadDescriptor { id, nbytes: length })?;
            }
        }
        Ok(catalog)
    }

    /// Validate a source-complete closure from signed descriptors alone.
    ///
    /// Every required payload must be located or derivable; derivation hints are
    /// parsed against the merged selected catalogue (rejecting unsupported
    /// operations, bad lengths, cycles, and depth beyond the limit), and every
    /// dependency is followed recursively until it reaches a located payload.
    pub fn validate_ready_closure(
        metadata: &reader::Metadata,
        verified: &[protection::VerifiedPackage],
        limits: &derivation::Limits,
    ) -> Result<()> {
        let located = located_payload_ids(metadata);
        let mut required: BTreeSet<Vec<u8>> = BTreeSet::new();
        for package in verified {
            for id in reader::required_payloads(package)? {
                required.insert(id);
            }
        }
        let missing: BTreeSet<Vec<u8>> = required
            .into_iter()
            .filter(|id| !located.contains(id))
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        let set = metadata.derivations.as_ref().ok_or_else(|| {
            invalid("payload is neither located nor derivable by a selected deployment")
        })?;
        let selected = selected_derivation_set(set, &missing, &located, limits)?;
        let catalog = Self::merged_catalog(verified, selected_derivation_ids(&selected))?;
        let plans = derivation::from_wire(&selected, &catalog, limits)?;
        derivation::order(&plans, limits)?;
        Ok(())
    }

    /// Verify that an existing entry holds the requested ready closure.
    ///
    /// Every payload the selected deployments require must be present at its
    /// recorded companion path (or its derived path) with the signed length and
    /// digest; existence alone never satisfies a warm hit.
    pub fn verify_ready(&self, entry: &Entry, source: &PackageSource<'_>) -> Result<()> {
        self.verify_ready_with(entry, source, None)
    }

    /// Verify an existing entry, recording each hashed payload as verification
    /// progress for the acquisition that requested it.
    pub fn verify_ready_with(
        &self,
        entry: &Entry,
        source: &PackageSource<'_>,
        progress: Option<&progress::AcquisitionProgress>,
    ) -> Result<()> {
        // The entry must be the same signed package. Its physical locations may
        // be normalized, but its signed envelope may not change.
        let installed = reader::Metadata::open(entry.directory())?;
        if installed.envelope != source.metadata.envelope {
            return Err("cached entry signed envelope differs from its source".into());
        }
        let mut required: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
        for verified in &source.verified {
            for id in reader::required_payloads(verified)? {
                let length = verified.catalog().length(&id)?;
                match required.entry(id.clone()) {
                    std::collections::btree_map::Entry::Vacant(slot) => {
                        slot.insert(length);
                    }
                    std::collections::btree_map::Entry::Occupied(slot)
                        if *slot.get() != length =>
                    {
                        return Err("selected deployments disagree about a payload length".into());
                    }
                    std::collections::btree_map::Entry::Occupied(_) => {}
                }
            }
        }
        // Resolve the installed locations and their exact tensor names. Cache
        // entries may normalize paths, but the entry's location tensor is the
        // binding the loader will follow.
        let mut catalog = schema::Catalog::new();
        for (id, length) in &required {
            catalog.insert(&wire::PayloadDescriptor {
                id: id.clone(),
                nbytes: *length,
            })?;
        }
        let files = installed.payloads(
            &catalog,
            required.keys().map(|id| id.as_slice()),
        )?;
        for (id, length) in required {
            let identity = PayloadId::from_bytes(
                id.as_slice()
                    .try_into()
                    .map_err(|_| "payload identity is not 32 bytes".to_string())?,
            );
            let slice = files.get(id.as_slice())?;
            if slice.len() != length {
                return Err(format!(
                    "cached entry payload {identity} has {} bytes, expected {length}",
                    slice.len()
                )
                .into());
            }
            verify_slice(&slice, identity, progress)?;
        }
        Ok(())
    }

    /// Select an already leased entry; callers retain the lease through loading.
    pub fn publish_entry(&self, entry: &Entry) -> Result<()> {
        self.publish(&entry.model, &entry.id)
    }

    /// Atomically select one published entry for a model.
    pub fn publish(&self, model: &str, entry: &str) -> Result<()> {
        validate_name(model, "model key")?;
        validate_name(entry, "entry id")?;
        let _publication = Lock::shared(self.publication_lock())?;
        let _coordination = Lock::shared(self.lock_path("models", model))?;
        if !self.root.join("models").join(model).join(entry).is_dir() {
            return Err(format!("model entry {model}/{entry} is not published").into());
        }
        self.write_ref(model, entry)
    }

    /// Resolve the model's selected entry, leasing it before the ref is released.
    pub fn resolve(&self, model: &str) -> Result<Option<Entry>> {
        validate_name(model, "model key")?;
        let coordination = Lock::shared(self.lock_path("models", model))?;
        let Some(entry) = self.ref_entry(model)? else {
            return Ok(None);
        };
        let entry = self.entry(model, &entry)?;
        drop(coordination);
        Ok(Some(entry))
    }

    /// Lease one published entry without consulting the ref.
    pub fn entry(&self, model: &str, entry: &str) -> Result<Entry> {
        validate_name(model, "model key")?;
        validate_name(entry, "entry id")?;
        let lease = Lock::shared(self.entry_lock(model, entry))?;
        let directory = self.root.join("models").join(model).join(entry);
        if !directory.is_dir() {
            return Err(format!("model entry {model}/{entry} is not published").into());
        }
        Ok(Entry {
            model: model.to_owned(),
            id: entry.to_owned(),
            directory,
            _lease: lease,
        })
    }

    pub fn ref_entry(&self, model: &str) -> Result<Option<String>> {
        validate_name(model, "model key")?;
        match fs::read_to_string(self.root.join("refs").join(model)) {
            Ok(text) => {
                let entry = text.trim_end_matches('\n').to_owned();
                validate_name(&entry, "entry id")?;
                Ok(Some(entry))
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(failure(error)),
        }
    }

    /// Remember an acquisition selector after its entry is ready. Source refs
    /// are lookup hints, not owners: eviction may leave a ref to an absent entry.
    pub fn publish_source_ref(
        &self,
        selector: &str,
        revision: &str,
        entry: &Entry,
        expected_envelope: [u8; 32],
    ) -> Result<()> {
        let directory = self.root.join("sources");
        let destination = directory.join(PayloadId::of(selector.as_bytes()).to_string());
        self.write_source_ref(&destination, selector, revision, entry, expected_envelope)
    }

    /// Retain the acquisition origin of an immutable version independently of moving selectors.
    pub fn publish_entry_source(
        &self,
        selector: &str,
        revision: &str,
        entry: &Entry,
        expected_envelope: [u8; 32],
    ) -> Result<()> {
        self.write_source_ref(
            &self.origin_path(entry.model(), entry.id()),
            selector,
            revision,
            entry,
            expected_envelope,
        )
    }

    fn origin_path(&self, model: &str, entry: &str) -> PathBuf {
        self.root
            .join("origins")
            .join(PayloadId::of(format!("{model}/{entry}").as_bytes()).to_string())
    }

    fn write_source_ref(
        &self,
        destination: &Path,
        selector: &str,
        revision: &str,
        entry: &Entry,
        expected_envelope: [u8; 32],
    ) -> Result<()> {
        let local = self.entry(entry.model(), entry.id())?;
        if local.directory() != entry.directory() {
            return Err("source reference entry belongs to another cache".into());
        }
        let directory = destination
            .parent()
            .ok_or("source reference has no parent")?;
        let temporary = directory.join(format!(".ref-{}", token()));
        let result = (|| {
            let bytes = serde_json::to_vec(&SourceRefRecord::Versioned {
                version: 2,
                selector: selector.to_owned(),
                revision: revision.to_owned(),
                model: entry.model().to_owned(),
                entry: entry.id().to_owned(),
                envelope_sha256: PayloadId::from_bytes(expected_envelope).to_hex(),
            })
            .map_err(|error| error.to_string())?;
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .map_err(failure)?;
            io::Write::write_all(&mut file, &bytes).map_err(failure)?;
            file.sync_all().map_err(failure)?;
            drop(file);
            fs::rename(&temporary, destination).map_err(failure)?;
            syncdir(&directory)
        })();
        if result.is_err() {
            let _ = fs::remove_file(temporary);
        }
        result
    }

    /// Read the remembered revision and entry for a selector without leasing or requiring the
    /// entry to still exist. Offline acquisition needs the resolved commit even after eviction.
    pub fn read_source_ref(&self, selector: &str) -> Result<Option<SourceRef>> {
        let path = self
            .root
            .join("sources")
            .join(PayloadId::of(selector.as_bytes()).to_string());
        let reference = Self::read_source_file(&path)?;
        if reference
            .as_ref()
            .is_some_and(|reference| reference.selector() != selector)
        {
            return Err("cache source reference selector does not match".into());
        }
        Ok(reference)
    }

    pub fn entry_source(&self, entry: &Entry) -> Result<Option<SourceRef>> {
        let reference = Self::read_source_file(&self.origin_path(entry.model(), entry.id()))?;
        if reference.as_ref().is_some_and(|reference| {
            reference.model() != entry.model() || reference.entry() != entry.id()
        }) {
            return Err("cache origin names a different model version".into());
        }
        Ok(reference)
    }

    /// Acquisition references retained by caches without a per-version origin record.
    pub fn entry_source_refs(&self, entry: &Entry) -> Result<Vec<SourceRef>> {
        let mut references = Vec::new();
        for path in read_dir(&self.root.join("sources"))? {
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if PayloadId::parse(name).is_err() {
                continue;
            }
            if let Some(reference) = Self::read_source_file(&path)? {
                if PayloadId::of(reference.selector().as_bytes()).to_string() != name {
                    return Err("cache source reference selector does not match".into());
                }
                if reference.model() == entry.model() && reference.entry() == entry.id() {
                    references.push(reference);
                }
            }
        }
        Ok(references)
    }

    fn read_source_file(path: &Path) -> Result<Option<SourceRef>> {
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(failure(error)),
        };
        let record: SourceRefRecord =
            serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        let (stored, revision, model, id, expected_envelope) = match record {
            SourceRefRecord::Legacy((stored, revision, model, id)) => {
                (stored, revision, model, id, None)
            }
            SourceRefRecord::Versioned {
                version,
                selector: stored,
                revision,
                model,
                entry: id,
                envelope_sha256,
            } => {
                if version != 2 {
                    return Err(
                        format!("unsupported cache source reference version {version}").into(),
                    );
                }
                let digest = PayloadId::parse(&envelope_sha256)?;
                (stored, revision, model, id, Some(*digest.as_bytes()))
            }
        };
        validate_name(&model, "model key")?;
        validate_name(&id, "entry id")?;
        Ok(Some(SourceRef {
            selector: stored,
            revision,
            model,
            entry: id,
            expected_envelope,
        }))
    }

    /// Resolve and lease a previously acquired entry without consulting its source.
    ///
    /// The returned `SourceRef` is the exact record used to lease the entry, so callers that
    /// validate an expected package identity do not reread a separately updated selector.
    pub fn resolve_source_ref_record(&self, selector: &str) -> Result<Option<(SourceRef, Entry)>> {
        let Some(reference) = self.read_source_ref(selector)? else {
            return Ok(None);
        };
        let _coordination = Lock::shared(self.lock_path("models", &reference.model))?;
        if !self
            .root
            .join("models")
            .join(&reference.model)
            .join(&reference.entry)
            .try_exists()
            .map_err(failure)?
        {
            return Ok(None);
        }
        self.entry(&reference.model, &reference.entry)
            .map(|entry| Some((reference, entry)))
    }

    /// Resolve the revision and lease of a previously acquired entry.
    pub fn resolve_source_ref(&self, selector: &str) -> Result<Option<(String, Entry)>> {
        self.resolve_source_ref_record(selector).map(|selected| {
            selected.map(|(reference, entry)| (reference.revision().to_owned(), entry))
        })
    }

    /// Read an entry's manifest summary without authenticating it. Listing reports these facts as
    /// unverified; `cache inspect` runs the shared verified inspection path instead.
    pub fn summarize_entry(&self, entry: &Entry) -> Result<EntrySummary> {
        match crate::package::reader::Metadata::open(entry.directory()) {
            Ok(metadata) => self.payload_summary(entry, &metadata),
            Err(_) => legacy_summary(self, entry),
        }
    }

    fn payload_summary(
        &self,
        entry: &Entry,
        metadata: &crate::package::reader::Metadata,
    ) -> Result<EntrySummary> {
        let envelope = crate::package::protection::envelope(&metadata.envelope)?;
        let package =
            crate::package::schema::wire::PackageBody::decode(envelope.package_body.as_slice())
                .map_err(|error| {
                    crate::error::Error::from(format!("invalid package body: {error}"))
                })?;
        // Only payloads this entry actually locates count toward its logical bytes. Cataloged
        // descriptors without a location belong to other closures.
        let located: BTreeSet<&[u8]> = metadata
            .locations
            .locations
            .iter()
            .filter(|location| {
                if location.file.is_empty() {
                    metadata.primary().tensor(&location.tensor).is_ok()
                } else {
                    fs::symlink_metadata(entry.directory().join(&location.file))
                        .is_ok_and(|metadata| metadata.is_file())
                }
            })
            .map(|location| location.payload_id.as_slice())
            .collect();
        let mut lengths: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
        let mut add = |id: &[u8], nbytes: u64| {
            if located.contains(id) {
                lengths.entry(id.to_vec()).or_insert(nbytes);
            }
        };
        for payload in &package.payloads {
            add(&payload.id, payload.nbytes);
        }
        let ready_ids: BTreeSet<&str> = metadata
            .locations
            .available_deployments
            .iter()
            .map(String::as_str)
            .collect();
        let mut deployments = Vec::new();
        for signed in &envelope.deployments {
            let body = crate::package::schema::wire::DeploymentBody::decode(signed.body.as_slice())
                .map_err(|error| {
                    crate::error::Error::from(format!("invalid deployment body: {error}"))
                })?;
            for payload in &body.payloads {
                add(&payload.id, payload.nbytes);
            }
            for rank in &envelope.ranks {
                if rank.deployment == signed.index {
                    if let Ok(rank_body) =
                        crate::package::schema::wire::RankBody::decode(rank.body.as_slice())
                    {
                        // Rank payloads carry exported weights, graphs, and state.
                        for payload in &rank_body.payloads {
                            add(&payload.id, payload.nbytes);
                        }
                    }
                }
            }
            let contract = body.contract;
            let deployment_id = body.deployment_id.clone();
            let ready = ready_ids.is_empty() || ready_ids.contains(deployment_id.as_str());
            deployments.push(DeploymentSummary {
                id: deployment_id,
                tensor_parallel: body.tensor_parallel,
                ready,
                pages: contract.as_ref().map(|contract| contract.pages),
                state_rows: contract.as_ref().map(|contract| contract.state_rows),
                max_sequence_tokens: contract
                    .as_ref()
                    .map(|contract| contract.max_sequence_tokens),
                prefill_chunk_size: contract
                    .as_ref()
                    .map(|contract| contract.prefill_chunk_size),
            });
        }
        deployments.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(EntrySummary {
            form: "payload",
            format_version: package.format_version,
            model_identity: package.model_identity,
            hardware_identity: package.hardware_identity,
            deployments,
            logical_bytes: lengths.values().copied().sum(),
            payloads: lengths.len(),
        })
    }

    /// The package primary of one entry: `model.safetensors` when present, otherwise the unique
    /// top-level container whose envelope parses as a signed package. Ambiguity is an error.
    pub fn primary(&self, entry: &Entry) -> Result<PathBuf> {
        let preferred = entry.directory().join("model.safetensors");
        if preferred.is_file() {
            return Ok(preferred);
        }
        let mut found: Option<PathBuf> = None;
        for candidate in read_dir(entry.directory())? {
            if candidate.extension().and_then(|value| value.to_str()) != Some("safetensors")
                || !candidate.is_file()
            {
                continue;
            }
            if crate::runtime_package::PackageClaim::read(&candidate).is_err() {
                continue;
            }
            if found.is_some() {
                return Err(format!(
                    "cache entry {}/{} has several package-like primaries",
                    entry.model(),
                    entry.id()
                )
                .into());
            }
            found = Some(candidate);
        }
        found.ok_or_else(|| {
            format!(
                "cache entry {}/{} has no primary",
                entry.model(),
                entry.id()
            )
            .into()
        })
    }

    /// Lease the published entry that contains `path`, when it lies inside this cache.
    ///
    /// Containment is checked on canonical paths and on validated name components, so a symlink
    /// or `..` component cannot escape `models/<model>/<entry>`. The entry lock is acquired before
    /// any caller reads through the path.
    pub fn entry_for_path(&self, path: &Path) -> Result<Option<Entry>> {
        let root = self.root.canonicalize().map_err(failure)?;
        let candidate = match path.canonicalize() {
            Ok(candidate) => candidate,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(failure(error)),
        };
        let Ok(relative) = candidate.strip_prefix(&root) else {
            return Ok(None);
        };
        let components: Vec<&std::ffi::OsStr> = relative.iter().collect();
        if components.first().and_then(|value| value.to_str()) != Some("models")
            || components.len() < 3
        {
            return Ok(None);
        }
        let model = components[1]
            .to_str()
            .ok_or("cache path model is not UTF-8")?;
        let entry = components[2]
            .to_str()
            .ok_or("cache path entry is not UTF-8")?;
        validate_name(model, "model key")?;
        validate_name(entry, "entry id")?;
        let directory = self.root.join("models").join(model).join(entry);
        // Lock first, then re-check under the lock: a concurrent removal either wins before the
        // lock (we report None) or is blocked by the shared lease we return.
        let lease = Lock::shared(self.entry_lock(model, entry))?;
        if !directory.is_dir() {
            return Ok(None);
        }
        Ok(Some(Entry {
            model: model.to_owned(),
            id: entry.to_owned(),
            directory,
            _lease: lease,
        }))
    }

    /// Whether another process or handle currently leases one entry. Reporting only; the
    /// exclusive attempt is released immediately.
    pub fn is_leased(&self, model: &str, entry: &str) -> Result<bool> {
        validate_name(model, "model key")?;
        validate_name(entry, "entry id")?;
        Ok(Lock::try_exclusive(self.entry_lock(model, entry))?.is_none())
    }

    /// Model keys this cache holds entries for, in sorted order.
    pub fn models(&self) -> Result<Vec<String>> {
        let mut models = Vec::new();
        for path in read_dir(&self.root.join("models"))? {
            if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
                validate_name(name, "model key")?;
                models.push(name.to_owned());
            }
        }
        models.sort();
        Ok(models)
    }

    /// Entry ids published for one model, in sorted order.
    pub fn versions(&self, model: &str) -> Result<Vec<String>> {
        validate_name(model, "model key")?;
        let directory = self.root.join("models").join(model);
        let entries = match fs::read_dir(directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(failure(error)),
        };
        let mut versions = Vec::new();
        for entry in entries {
            let path = entry.map_err(failure)?.path();
            if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
                validate_name(name, "entry id")?;
                versions.push(name.to_owned());
            }
        }
        versions.sort();
        Ok(versions)
    }

    /// Clear a model's selected entry.
    pub fn clear_default(&self, model: &str) -> Result<()> {
        validate_name(model, "model key")?;
        let _publication = Lock::shared(self.publication_lock())?;
        let _coordination = Lock::exclusive(self.lock_path("models", model))?;
        match fs::remove_file(self.root.join("refs").join(model)) {
            Ok(()) => syncdir(&self.root.join("refs")),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(failure(error)),
        }
    }

    /// Remove one entry, refusing a lease holder and refusing to change the default unless the
    /// caller asked to clear it.
    pub fn remove(&self, model: &str, entry: &str, clear_default: bool) -> Result<()> {
        validate_name(model, "model key")?;
        validate_name(entry, "entry id")?;
        let _publication = Lock::shared(self.publication_lock())?;
        let _coordination = Lock::exclusive(self.lock_path("models", model))?;
        let directory = self.root.join("models").join(model).join(entry);
        if !directory.is_dir() {
            return Err(format!("model entry {model}/{entry} is not published").into());
        }
        let selected = self.ref_entry(model)?.as_deref() == Some(entry);
        if selected && !clear_default {
            return Err(format!(
                "model entry {model}/{entry} is the selected default; select another entry or pass --clear-default"
            )
            .into());
        }
        let _lease = Lock::try_exclusive(self.entry_lock(model, entry))?
            .ok_or_else(|| format!("model entry {model}/{entry} is leased"))?;
        // Clear the default before deleting the entry: a failure must leave the
        // entry intact rather than a ref naming a directory that no longer exists.
        if selected {
            match fs::remove_file(self.root.join("refs").join(model)) {
                Ok(()) => syncdir(&self.root.join("refs"))?,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(failure(error)),
            }
        }
        remove_tree(&directory)?;
        self.remove_origin(model, entry)?;
        syncdir(&self.root.join("models").join(model))?;
        Ok(())
    }

    fn remove_origin(&self, model: &str, entry: &str) -> Result<()> {
        match fs::remove_file(self.origin_path(model, entry)) {
            Ok(()) => syncdir(&self.root.join("origins")),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(failure(error)),
        }
    }

    /// Export a committed entry to a standalone directory, hardlinking objects
    /// where the filesystem allows and copying explicitly otherwise.
    ///
    /// The tree is assembled in a hidden sibling and renamed into place, so a
    /// failure never exposes a partial export or replaces existing output. The
    /// caller keeps the entry lease for the duration.
    pub fn export_entry(&self, entry: &Entry, destination: &Path) -> Result<()> {
        if fs::symlink_metadata(destination).is_ok() {
            return Err(format!(
                "export destination {} already exists",
                destination.display()
            )
            .into());
        }
        let parent = parent_of(destination);
        let name = destination
            .file_name()
            .ok_or_else(|| "export destination has no file name".to_string())?
            .to_string_lossy()
            .into_owned();
        fs::create_dir_all(parent).map_err(failure)?;
        let staging = parent.join(format!(".{name}.flare-{}", token()));
        let result = (|| {
            export_tree(&entry.directory, &staging)?;
            synchronize(&staging)?;
            seal(&staging)?;
            rename_noreplace(&staging, destination).map_err(failure)?;
            syncdir(parent)
        })();
        if result.is_err() {
            let _ = remove_tree(&staging);
        }
        result
    }

    /// Remove one published entry. Fails while any lease is held.
    pub fn evict(&self, model: &str, entry: &str) -> Result<()> {
        validate_name(model, "model key")?;
        validate_name(entry, "entry id")?;
        let _publication = Lock::shared(self.publication_lock())?;
        let _coordination = Lock::exclusive(self.lock_path("models", model))?;
        let _lease = Lock::try_exclusive(self.entry_lock(model, entry))?
            .ok_or_else(|| format!("model entry {model}/{entry} is leased"))?;
        let directory = self.root.join("models").join(model).join(entry);
        if directory.is_dir() {
            remove_tree(&directory)?;
            syncdir(&self.root.join("models").join(model))?;
        }
        self.remove_origin(model, entry)?;
        if self.ref_entry(model)?.as_deref() == Some(entry) {
            match fs::remove_file(self.root.join("refs").join(model)) {
                Ok(()) => syncdir(&self.root.join("refs"))?,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(failure(error)),
            }
        }
        Ok(())
    }

    /// Reclaim abandoned staging work and objects no published entry or live
    /// staging tree references, under exclusive publication coordination.
    pub fn collect(&self) -> Result<Collected> {
        let _publication = Lock::exclusive(self.publication_lock())?;
        let mut collected = Collected::default();
        let mut live = HashSet::new();
        for model in read_dir(&self.root.join("models"))? {
            for entry in read_dir(&model)? {
                collect_inodes(&entry, &mut live)?;
            }
        }
        for tree in read_dir(&self.root.join("staging/trees"))? {
            // A staging transaction that carries a Hugging Face transport cache is a
            // resumable download partial. It is preserved here and reclaimed only by
            // `flare models gc --downloads`, which also checks the transport locks.
            if is_legacy_hub_transport(&tree) {
                collect_inodes(&tree, &mut live)?;
                collected.preserved_transport += 1;
                continue;
            }
            let Some(token) = tree.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let Some(_lease) = Lock::try_exclusive(self.lock_path("staging", token))? else {
                collect_inodes(&tree, &mut live)?;
                continue;
            };
            remove_tree(&tree)?;
            collected.staging += 1;
        }
        for (directory, legacy) in [("blobs", false), ("blobs/legacy", true)] {
            let path = self.root.join(directory);
            for item in read_dir(&path)? {
                let Some(name) = item.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                let Ok(id) = PayloadId::parse(name.trim_end_matches(".safetensors")) else {
                    continue;
                };
                let Some(_holding) = Lock::try_exclusive(self.lock_path("blobs", &id.to_hex()))?
                else {
                    continue;
                };
                let identity = inode(&item)
                    .ok_or_else(|| format!("cannot identify cached object {}", item.display()))?;
                if live.contains(&identity) {
                    continue;
                }
                fs::remove_file(&item).map_err(failure)?;
                collected.objects += 1;
                collected.legacy += usize::from(legacy);
            }
            syncdir(&path)?;
        }
        for directory in ["staging/incoming", "staging/wrapped"] {
            for item in read_dir(&self.root.join(directory))? {
                let Some(token) = item.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                let Some(_lease) = Lock::try_exclusive(self.lock_path("staging", token))? else {
                    continue;
                };
                if item.is_file() {
                    fs::remove_file(&item).map_err(failure)?;
                }
            }
        }
        Ok(collected)
    }

    /// Write one companion wrapper, hashing the payload as it streams once.
    fn stage_payload(
        &self,
        length: u64,
        source: impl Read,
    ) -> Result<(Transaction, PathBuf, PayloadId)> {
        let transaction = Transaction::begin(self)?;
        let staged = transaction.staging_file("wrapped")?;
        let result = (|| {
            let mut output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&staged)
                .map_err(failure)?;
            // Everything runs synchronously in this call, so a caller-thread
            // reader needs only shared cells, not Send or a mutex.
            let digest = std::rc::Rc::new(std::cell::RefCell::new(DigestContext::new(&SHA256)));
            let hasher = std::rc::Rc::clone(&digest);
            let source = std::rc::Rc::new(std::cell::RefCell::new(Some(source)));
            let handle = std::rc::Rc::clone(&source);
            let mut cursor = 0u64;
            let reader = move |offset: u64, buffer: &mut [u8]| -> Result<()> {
                if offset != cursor {
                    return Err("payload source must be read sequentially".into());
                }
                let mut guard = handle.borrow_mut();
                let reader = guard
                    .as_mut()
                    .ok_or_else(|| "payload source is closed".to_string())?;
                reader.read_exact(buffer).map_err(failure)?;
                cursor += buffer.len() as u64;
                hasher.borrow_mut().update(buffer);
                Ok(())
            };
            let mut writer = Writer::new(&mut output, PAYLOAD_FORMAT)?;
            writer.add(COMPANION_TENSOR, length, Source::Reader(Box::new(reader)))?;
            writer.finish()?;
            let mut guard = source.borrow_mut();
            if let Some(reader) = guard.as_mut() {
                let mut extra = [0u8; 1];
                if reader.read(&mut extra).map_err(failure)? != 0 {
                    return Err(format!("source carries bytes beyond the declared {length}").into());
                }
            }
            drop(guard);
            output.sync_all().map_err(failure)?;
            drop(output);
            let context = std::rc::Rc::try_unwrap(digest)
                .map_err(|_| "payload digest remains shared".to_string())?
                .into_inner();
            Ok((
                transaction,
                staged.clone(),
                PayloadId::from_digest(context.finish()),
            ))
        })();
        match result {
            Ok(value) => Ok(value),
            Err(error) => {
                let _ = fs::remove_file(&staged);
                Err(error)
            }
        }
    }

    fn publish_payload(
        &self,
        _transaction: &Transaction,
        staged: &Path,
        id: PayloadId,
        length: u64,
    ) -> Result<PayloadObject> {
        let _publication = Lock::shared(self.publication_lock())?;
        let result = (|| {
            let destination = self.root.join("blobs").join(id.file_name());
            self.install_payload(staged, &destination, id, length)?;
            syncdir(&self.root.join("blobs"))?;
            let guard = Lock::shared(self.lock_path("blobs", &id.to_hex()))?;
            let slice = Container::open(&destination)?.tensor(COMPANION_TENSOR)?;
            if slice.len() != length {
                return Err(format!(
                    "installed object {id} holds {} bytes, expected {length}",
                    slice.len()
                )
                .into());
            }
            verify_slice(&slice, id, None)?;
            Ok(PayloadObject {
                id,
                path: destination,
                slice,
                _guard: Arc::new(guard),
            })
        })();
        if result.is_err() {
            let _ = fs::remove_file(staged);
        }
        result
    }

    /// Install one verified wrapper object without ever replacing existing bytes.
    fn install_payload(
        &self,
        staged: &Path,
        destination: &Path,
        id: PayloadId,
        length: u64,
    ) -> Result<()> {
        make_immutable(staged)?;
        match fs::hard_link(staged, destination) {
            Ok(()) => {
                fs::remove_file(staged).map_err(failure)?;
                Ok(())
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                let slice = Container::open(destination)?.tensor(COMPANION_TENSOR)?;
                if slice.len() != length {
                    return Err(format!(
                        "object {id} holds {} bytes, expected {length}",
                        slice.len()
                    )
                    .into());
                }
                verify_slice(&slice, id, None)?;
                make_immutable(destination)?;
                fs::remove_file(staged).map_err(failure)
            }
            Err(error) => Err(failure(error)),
        }
    }

    /// Install one legacy container after hashing its complete bytes.
    fn install_legacy(&self, staged: &Path, destination: &Path, id: PayloadId) -> Result<()> {
        make_immutable(staged)?;
        match fs::hard_link(staged, destination) {
            Ok(()) => {
                fs::remove_file(staged).map_err(failure)?;
                Ok(())
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                verify_file(destination, id)?;
                make_immutable(destination)?;
                fs::remove_file(staged).map_err(failure)
            }
            Err(error) => Err(failure(error)),
        }
    }

    /// Install one selected deployment's ready closure as a directory cache entry.
    ///
    /// Every required payload is hashed against its signed identity before the entry
    /// becomes ready, absent payloads are streamed into the object store once, and
    /// the entry primary is rebuilt from pinned metadata tensors with locations
    /// rewritten to the entry-relative companions. The returned entry stays leased
    /// until the caller drops it, so the ref switch and a subsequent load cannot race
    /// eviction.
    pub fn install_package_staged(
        &self,
        source: PackageSource<'_>,
        model_key: &str,
        entry: &str,
    ) -> Result<Entry> {
        let plan = self.plan_acquisition(source, &derivation::Limits::default())?;
        self.install_planned(plan, model_key, entry)
    }

    /// Install and select one ready closure.
    pub fn install_package(
        &self,
        source: PackageSource<'_>,
        model_key: &str,
        entry: &str,
    ) -> Result<Entry> {
        let entry = self.install_package_staged(source, model_key, entry)?;
        self.publish_entry(&entry)?;
        Ok(entry)
    }

    fn write_ref(&self, model: &str, entry: &str) -> Result<()> {
        let directory = self.root.join("refs");
        let temporary = directory.join(format!(".ref-{}", token()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(failure)?;
        io::Write::write_all(&mut file, format!("{entry}\n").as_bytes()).map_err(failure)?;
        file.sync_all().map_err(failure)?;
        drop(file);
        fs::rename(&temporary, directory.join(model)).map_err(failure)?;
        syncdir(&directory)
    }

    fn entry_lock(&self, model: &str, entry: &str) -> PathBuf {
        self.lock_path("entries", &format!("{model}--{entry}"))
    }

    fn publication_lock(&self) -> PathBuf {
        self.root.join("locks/publication.lock")
    }

    fn lock_path(&self, kind: &str, key: &str) -> PathBuf {
        self.root
            .join("locks")
            .join(kind)
            .join(format!("{key}.lock"))
    }
}

/// One transaction's stable lock and working-file namespace.
struct Transaction {
    root: PathBuf,
    token: String,
    _lease: Lock,
}

impl Transaction {
    fn begin(root: &Root) -> Result<Self> {
        let token = token();
        let lease = Lock::exclusive(root.lock_path("staging", &token))?;
        Ok(Self {
            root: root.path().to_owned(),
            token,
            _lease: lease,
        })
    }

    fn staging_file(&self, directory: &str) -> Result<PathBuf> {
        let path = self.root.join("staging").join(directory).join(&self.token);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(failure)?;
        }
        Ok(path)
    }
}

/// Counts payload bytes at actual reader boundaries for one acquisition.
struct ProgressReader<R> {
    inner: R,
    progress: Option<progress::AcquisitionProgress>,
}

impl<R: Read> Read for ProgressReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let count = self.inner.read(buffer)?;
        if let Some(progress) = &self.progress {
            progress.record_local_source_bytes(count as u64);
            progress.record_verification_bytes(count as u64);
        }
        Ok(count)
    }
}

/// Streams one verified payload slice into the object store.
struct SliceReader {
    slice: Slice,
    offset: u64,
}

impl Read for SliceReader {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let remaining = self.slice.len().saturating_sub(self.offset);
        let count = (output.len() as u64).min(remaining) as usize;
        if count == 0 {
            return Ok(0);
        }
        self.slice
            .read_at(self.offset, &mut output[..count])
            .map_err(|error| io::Error::other(error.to_string()))?;
        self.offset += count as u64;
        Ok(count)
    }
}

/// An authenticated package and the primary it was read from.
///
/// The pair is produced by verification itself: the metadata primary is read
/// first and the envelope bytes from that same file are what get verified, so a
/// caller cannot present one package's signatures with another package's bytes.
pub struct PackageSource<'a> {
    verified: Vec<protection::VerifiedPackage>,
    metadata: reader::Metadata,
    deployments: Vec<String>,
    union_ids: Vec<Vec<u8>>,
    asset_payloads: BTreeSet<Vec<u8>>,
    _source: std::marker::PhantomData<&'a ()>,
}

/// Return every payload identity a selected closure needs, including derivation
/// dependencies reachable only through recipes. The selected deployment
/// catalogs, not the package-wide hint catalog, authorize every target and
/// source. Bounds and duplicate/cycle rules come from the derivation parser.
pub fn selected_derivation_closure(
    metadata: &reader::Metadata,
    verified: &[protection::VerifiedPackage],
    limits: &derivation::Limits,
) -> Result<BTreeSet<Vec<u8>>> {
    let mut needed: BTreeSet<Vec<u8>> = BTreeSet::new();
    for package in verified {
        needed.extend(reader::required_payloads(package)?);
    }
    let located = located_payload_ids(metadata);
    let missing: BTreeSet<Vec<u8>> = needed
        .iter()
        .filter(|id| !located.contains(*id))
        .cloned()
        .collect();
    if missing.is_empty() {
        let _ = Root::merged_catalog(verified, needed.iter().cloned())?;
        return Ok(needed);
    }
    let set = metadata.derivations.as_ref().ok_or_else(|| {
        invalid("payload is neither located nor derivable by a selected deployment")
    })?;
    let selected = selected_derivation_set(set, &missing, &located, limits)?;
    let ids = selected_derivation_ids(&selected);
    let mut authorized = needed.clone();
    authorized.extend(ids.iter().cloned());
    let catalog = Root::merged_catalog(verified, authorized.iter().cloned())?;
    let plans = derivation::from_wire(&selected, &catalog, limits)?;
    derivation::order(&plans, limits)?;
    needed.extend(ids);
    Ok(needed)
}

impl<'a> PackageSource<'a> {
    pub fn open(
        path: &Path,
        publishers: &[crate::runtime_package::Publisher],
        selection: protection::Selection,
    ) -> Result<Self> {
        let metadata = reader::Metadata::open(path)?;
        let verified = protection::verify(&metadata.envelope, publishers, selection)?;
        if !verified.is_executable() {
            return Err("installation requires a selected prepared deployment".into());
        }
        let deployments = verified
            .deployment()
            .map(|deployment| vec![deployment.deployment_id.clone()])
            .unwrap_or_default();
        let asset_payloads = package_asset_payloads(&verified);
        Ok(Self {
            verified: vec![verified],
            metadata,
            deployments,
            union_ids: Vec::new(),
            asset_payloads,
            _source: std::marker::PhantomData,
        })
    }

    /// Pair an already verified package with the primary it was read from.
    pub fn new(metadata: reader::Metadata, verified: protection::VerifiedPackage) -> Result<Self> {
        if !verified.matches_envelope(&metadata.envelope) {
            return Err("verified package does not match this primary".into());
        }
        if !verified.is_executable() {
            return Err("installation requires a selected prepared deployment".into());
        }
        let deployments = verified
            .deployment()
            .map(|deployment| vec![deployment.deployment_id.clone()])
            .unwrap_or_default();
        let asset_payloads = package_asset_payloads(&verified);
        Ok(Self {
            verified: vec![verified],
            metadata,
            deployments,
            union_ids: Vec::new(),
            asset_payloads,
            _source: std::marker::PhantomData,
        })
    }

    /// Verify every requested deployment from this primary and union their closures.
    pub fn union(
        mut self,
        publishers: &[crate::runtime_package::Publisher],
        selections: &[protection::Selection],
    ) -> Result<Self> {
        if selections.is_empty() {
            return Err("installation requires at least one selected deployment".into());
        }
        self.verified = Vec::with_capacity(selections.len());
        self.deployments.clear();
        self.asset_payloads.clear();
        self.union_ids.clear();
        for selection in selections {
            let verified = protection::verify(&self.metadata.envelope, publishers, *selection)?;
            let deployment = verified
                .deployment()
                .ok_or_else(|| "selection has no authenticated deployment".to_string())?;
            if !self.deployments.contains(&deployment.deployment_id) {
                self.deployments.push(deployment.deployment_id.clone());
            }
            self.asset_payloads.extend(package_asset_payloads(&verified));
            for id in reader::required_payloads(&verified)? {
                if !self.union_ids.contains(&id) {
                    self.union_ids.push(id);
                }
            }
            self.verified.push(verified);
        }
        Ok(self)
    }

    pub fn verified(&self) -> &[protection::VerifiedPackage] {
        &self.verified
    }

    pub fn metadata(&self) -> &reader::Metadata {
        &self.metadata
    }
}

/// Write the entry primary from bounded metadata and authenticated asset uses.
///
/// Only the signed envelope, the rewritten locations, and payload tensors that
/// authenticated assets reference are copied. Untrusted tensor names in the
/// source header grant nothing: any other `frontend.*` entry is ignored, and
/// asset bytes keep their signed lengths. The primary streams into its staged
/// file, so no payload-sized buffer is ever formed.
fn write_primary(
    staging: &Staging,
    source: &PackageSource<'_>,
    locations: Vec<wire::PayloadLocation>,
    progress: Option<&progress::AcquisitionProgress>,
) -> Result<()> {
    use prost::Message;
    let metadata = &source.metadata;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(staging.directory().join("model.safetensors"))
        .map_err(failure)?;
    let mut writer = Writer::new(&mut file, crate::package::container::PAYLOAD_FORMAT)?;
    let manifest = metadata.primary().tensor(reader::MANIFEST)?;
    writer.add(reader::MANIFEST, manifest.len(), Source::Slice(manifest))?;
    let encoded = wire::LocationSet {
        version: metadata.locations.version,
        locations,
        available_deployments: source.deployments.clone(),
    }
    .encode_to_vec();
    writer.add(
        reader::LOCATIONS,
        encoded.len() as u64,
        Source::Bytes(encoded),
    )?;
    let placed: BTreeMap<Vec<u8>, &wire::PayloadLocation> = metadata
        .locations
        .locations
        .iter()
        .map(|location| (location.payload_id.clone(), location))
        .collect();
    let mut embedded = BTreeSet::new();
    let mut assets: Vec<&wire::PayloadUse> = Vec::new();
    for verified in &source.verified {
        assets.extend(
            verified
                .package()
                .assets
                .iter()
                .filter_map(|asset| asset.payload.as_ref()),
        );
    }
    for asset in assets {
        let Some(slice) = asset.slice.as_ref() else {
            continue;
        };
        let Some(location) = placed.get(&slice.payload_id) else {
            continue;
        };
        if !location.file.is_empty() || !embedded.insert(location.tensor.clone()) {
            continue;
        }
        let tensor = metadata.primary().tensor(&location.tensor)?;
        let declared = PayloadId::from_bytes(
            slice
                .payload_id
                .as_slice()
                .try_into()
                .map_err(|_| "embedded asset identity is not 32 bytes".to_string())?,
        );
        let digest = tensor.digest_with(progress)?;
        if let Some(progress) = progress {
            progress.record_local_source_bytes(tensor.len());
        }
        if digest != declared {
            return Err("embedded asset bytes differ from their signed identity".into());
        }
        let length = tensor.len();
        let slice = tensor.clone();
        let mut cursor = 0u64;
        let reader_progress = progress.cloned();
        let reader = move |offset: u64, buffer: &mut [u8]| -> Result<()> {
            if offset != cursor {
                return Err("embedded asset copy must read sequentially".into());
            }
            slice.read_at(cursor, buffer)?;
            if let Some(progress) = &reader_progress {
                progress.record_local_source_bytes(buffer.len() as u64);
            }
            cursor += buffer.len() as u64;
            Ok(())
        };
        writer.add(
            &location.tensor,
            length,
            Source::Reader(Box::new(reader)),
        )?;
    }
    writer.finish()?;
    file.sync_all().map_err(failure)?;
    Ok(())
}

fn token() -> String {
    format!(
        "{}-{:x}",
        std::process::id(),
        TRANSACTION.fetch_add(1, Ordering::Relaxed)
    )
}

/// One staged entry tree that is not yet published.
pub struct Staging {
    root: Root,
    _transaction: Transaction,
    model: String,
    entry: String,
    directory: PathBuf,
    copied: bool,
    published: bool,
    /// Exclusive entry-name reservation held until this tree is committed or dropped.
    reservation: Option<Lock>,
}

impl Staging {
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// The entry id this staging transaction will publish.
    pub fn entry(&self) -> &str {
        &self.entry
    }

    /// Link one published object into the staged tree.
    pub fn link(&mut self, object: &PayloadObject, relative: &str) -> Result<()> {
        validate_relative_path(relative)?;
        let _publication = Lock::shared(self.root.publication_lock())?;
        let destination = self.directory.join(relative);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(failure)?;
        }
        match fs::hard_link(object.path(), &destination) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::CrossesDevices => Err(format!(
                "object {} cannot be hardlinked from another filesystem",
                object.id()
            )
            .into()),
            Err(error) => Err(failure(error)),
        }
    }

    /// Copy one published object into the staged tree for a standalone export.
    pub fn copy(&mut self, object: &PayloadObject, relative: &str) -> Result<()> {
        validate_relative_path(relative)?;
        let _publication = Lock::shared(self.root.publication_lock())?;
        let destination = self.directory.join(relative);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(failure)?;
        }
        fs::copy(object.path(), &destination).map_err(failure)?;
        sync_file(&destination).map_err(failure)?;
        self.copied = true;
        Ok(())
    }

    /// Write one small file such as the entry primary.
    pub fn write(&mut self, relative: &str, bytes: &[u8]) -> Result<()> {
        validate_relative_path(relative)?;
        let destination = self.directory.join(relative);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(failure)?;
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&destination)
            .map_err(failure)?;
        io::Write::write_all(&mut file, bytes).map_err(failure)?;
        file.sync_all().map_err(failure)
    }

    /// Publish this complete tree as a cache entry and return its lease.
    pub fn commit(mut self) -> Result<Entry> {
        if self.copied {
            return Err("copied payloads belong to a standalone export".into());
        }
        let _publication = Lock::shared(self.root.publication_lock())
            .map_err(|error| format!("lock cache publication: {error}"))?;
        synchronize(&self.directory)
            .map_err(|error| format!("sync staged entry: {error}"))?;
        seal(&self.directory).map_err(|error| format!("seal staged entry: {error}"))?;
        let _coordination = Lock::exclusive(self.root.lock_path("models", &self.model))
            .map_err(|error| format!("lock model coordination: {error}"))?;
        let models = self.root.path().join("models").join(&self.model);
        fs::create_dir_all(&models)
            .map_err(|error| format!("create model directory {}: {error}", models.display()))?;
        let destination = models.join(&self.entry);
        match fs::symlink_metadata(&destination) {
            Ok(_) => {
                return Err(
                    format!("model entry {}/{} already exists", self.model, self.entry).into(),
                );
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(failure(error)),
        }
        fs::rename(&self.directory, &destination).map_err(|error| {
            format!(
                "move staged entry {} to {}: {error}",
                self.directory.display(),
                destination.display()
            )
        })?;
        syncdir(&models).map_err(|error| format!("sync model directory: {error}"))?;
        syncdir(&self.root.path().join("models"))
            .map_err(|error| format!("sync models directory: {error}"))?;
        syncdir(&self.root.path().join("staging/trees"))
            .map_err(|error| format!("sync staging trees directory: {error}"))?;
        // A named reservation already holds this entry's lock exclusively on one descriptor;
        // downgrading that same descriptor avoids acquiring a second shared lock on a new
        // descriptor while the exclusive lock is held, which would deadlock.
        let lease = match self.reservation.take() {
            Some(reservation) => reservation
                .downgrade()
                .map_err(|error| format!("downgrade entry reservation: {error}"))?,
            None => Lock::shared(self.root.entry_lock(&self.model, &self.entry))
                .map_err(|error| format!("lock entry lease: {error}"))?,
        };
        let (model, id) = (self.model.clone(), self.entry.clone());
        self.published = true;
        Ok(Entry {
            model,
            id,
            directory: destination,
            _lease: lease,
        })
    }

    /// Publish this tree outside the cache as a standalone export artifact.
    ///
    /// A cross-filesystem destination is assembled in a hidden sibling directory
    /// and renamed into place, so a failed or interrupted export never exposes a
    /// partial tree and never replaces an existing output.
    pub fn commit_standalone(mut self, destination: &Path) -> Result<()> {
        synchronize(&self.directory)?;
        if fs::symlink_metadata(destination).is_ok() {
            return Err(format!(
                "export destination {} already exists",
                destination.display()
            )
            .into());
        }
        let parent = parent_of(destination);
        let name = destination
            .file_name()
            .ok_or_else(|| "export destination has no file name".to_string())?
            .to_string_lossy()
            .into_owned();
        let staging = parent.join(format!(".{name}.flare-{}", token()));
        fs::create_dir_all(parent).map_err(failure)?;
        if let Err(error) = copy_tree(&self.directory, &staging) {
            let _ = remove_tree(&staging);
            return Err(error);
        }
        if let Err(error) = synchronize(&staging) {
            let _ = remove_tree(&staging);
            return Err(error);
        }
        if let Err(error) = rename_noreplace(&staging, destination) {
            let _ = remove_tree(&staging);
            return Err(failure(error));
        }
        syncdir(parent)?;
        let _ = remove_tree(&self.directory);
        self.published = true;
        Ok(())
    }

    /// Discard the staged tree.
    pub fn abort(mut self) {
        let _ = remove_tree(&self.directory);
        self.published = true;
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        if !self.published {
            let _ = remove_tree(&self.directory);
        }
    }
}

/// One published entry held open by a lease.
pub struct Entry {
    model: String,
    id: String,
    directory: PathBuf,
    _lease: Lock,
}

impl Entry {
    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Pinned handle to one entry-relative file.
    pub fn open(&self, relative: &str) -> Result<Arc<File>> {
        Ok(open_relative(&self.directory, relative)?.1)
    }

    /// One payload held by this entry, parsed from the exact opened handle.
    pub fn payload(&self, relative: &str) -> Result<Slice> {
        let (_, file) = open_relative(&self.directory, relative)?;
        Container::open_pinned(file)?.tensor(COMPANION_TENSOR)
    }
}

/// A remembered acquisition selector: the resolved revision, the entry it produced,
/// and the signed envelope identity authenticated when the source was acquired.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceRef {
    selector: String,
    revision: String,
    model: String,
    entry: String,
    expected_envelope: Option<[u8; 32]>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
enum SourceRefRecord {
    Legacy((String, String, String, String)),
    Versioned {
        version: u32,
        selector: String,
        revision: String,
        model: String,
        entry: String,
        envelope_sha256: String,
    },
}

impl SourceRef {
    pub fn selector(&self) -> &str {
        &self.selector
    }
    pub fn revision(&self) -> &str {
        &self.revision
    }
    pub fn model(&self) -> &str {
        &self.model
    }
    pub fn entry(&self) -> &str {
        &self.entry
    }
    pub fn expected_envelope(&self) -> Option<[u8; 32]> {
        self.expected_envelope
    }
}

/// Manifest facts a cache listing reports without authenticating the entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntrySummary {
    pub form: &'static str,
    pub format_version: u32,
    pub model_identity: String,
    pub hardware_identity: String,
    pub deployments: Vec<DeploymentSummary>,
    pub logical_bytes: u64,
    pub payloads: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeploymentSummary {
    pub id: String,
    pub tensor_parallel: u32,
    /// Ready means this entry holds the deployment's located closure; catalog-only deployments
    /// are metadata that cannot be served from this entry.
    pub ready: bool,
    pub pages: Option<u64>,
    pub state_rows: Option<u64>,
    pub max_sequence_tokens: Option<u64>,
    pub prefill_chunk_size: Option<u64>,
}

/// Reclamation results for one `collect` pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Collected {
    pub staging: usize,
    pub objects: usize,
    pub legacy: usize,
    /// Unlocked staging trees holding a resumable Hugging Face transport cache.
    pub preserved_transport: usize,
}

/// Process-owned file lock; dropping the descriptor releases it.
#[allow(dead_code)]
struct Lock(File);

impl Lock {
    fn shared(path: impl AsRef<Path>) -> Result<Self> {
        let file = open_lock(path.as_ref())?;
        lock(&file, false, true).map_err(failure)?;
        Ok(Self(file))
    }

    fn exclusive(path: impl AsRef<Path>) -> Result<Self> {
        let file = open_lock(path.as_ref())?;
        lock(&file, true, true).map_err(failure)?;
        Ok(Self(file))
    }

    /// Convert this lock to shared on the same open file description.
    fn downgrade(self) -> Result<Self> {
        downgrade_lock(&self.0).map_err(failure)?;
        Ok(self)
    }

    fn try_exclusive(path: impl AsRef<Path>) -> Result<Option<Self>> {
        let file = open_lock(path.as_ref())?;
        match lock(&file, true, false) {
            Ok(()) => Ok(Some(Self(file))),
            Err(error) if contention(&error) => Ok(None),
            Err(error) => Err(failure(error)),
        }
    }
}

/// Whether a failed non-blocking lock reports contention rather than a defect.
fn contention(error: &io::Error) -> bool {
    if error.kind() == io::ErrorKind::WouldBlock {
        return true;
    }
    #[cfg(windows)]
    {
        // Windows reports lock contention as ERROR_LOCK_VIOLATION.
        return error.raw_os_error() == Some(33);
    }
    #[cfg(not(windows))]
    {
        false
    }
}

fn open_lock(path: &Path) -> Result<File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(failure)?;
    }
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(failure)
}

#[cfg(unix)]
fn lock(file: &File, exclusive: bool, blocking: bool) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    let mut mode = if exclusive {
        libc::LOCK_EX
    } else {
        libc::LOCK_SH
    };
    if !blocking {
        mode |= libc::LOCK_NB;
    }
    if unsafe { libc::flock(file.as_raw_fd(), mode) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(windows)]
fn lock(file: &File, exclusive: bool, blocking: bool) -> io::Result<()> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::Storage::FileSystem::{
        LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY, LockFileEx,
    };
    use windows_sys::Win32::System::IO::OVERLAPPED;
    let handle = file.as_raw_handle() as HANDLE;
    let mut overlapped: OVERLAPPED = unsafe { std::mem::zeroed() };
    let mut flags = 0u32;
    if exclusive {
        flags |= LOCKFILE_EXCLUSIVE_LOCK;
    }
    if !blocking {
        flags |= LOCKFILE_FAIL_IMMEDIATELY;
    }
    if unsafe { LockFileEx(handle, flags, 0, u32::MAX, u32::MAX, &mut overlapped) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(unix)]
fn downgrade_lock(file: &File) -> io::Result<()> {
    lock(file, false, true)
}

/// Windows cannot change a lock's type in place, so the exclusive lock is released first.
#[cfg(windows)]
fn downgrade_lock(file: &File) -> io::Result<()> {
    unlock(file)?;
    lock(file, false, true)
}

#[cfg(windows)]
fn unlock(file: &File) -> io::Result<()> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::Storage::FileSystem::UnlockFileEx;
    use windows_sys::Win32::System::IO::OVERLAPPED;
    let handle = file.as_raw_handle() as HANDLE;
    let mut overlapped: OVERLAPPED = unsafe { std::mem::zeroed() };
    if unsafe { UnlockFileEx(handle, 0, u32::MAX, u32::MAX, &mut overlapped) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Stream a source into a staged file, returning its digest and length.
fn write_stream(path: &Path, mut source: impl Read) -> Result<(PayloadId, u64)> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(failure)?;
    let mut context = DigestContext::new(&SHA256);
    let mut buffer = vec![0; 1 << 20];
    let mut written = 0u64;
    loop {
        let count = match source.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(failure(error)),
        };
        context.update(&buffer[..count]);
        io::Write::write_all(&mut file, &buffer[..count]).map_err(failure)?;
        written += count as u64;
    }
    file.sync_all().map_err(failure)?;
    drop(file);
    if written == 0 {
        let _ = fs::remove_file(path);
        return Err("empty payload".into());
    }
    Ok((PayloadId::from_digest(context.finish()), written))
}

/// Validate a cache hit against the caller's declared identity and length.
///
/// Hashing reports verification bytes as it reads, so a large cached payload is
/// visible while it is being checked rather than only after it completes.
fn validate_hit(
    object: &PayloadObject,
    declared: PayloadId,
    length: u64,
    progress: Option<&progress::AcquisitionProgress>,
) -> Result<()> {
    if object.length() != length {
        return Err(format!(
            "object {declared} holds {} bytes, expected {length}",
            object.length()
        )
        .into());
    }
    verify_slice(object.slice(), declared, progress)
}

/// Hash a complete file and compare it with a declared container identity.
fn verify_file(path: &Path, declared: PayloadId) -> Result<()> {
    let mut file = File::open(path).map_err(failure)?;
    let mut context = DigestContext::new(&SHA256);
    let mut buffer = vec![0; 1 << 20];
    loop {
        let count = match file.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(failure(error)),
        };
        context.update(&buffer[..count]);
    }
    if PayloadId::from_digest(context.finish()) != declared {
        return Err(format!("container {declared} does not match its stored bytes").into());
    }
    Ok(())
}

/// Hash the installed payload bytes and compare them with their identity.
fn verify_slice(
    slice: &Slice,
    declared: PayloadId,
    progress: Option<&progress::AcquisitionProgress>,
) -> Result<()> {
    let mut context = DigestContext::new(&SHA256);
    let mut buffer = vec![0; 1 << 20];
    let mut offset = 0u64;
    while offset < slice.len() {
        let count = (slice.len() - offset).min(buffer.len() as u64) as usize;
        slice.read_at(offset, &mut buffer[..count])?;
        context.update(&buffer[..count]);
        if let Some(progress) = progress {
            progress.record_verification_bytes(count as u64);
        }
        offset += count as u64;
    }
    if PayloadId::from_digest(context.finish()) != declared {
        return Err(format!("installed object {declared} does not match its payload bytes").into());
    }
    Ok(())
}

/// Hash a supplied source and compare it with an installed object's identity.
fn verify_stream(declared: PayloadId, length: u64, mut source: impl Read) -> Result<()> {
    let mut context = DigestContext::new(&SHA256);
    let mut buffer = vec![0; 1 << 20];
    let mut read = 0u64;
    loop {
        let count = match source.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(failure(error)),
        };
        read += count as u64;
        if read > length {
            return Err(format!("supplied payload is longer than {length} bytes").into());
        }
        context.update(&buffer[..count]);
    }
    if read != length || PayloadId::from_digest(context.finish()) != declared {
        return Err(format!("supplied payload does not match object {declared}").into());
    }
    Ok(())
}

/// Hash one file by streaming it.
fn hash_file(path: &Path) -> Result<PayloadId> {
    let mut file = File::open(path).map_err(failure)?;
    let mut context = DigestContext::new(&SHA256);
    let mut buffer = vec![0; 1 << 20];
    loop {
        let count = match file.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(failure(error)),
        };
        context.update(&buffer[..count]);
    }
    Ok(PayloadId::from_digest(context.finish()))
}

fn validate_name(name: &str, what: &str) -> Result<()> {
    if name.is_empty() || name.len() > MAX_NAME {
        return Err(format!("{what} must be 1 to {MAX_NAME} bytes").into());
    }
    let invalid = name == "."
        || name == ".."
        || name.bytes().any(|byte| {
            !(byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'+'))
        });
    if invalid {
        return Err(format!("{what} {name:?} contains unsupported characters").into());
    }
    Ok(())
}

/// One legacy Hugging Face transport cache tree, identified by a top-level
/// repository folder with its own blob directory. Flare previously created these
/// directly in `--cache-dir` and under `staging/trees`; the shared transport root
/// has a `version` marker and no repository folders at its top level.
pub fn is_legacy_hub_transport(path: &Path) -> bool {
    let Ok(entries) = fs::read_dir(path) else {
        return false;
    };
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let repository = ["models--", "datasets--", "spaces--"]
            .iter()
            .any(|prefix| name.starts_with(prefix));
        if file_type.is_dir() && repository && entry.path().join("blobs").is_dir() {
            return true;
        }
    }
    false
}

fn read_dir(path: &Path) -> Result<Vec<PathBuf>> {
    let mut items = Vec::new();
    for entry in fs::read_dir(path).map_err(failure)? {
        items.push(entry.map_err(failure)?.path());
    }
    items.sort();
    Ok(items)
}

/// Record the identity of every file in one tree for reachability checks.
fn collect_inodes(path: &Path, live: &mut HashSet<(u64, u64)>) -> Result<()> {
    let metadata = fs::symlink_metadata(path).map_err(failure)?;
    if metadata.is_dir() {
        for item in read_dir(path)? {
            collect_inodes(&item, live)?;
        }
        return Ok(());
    }
    live.extend(inode(path));
    Ok(())
}

#[cfg(unix)]
fn inode(path: &Path) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    fs::metadata(path)
        .ok()
        .map(|metadata| (metadata.dev(), metadata.ino()))
}

/// Windows volume serial and file index identify one immutable object.
///
/// Reclamation refuses to guess: without an identity the sweep fails rather than
/// treating published objects as unreachable. Untested on Windows in this build.
#[cfg(windows)]
fn inode(path: &Path) -> Option<(u64, u64)> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
    };
    let file = File::open(path).ok()?;
    let mut information: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
    if unsafe { GetFileInformationByHandle(file.as_raw_handle().cast(), &mut information) } == 0 {
        return None;
    }
    let index = u64::from(information.nFileIndexHigh) << 32 | u64::from(information.nFileIndexLow);
    Some((u64::from(information.dwVolumeSerialNumber), index))
}

/// Publication parent: an empty parent means the current directory.
fn parent_of(destination: &Path) -> &Path {
    destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

/// Rename without ever replacing an existing destination.
#[cfg(target_os = "linux")]
pub(crate) fn rename_noreplace(from: &Path, to: &Path) -> io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let from = CString::new(from.as_os_str().as_bytes()).map_err(io::Error::other)?;
    let to = CString::new(to.as_os_str().as_bytes()).map_err(io::Error::other)?;
    let status = unsafe {
        libc::syscall(
            libc::SYS_renameat2,
            libc::AT_FDCWD,
            from.as_ptr(),
            libc::AT_FDCWD,
            to.as_ptr(),
            1u32, // RENAME_NOREPLACE
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn rename_noreplace(from: &Path, to: &Path) -> io::Result<()> {
    if to.try_exists()? {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "destination exists",
        ));
    }
    fs::rename(from, to)
}

fn remove_tree(path: &Path) -> Result<()> {
    match fs::remove_dir_all(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(failure(error)),
    }
}

/// Recreate one entry tree: directories, hardlinked files, explicit copies off-device.
fn export_tree(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination).map_err(failure)?;
    for item in read_dir(source)? {
        let name = item
            .file_name()
            .ok_or_else(|| "cache entry contains an unnamed path".to_string())?
            .to_owned();
        let target = destination.join(&name);
        let kind = fs::symlink_metadata(&item).map_err(failure)?.file_type();
        if kind.is_symlink() {
            return Err(format!("cache entry contains a symlink: {}", item.display()).into());
        }
        if kind.is_dir() {
            export_tree(&item, &target)?;
        } else if kind.is_file() {
            match fs::hard_link(&item, &target) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::CrossesDevices => {
                    fs::copy(&item, &target).map_err(failure)?;
                }
                Err(error) => return Err(failure(error)),
            }
        } else {
            return Err(format!("cache entry is not a regular file: {}", item.display()).into());
        }
    }
    Ok(())
}

fn copy_tree(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination).map_err(failure)?;
    for item in read_dir(source)? {
        let name = item
            .file_name()
            .ok_or_else(|| "cache entry contains an unnamed path".to_string())?
            .to_owned();
        let target = destination.join(name);
        let kind = fs::symlink_metadata(&item).map_err(failure)?.file_type();
        if kind.is_symlink() {
            return Err(format!("export tree contains a symlink: {}", item.display()).into());
        }
        if kind.is_dir() {
            copy_tree(&item, &target)?;
        } else if kind.is_file() {
            fs::copy(&item, &target).map_err(failure)?;
        } else {
            return Err(format!("export tree is not a regular file: {}", item.display()).into());
        }
    }
    Ok(())
}

fn make_immutable(path: &Path) -> Result<()> {
    let mut permissions = fs::metadata(path).map_err(failure)?.permissions();
    permissions.set_readonly(true);
    fs::set_permissions(path, permissions).map_err(failure)
}

/// Flush one file to its device.
///
/// Windows requires write access to flush a file's buffers, and a sealed (read-only) file cannot be
/// opened for writing. Sealing follows the writer's own flush, so such a file is skipped rather
/// than failed; a writable file that cannot be flushed is still an error.
#[cfg(windows)]
pub(crate) fn sync_file(path: &Path) -> io::Result<()> {
    match OpenOptions::new().write(true).open(path) {
        Ok(file) => file.sync_all(),
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(not(windows))]
pub(crate) fn sync_file(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

pub(crate) fn synchronize(directory: &Path) -> Result<()> {
    for item in read_dir(directory)? {
        let metadata = fs::symlink_metadata(&item).map_err(failure)?;
        let kind = metadata.file_type();
        if kind.is_symlink() {
            return Err(format!("staged entry contains a symlink: {}", item.display()).into());
        }
        if kind.is_dir() {
            synchronize(&item)?;
        } else if kind.is_file() {
            sync_file(&item).map_err(failure)?;
        } else {
            return Err(format!("staged entry is not a regular file: {}", item.display()).into());
        }
    }
    syncdir(directory)
}

/// Make every regular file in a published entry read-only before publication.
fn seal(directory: &Path) -> Result<()> {
    for item in read_dir(directory)? {
        let metadata = fs::symlink_metadata(&item).map_err(failure)?;
        if metadata.is_dir() {
            seal(&item)?;
        } else if metadata.is_file() {
            make_immutable(&item)?;
        } else {
            return Err(format!("staged entry is not a regular file: {}", item.display()).into());
        }
    }
    Ok(())
}

#[cfg(unix)]
pub(crate) fn syncdir(path: &Path) -> Result<()> {
    File::open(path)
        .map_err(failure)?
        .sync_all()
        .map_err(failure)
}

#[cfg(not(unix))]
pub(crate) fn syncdir(path: &Path) -> Result<()> {
    let _ = path;
    Ok(())
}

fn failure(error: io::Error) -> crate::Error {
    error.to_string().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::package::container::{normalized_payload_companion, path_conflicts};
    use std::sync::Barrier;
    use std::sync::mpsc;
    use std::time::Duration;

    struct Temporary(PathBuf);

    impl Temporary {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("helios-package-cache-{}", token()));
            let _ = fs::remove_dir_all(&path);
            Self(path)
        }

        fn root(&self) -> Root {
            Root::open(&self.0).unwrap()
        }
    }

    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn install(root: &Root, bytes: &[u8]) -> PayloadObject {
        root.insert_payload(PayloadId::of(bytes), bytes.len() as u64, bytes)
            .unwrap()
    }

    fn objects(root: &Root) -> usize {
        read_dir(&root.path().join("blobs"))
            .unwrap()
            .into_iter()
            .filter(|path| path.is_file())
            .count()
    }

    fn staged(root: &Root) -> usize {
        ["staging/incoming", "staging/wrapped"]
            .into_iter()
            .map(|directory| read_dir(&root.path().join(directory)).unwrap().len())
            .sum()
    }

    #[cfg(unix)]
    fn inode_of(path: &Path) -> u64 {
        use std::os::unix::fs::MetadataExt;
        fs::metadata(path).unwrap().ino()
    }

    fn derivation_hint(target: &[u8; 32], source: &[u8; 32]) -> wire::Derivation {
        wire::Derivation {
            target_payload_id: target.to_vec(),
            operations: vec![wire::DerivationOperation {
                operation: Some(wire::derivation_operation::Operation::Copy(
                    wire::DerivationCopy {
                        source_payload_id: source.to_vec(),
                        source_offset: 0,
                        destination_offset: 0,
                        contiguous_bytes: 1,
                        extents: vec![],
                        source_strides: vec![],
                        destination_strides: vec![],
                    },
                )),
            }],
        }
    }

    #[test]
    fn selected_derivation_walk_follows_transitive_missing_sources() {
        let a = [1u8; 32];
        let b = [2u8; 32];
        let c = [3u8; 32];
        let unrelated = [4u8; 32];
        let unrelated_source = [5u8; 32];
        let set = wire::DerivationSet {
            version: derivation::DERIVATION_VERSION,
            derivations: vec![
                derivation_hint(&a, &b),
                derivation_hint(&b, &c),
                derivation_hint(&unrelated, &unrelated_source),
            ],
        };
        let selected = super::selected_derivation_set(
            &set,
            &BTreeSet::from([a.to_vec()]),
            &BTreeSet::from([c.to_vec()]),
            &derivation::Limits::default(),
        )
        .unwrap();
        let targets: BTreeSet<Vec<u8>> = selected
            .derivations
            .iter()
            .map(|hint| hint.target_payload_id.clone())
            .collect();
        assert_eq!(targets, BTreeSet::from([a.to_vec(), b.to_vec()]));
    }

    #[test]
    fn installed_payloads_are_content_addressed_and_idempotent() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let bytes = b"layer zero weights".repeat(9);
        let first = install(&root, &bytes);
        assert_eq!(first.length(), bytes.len() as u64);
        assert_eq!(first.slice().len(), bytes.len() as u64);
        let second = install(&root, &bytes);
        assert_eq!(first.path(), second.path());
        assert_eq!(objects(&root), 1);
        assert!(root.payload(PayloadId::of(&bytes)).unwrap().is_some());
        assert!(root.payload(PayloadId::of(b"absent")).unwrap().is_none());
    }

    #[test]
    fn cache_hit_verifies_without_staging_writes() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let bytes = b"unchanged layer bytes".repeat(16);
        let first = install(&root, &bytes);
        let first_path = first.path().to_path_buf();
        #[cfg(unix)]
        let inode = inode_of(first.path());
        let length = fs::metadata(first.path()).unwrap().len();
        drop(first);

        let hit = install(&root, &bytes);
        assert_eq!(hit.path(), first_path);
        #[cfg(unix)]
        assert_eq!(inode_of(hit.path()), inode);
        assert_eq!(fs::metadata(hit.path()).unwrap().len(), length);
        assert_eq!(objects(&root), 1);
        assert_eq!(staged(&root), 0);

        // A hit still rejects a source that disagrees with the installed bytes.
        assert!(
            root.insert_payload(
                PayloadId::of(&bytes),
                bytes.len() as u64,
                b"wrong bytes".as_slice()
            )
            .is_err()
        );
        assert_eq!(staged(&root), 0);
    }

    #[test]
    fn unknown_model_has_no_versions() {
        let temporary = Temporary::new();
        let root = temporary.root();
        assert!(root.versions("missing").unwrap().is_empty());
        assert!(root.versions("../invalid").is_err());
        fs::write(root.path().join("models").join("broken"), b"not a directory").unwrap();
        assert!(root.versions("broken").is_err());
    }

    #[test]
    fn entry_origins_survive_selector_updates_and_are_removed_with_versions() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let first = root.stage("model", "first").unwrap().commit().unwrap();
        let second = root.stage("model", "second").unwrap().commit().unwrap();
        for (entry, revision, digest) in [
            (&first, "commit-1", [1; 32]),
            (&second, "commit-2", [2; 32]),
        ] {
            root.publish_source_ref("repository/branch", revision, entry, digest)
                .unwrap();
            root.publish_entry_source("repository/branch", revision, entry, digest)
                .unwrap();
        }
        assert_eq!(
            root.read_source_ref("repository/branch")
                .unwrap()
                .unwrap()
                .entry(),
            "second"
        );
        let origin = root.entry_source(&first).unwrap().unwrap();
        assert_eq!(origin.revision(), "commit-1");
        assert_eq!(origin.expected_envelope(), Some([1; 32]));
        assert!(root.entry_source_refs(&first).unwrap().is_empty());
        assert_eq!(root.entry_source_refs(&second).unwrap().len(), 1);
        fs::copy(
            root.origin_path("model", "first"),
            root.origin_path("model", "second"),
        )
        .unwrap();
        assert!(root.entry_source(&second).is_err());
        drop(first);
        root.evict("model", "first").unwrap();
        assert!(!root.origin_path("model", "first").exists());
    }

    #[test]
    fn source_refs_lease_entries_and_reject_invalid_targets() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let entry = root.stage("model", "entry").unwrap().commit().unwrap();
        let selector = "repository/revision/filename";
        root.publish_source_ref(selector, "commit", &entry, [7; 32]).unwrap();
        drop(entry);
        let (revision, leased) = root.resolve_source_ref(selector).unwrap().unwrap();
        assert_eq!(revision, "commit");
        assert_eq!(
            root.read_source_ref(selector)
                .unwrap()
                .unwrap()
                .expected_envelope(),
            Some([7; 32])
        );
        assert!(root.evict("model", "entry").is_err());
        drop(leased);
        root.evict("model", "entry").unwrap();
        assert!(root.resolve_source_ref(selector).unwrap().is_none());
        let path = root
            .path()
            .join("sources")
            .join(PayloadId::of(selector.as_bytes()).to_string());
        fs::write(
            &path,
            serde_json::to_vec(&(selector, "commit", "../escape", "entry")).unwrap(),
        )
        .unwrap();
        assert!(root.resolve_source_ref(selector).is_err());
        fs::write(
            &path,
            serde_json::to_vec(&("different", "commit", "model", "entry")).unwrap(),
        )
        .unwrap();
        assert!(root.resolve_source_ref(selector).is_err());
    }

    #[test]
    fn same_length_corrupt_object_is_rejected() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let bytes = b"payload that will be corrupted".to_vec();
        let object = install(&root, &bytes);
        let path = object.path().to_owned();
        drop(object);

        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_readonly(false);
        fs::set_permissions(&path, permissions).unwrap();
        let mut wrapper = fs::read(&path).unwrap();
        let last = wrapper.len() - 1;
        wrapper[last] ^= 0xff;
        fs::write(&path, &wrapper).unwrap();

        assert!(root.payload(PayloadId::of(&bytes)).unwrap().is_some());
        assert!(
            verify_slice(
                &root
                    .payload(PayloadId::of(&bytes))
                    .unwrap()
                    .unwrap()
                    .slice(),
                PayloadId::of(&bytes),
                None,
            )
            .is_err()
        );
        assert!(
            root.insert_payload(PayloadId::of(&bytes), bytes.len() as u64, bytes.as_slice())
                .is_err()
        );
    }

    #[test]
    fn insert_rejects_missing_or_mismatched_bytes() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let bytes = b"declared payload";
        assert!(
            root.insert_payload(PayloadId::of(bytes), bytes.len() as u64 + 1, &bytes[..])
                .is_err()
        );
        assert!(
            root.insert_payload(PayloadId::of(b"other"), bytes.len() as u64, &bytes[..])
                .is_err()
        );
        assert!(
            root.capture_payload(bytes.len() as u64 + 2, &bytes[..])
                .is_err()
        );
        assert_eq!(objects(&root), 0);
        assert_eq!(staged(&root), 0);
        assert_eq!(root.collect().unwrap(), Collected::default());
    }

    #[test]
    fn concurrent_installs_converge_on_one_object() {
        let temporary = Temporary::new();
        let root = Arc::new(temporary.root());
        let bytes = b"shared expert weights".repeat(64);
        let barrier = Arc::new(Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let root = Arc::clone(&root);
                let barrier = Arc::clone(&barrier);
                let bytes = bytes.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    install(&root, &bytes).id()
                })
            })
            .collect();
        let ids: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert!(ids.iter().all(|id| *id == ids[0]));
        assert_eq!(objects(&root), 1);
        assert_eq!(staged(&root), 0);
    }

    #[test]
    fn object_handles_block_collection_with_barriers() {
        let temporary = Temporary::new();
        let root = Arc::new(temporary.root());
        let bytes = b"object held across collection".repeat(8);
        let object = install(&root, &bytes);
        assert_eq!(root.collect().unwrap().objects, 0);

        let (send, receive) = mpsc::channel();
        let collector = {
            let root = Arc::clone(&root);
            std::thread::spawn(move || {
                send.send(root.collect().unwrap()).unwrap();
            })
        };
        // The held object guard excludes the sweep for this object.
        let _ = receive.recv_timeout(Duration::from_millis(200));
        drop(object);
        collector.join().unwrap();
        assert!(!root.payload(PayloadId::of(&bytes)).unwrap().is_none());
        assert_eq!(root.collect().unwrap().objects, 1);
    }

    #[test]
    fn publication_coordination_excludes_collection() {
        let temporary = Temporary::new();
        let root = Arc::new(temporary.root());
        let bytes = b"publication coordination".repeat(4);
        let object = install(&root, &bytes);
        drop(object);

        let held = Lock::shared(root.publication_lock()).unwrap();
        let (send, receive) = mpsc::channel();
        let collector = {
            let root = Arc::clone(&root);
            std::thread::spawn(move || {
                send.send(root.collect().unwrap()).unwrap();
            })
        };
        assert!(receive.recv_timeout(Duration::from_millis(200)).is_err());
        drop(held);
        assert_eq!(
            receive
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .objects,
            1
        );
        collector.join().unwrap();
    }

    #[test]
    fn entry_lease_is_held_before_existence_check() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let object = install(&root, b"entry race payload");
        let mut staging = root.stage("model-h", "only").unwrap();
        staging.link(&object, "weights/layer.safetensors").unwrap();
        staging.commit().unwrap();

        let entry = root.entry("model-h", "only").unwrap();
        // A leased entry cannot be evicted, so the checked directory cannot vanish.
        assert!(root.evict("model-h", "only").is_err());
        assert!(entry.payload("weights/layer.safetensors").is_ok());
        drop(entry);
        root.evict("model-h", "only").unwrap();
        assert!(root.entry("model-h", "only").is_err());
    }

    #[test]
    fn eviction_waits_for_collection_coordination() {
        let temporary = Temporary::new();
        let root = Arc::new(temporary.root());
        let object = install(&root, b"eviction coordination");
        drop(object);
        let mut staging = root.stage("model-i", "kept").unwrap();
        staging.write("model.safetensors", b"primary").unwrap();
        staging.commit().unwrap();
        root.publish("model-i", "kept").unwrap();
        drop(root.resolve("model-i").unwrap());
        let expected = PayloadId::of(b"eviction coordination");

        let held = Lock::exclusive(root.publication_lock()).unwrap();
        let (send, receive) = mpsc::channel();
        let evictor = {
            let root = Arc::clone(&root);
            std::thread::spawn(move || {
                send.send(root.evict("model-i", "kept").is_ok()).unwrap();
            })
        };
        assert!(receive.recv_timeout(Duration::from_millis(200)).is_err());
        drop(held);
        assert!(receive.recv_timeout(Duration::from_secs(5)).unwrap());
        evictor.join().unwrap();
        assert_eq!(root.collect().unwrap().objects, 1);
        assert!(root.payload(expected).unwrap().is_none());
    }

    #[test]
    fn standalone_export_preserves_existing_output_and_leaves_nothing_partial() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let object = install(&root, b"standalone export payload");
        let occupied = temporary.0.join("occupied");
        fs::create_dir_all(&occupied).unwrap();
        fs::write(occupied.join("keep"), b"keep").unwrap();

        let mut staging = root.stage("model-j", "export").unwrap();
        staging.copy(&object, "weights/layer.safetensors").unwrap();
        assert!(staging.commit_standalone(&occupied).is_err());
        assert_eq!(fs::read(occupied.join("keep")).unwrap(), b"keep");

        let mut staging = root.stage("model-j", "export").unwrap();
        staging.copy(&object, "weights/layer.safetensors").unwrap();
        let destination = temporary.0.join("fresh");
        staging.commit_standalone(&destination).unwrap();
        assert!(destination.join("weights/layer.safetensors").is_file());
        let leftovers: Vec<_> = read_dir(&temporary.0)
            .unwrap()
            .into_iter()
            .filter(|path| {
                path.file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with(".fresh"))
            })
            .collect();
        assert!(leftovers.is_empty());
    }

    #[test]
    fn aborted_staging_is_collectable_and_live_staging_is_not() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let mut aborting = root.stage("model-k", "aborted").unwrap();
        aborting.write("model.safetensors", b"primary").unwrap();
        let aborted = aborting.directory().to_owned();
        aborting.abort();
        assert!(!aborted.exists());

        let mut live = root.stage("model-k", "live").unwrap();
        live.write("model.safetensors", b"primary").unwrap();
        let live_directory = live.directory().to_owned();
        let collected = root.collect().unwrap();
        assert_eq!(collected.staging, 0);
        assert!(live_directory.is_dir());
        live.abort();
        assert!(!live_directory.exists());
    }

    #[test]
    #[cfg(unix)]
    fn commit_seals_every_regular_file_including_the_primary() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let object = install(&root, b"sealed payload");
        let mut staging = root.stage("model-l", "sealed").unwrap();
        staging.write("model.safetensors", b"primary").unwrap();
        staging.link(&object, "weights/layer.safetensors").unwrap();
        let entry = staging.commit().unwrap();
        for relative in ["model.safetensors", "weights/layer.safetensors"] {
            let metadata = fs::metadata(entry.directory().join(relative)).unwrap();
            assert!(metadata.permissions().readonly(), "{relative} is writable");
        }
    }

    #[test]
    #[cfg(unix)]
    fn commit_rejects_symlinked_staging_entries() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let mut staging = root.stage("model-m", "linked").unwrap();
        staging.write("model.safetensors", b"primary").unwrap();
        std::os::unix::fs::symlink(
            "model.safetensors",
            staging.directory().join("weights.safetensors"),
        )
        .unwrap();
        assert!(staging.commit().is_err());
        assert!(read_dir(&root.path().join("models")).unwrap().is_empty());
    }

    #[test]
    #[cfg(unix)]
    fn published_entries_hardlink_one_object() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let bytes = b"embedding table".repeat(5);
        let object = install(&root, &bytes);
        for entry in ["tp1", "tp2"] {
            let mut staging = root.stage("model-a", entry).unwrap();
            staging.write("model.safetensors", b"primary").unwrap();
            staging
                .link(&object, "weights/embedding.safetensors")
                .unwrap();
            let published = staging.commit().unwrap();
            assert_eq!(published.id(), entry);
            assert_eq!(
                published
                    .payload("weights/embedding.safetensors")
                    .unwrap()
                    .len(),
                bytes.len() as u64
            );
        }
        let first = root.entry("model-a", "tp1").unwrap();
        let second = root.entry("model-a", "tp2").unwrap();
        let first_path = first.directory().join("weights/embedding.safetensors");
        let second_path = second.directory().join("weights/embedding.safetensors");
        assert_eq!(inode_of(&first_path), inode_of(&second_path));
        assert_eq!(inode_of(&first_path), inode_of(object.path()));
    }

    #[test]
    fn publication_refuses_copied_trees_but_supports_standalone_export() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let object = install(&root, b"copied payload");

        let mut staging = root.stage("model-b", "copied").unwrap();
        staging.copy(&object, "weights/copied.safetensors").unwrap();
        assert!(staging.commit().is_err());
        assert!(read_dir(&root.path().join("models")).unwrap().is_empty());

        let standalone = temporary.0.join("export");
        let mut staging = root.stage("model-b", "export").unwrap();
        staging.copy(&object, "weights/copied.safetensors").unwrap();
        staging.commit_standalone(&standalone).unwrap();
        let copied = standalone.join("weights/copied.safetensors");
        assert_eq!(
            Container::open(&copied)
                .unwrap()
                .tensor(COMPANION_TENSOR)
                .unwrap()
                .len(),
            object.length()
        );
        assert!(read_dir(&root.path().join("models")).unwrap().is_empty());

        {
            let mut staging = root.stage("model-b", "abandoned").unwrap();
            staging.write("model.safetensors", b"primary").unwrap();
        }
        assert!(
            read_dir(&root.path().join("staging/trees"))
                .unwrap()
                .is_empty()
        );
        assert!(read_dir(&root.path().join("models")).unwrap().is_empty());
    }

    #[test]
    fn leases_block_eviction_and_protect_objects() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let object = install(&root, b"leased weights");
        for entry in ["kept", "other"] {
            let mut staging = root.stage("model-c", entry).unwrap();
            staging.link(&object, "weights/layer.safetensors").unwrap();
            staging.commit().unwrap();
        }
        root.publish("model-c", "kept").unwrap();
        let leased = root.resolve("model-c").unwrap().unwrap();
        assert_eq!(leased.id(), "kept");
        assert!(root.evict("model-c", "kept").is_err());
        assert_eq!(root.collect().unwrap().objects, 0);
        assert!(object.path().exists());

        drop(leased);
        root.evict("model-c", "kept").unwrap();
        root.evict("model-c", "other").unwrap();
        // The object handle itself keeps the blob reachable until it is dropped.
        assert_eq!(root.collect().unwrap().objects, 0);
        let path = object.path().to_owned();
        drop(object);
        assert_eq!(root.collect().unwrap().objects, 1);
        assert!(!path.exists());
        assert!(root.resolve("model-c").unwrap().is_none());
    }

    #[test]
    fn ref_switch_keeps_old_entry_loadable() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let first = install(&root, b"first selection");
        let second = install(&root, b"second selection");

        let mut staging = root.stage("model-d", "one").unwrap();
        staging.link(&first, "weights/layer.safetensors").unwrap();
        staging.commit().unwrap();
        root.publish("model-d", "one").unwrap();
        let held = root.resolve("model-d").unwrap().unwrap();

        let mut staging = root.stage("model-d", "two").unwrap();
        staging.link(&second, "weights/layer.safetensors").unwrap();
        staging.commit().unwrap();
        root.publish("model-d", "two").unwrap();

        assert_eq!(root.resolve("model-d").unwrap().unwrap().id(), "two");
        assert_eq!(
            held.payload("weights/layer.safetensors").unwrap().len(),
            b"first selection".len() as u64
        );
        assert!(root.evict("model-d", "one").is_err());
        drop(held);
        root.evict("model-d", "one").unwrap();
        assert_eq!(root.ref_entry("model-d").unwrap().as_deref(), Some("two"));
        assert!(first.path().exists());
    }

    #[test]
    fn corrupt_or_missing_objects_fail_resolution() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let object = install(&root, b"payload to damage");
        let mut staging = root.stage("model-e", "only").unwrap();
        staging.link(&object, "weights/layer.safetensors").unwrap();
        let entry = staging.commit().unwrap();
        assert!(entry.payload("weights/layer.safetensors").is_ok());

        let path = entry.directory().join("weights/layer.safetensors");
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_readonly(false);
        fs::set_permissions(&path, permissions).unwrap();
        let length = fs::metadata(&path).unwrap().len();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(length - 1)
            .unwrap();
        assert!(entry.payload("weights/layer.safetensors").is_err());

        fs::remove_file(&path).unwrap();
        assert!(entry.payload("weights/layer.safetensors").is_err());
        assert!(entry.open("weights/missing.safetensors").is_err());
        assert!(entry.open("../escape").is_err());
        assert!(entry.open("/absolute").is_err());
    }

    #[test]
    #[cfg(unix)]
    fn symlinked_companions_are_rejected() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let object = install(&root, b"symlink target");
        let mut staging = root.stage("model-f", "links").unwrap();
        staging.link(&object, "weights/real.safetensors").unwrap();
        let entry = staging.commit().unwrap();

        // Immutable entries resolve only real files: every symlink component,
        // internal or escaping, is rejected rather than followed.
        let alias = entry.directory().join("weights/alias.safetensors");
        std::os::unix::fs::symlink("real.safetensors", &alias).unwrap();
        assert!(entry.payload("weights/alias.safetensors").is_err());
        assert!(entry.open("weights/alias.safetensors").is_err());

        let escape = entry.directory().join("weights/escape.safetensors");
        std::os::unix::fs::symlink("/etc/hostname", &escape).unwrap();
        assert!(entry.payload("weights/escape.safetensors").is_err());

        let linked = entry.directory().join("weights/linked.safetensors");
        fs::hard_link(entry.directory().join("weights/real.safetensors"), &linked).unwrap();
        assert_eq!(
            entry.payload("weights/linked.safetensors").unwrap().len(),
            object.length()
        );
    }

    #[test]
    fn collect_reclaims_abandoned_staging_and_unreferenced_objects() {
        let temporary = Temporary::new();
        let root = temporary.root();

        let mut live = root.stage("model-g", "live").unwrap();
        live.write("model.safetensors", b"primary").unwrap();
        let live_directory = live.directory().to_owned();
        let staged_object = install(&root, b"linked into live staging");
        live.link(&staged_object, "weights/staged.safetensors")
            .unwrap();

        let abandoned = root.path().join("staging/trees/abandoned-writer");
        fs::create_dir_all(&abandoned).unwrap();
        fs::write(abandoned.join("model.safetensors"), b"partial").unwrap();
        let orphan = install(&root, b"never referenced");
        let retained = install(&root, b"referenced payload");
        let mut staging = root.stage("model-g", "kept").unwrap();
        staging
            .link(&retained, "weights/layer.safetensors")
            .unwrap();
        staging.commit().unwrap();

        // Reachability comes from published entries and the live staging tree.
        drop((staged_object, retained, orphan));
        let collected = root.collect().unwrap();
        assert_eq!(collected.staging, 1);
        assert!(live_directory.is_dir());
        assert!(!abandoned.exists());
        assert_eq!(collected.objects, 1);
        assert!(
            root.payload(PayloadId::of(b"linked into live staging"))
                .unwrap()
                .is_some()
        );
        assert!(
            root.payload(PayloadId::of(b"referenced payload"))
                .unwrap()
                .is_some()
        );
        assert!(
            root.payload(PayloadId::of(b"never referenced"))
                .unwrap()
                .is_none()
        );

        live.abort();
        assert!(!live_directory.exists());
        assert_eq!(root.collect().unwrap().objects, 1);
        assert!(
            root.payload(PayloadId::of(b"linked into live staging"))
                .unwrap()
                .is_none()
        );
        assert!(
            root.payload(PayloadId::of(b"referenced payload"))
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn collect_preserves_legacy_hub_transport_staging() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let transport = root.path().join("staging/trees/transfer-writer");
        let blobs = transport.join("models--org--name/blobs");
        fs::create_dir_all(&blobs).unwrap();
        // A colon is not a legal regular-file name on Windows, so the fixtures use the
        // hyphen form; no production code emits either prefix.
        fs::write(blobs.join("sha256-abc"), b"complete").unwrap();
        fs::write(blobs.join("sha256-abc.part"), b"partial").unwrap();
        let export = root.path().join("staging/trees/export-writer");
        fs::create_dir_all(&export).unwrap();
        fs::write(export.join("model.safetensors"), b"partial").unwrap();

        let collected = root.collect().unwrap();
        assert_eq!(collected.preserved_transport, 1);
        assert_eq!(collected.staging, 1);
        assert!(transport.is_dir());
        assert!(!export.exists());
    }

    #[test]
    fn collect_keeps_objects_linked_from_preserved_transport() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let object = install(&root, b"linked from preserved transport");
        let tree = root.path().join("staging/trees/transfer-writer");
        let blobs = tree.join("models--org--name/blobs");
        fs::create_dir_all(&blobs).unwrap();
        fs::hard_link(
            object.path(),
            blobs.join("sha256-0123456789abcdef"),
        )
        .unwrap();
        drop(object);

        let collected = root.collect().unwrap();
        assert_eq!(collected.preserved_transport, 1);
        assert_eq!(collected.objects, 0);
        assert!(tree.is_dir());
        assert!(
            root.payload(PayloadId::of(b"linked from preserved transport"))
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn capture_hit_rejects_corruption_and_length_changes() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let bytes = b"captured payload".repeat(8);
        let object = root
            .capture_payload(bytes.len() as u64, bytes.as_slice())
            .unwrap();
        assert_eq!(object.id(), PayloadId::of(&bytes));
        let path = object.path().to_owned();
        drop(object);

        // A repeated capture returns the verified object.
        let again = root
            .capture_payload(bytes.len() as u64, bytes.as_slice())
            .unwrap();
        assert_eq!(again.id(), PayloadId::of(&bytes));
        drop(again);

        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_readonly(false);
        fs::set_permissions(&path, permissions).unwrap();
        let mut wrapper = fs::read(&path).unwrap();
        let last = wrapper.len() - 1;
        wrapper[last] ^= 0xff;
        fs::write(&path, &wrapper).unwrap();
        assert!(
            root.capture_payload(bytes.len() as u64, bytes.as_slice())
                .is_err()
        );
        assert!(
            root.insert_payload(PayloadId::of(&bytes), bytes.len() as u64, bytes.as_slice())
                .is_err()
        );
    }

    #[test]
    fn repeated_legacy_insertion_detects_corruption() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let container = b"legacy container bytes".repeat(4);
        let digest = PayloadId::of(&container);
        let first = root
            .insert_legacy_container(digest, container.as_slice())
            .unwrap();
        let path = first.path().to_owned();
        drop(first);
        root.insert_legacy_container(digest, container.as_slice())
            .unwrap();

        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_readonly(false);
        fs::set_permissions(&path, permissions).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        bytes[0] ^= 0xff;
        fs::write(&path, &bytes).unwrap();

        assert!(
            root.insert_legacy_container(digest, container.as_slice())
                .is_err()
        );
        assert!(root.legacy_container(digest).unwrap().is_some());
    }

    #[test]
    fn concurrent_inserts_return_while_first_handle_is_retained() {
        let temporary = Temporary::new();
        let root = Arc::new(temporary.root());
        let bytes = b"retained handle payload".repeat(16);
        let barrier = Arc::new(Barrier::new(2));
        let first = {
            let root = Arc::clone(&root);
            let barrier = Arc::clone(&barrier);
            let bytes = bytes.clone();
            std::thread::spawn(move || {
                barrier.wait();
                install(&root, &bytes)
            })
        };
        let second = std::thread::spawn({
            let root = Arc::clone(&root);
            let barrier = Arc::clone(&barrier);
            let bytes = bytes.clone();
            move || {
                barrier.wait();
                install(&root, &bytes)
            }
        });
        let first = first.join().unwrap();
        let second = second.join().unwrap();
        assert_eq!(first.id(), second.id());
        assert_eq!(first.path(), second.path());
        assert_eq!(objects(&root), 1);
        assert_eq!(staged(&root), 0);
        // Both handles stay live and both still read their payload.
        assert_eq!(first.length(), bytes.len() as u64);
        assert_eq!(second.length(), bytes.len() as u64);
    }

    #[test]
    fn source_longer_than_declared_length_is_rejected() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let bytes = b"exact payload";
        assert!(
            root.insert_payload(PayloadId::of(bytes), bytes.len() as u64 - 1, &bytes[..])
                .is_err()
        );
        assert!(
            root.capture_payload(bytes.len() as u64 - 1, &bytes[..])
                .is_err()
        );
        assert_eq!(objects(&root), 0);
        assert_eq!(staged(&root), 0);
    }

    #[test]
    fn exported_entries_never_replace_existing_output() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let object = install(&root, b"exported payload");
        let mut staging = root.stage("model-n", "only").unwrap();
        staging.link(&object, "weights/layer.safetensors").unwrap();
        let entry = staging.commit().unwrap();

        let existing = temporary.0.join("existing");
        fs::create_dir_all(&existing).unwrap();
        fs::write(existing.join("keep"), b"keep").unwrap();
        assert!(root.export_entry(&entry, &existing).is_err());
        assert_eq!(fs::read(existing.join("keep")).unwrap(), b"keep");

        // An empty directory created concurrently is not replaced either.
        let empty = temporary.0.join("empty");
        fs::create_dir_all(&empty).unwrap();
        assert!(root.export_entry(&entry, &empty).is_err());
        assert!(empty.is_dir());

        // A single-component relative destination with an empty parent works.
        let previous = std::env::current_dir().unwrap();
        std::env::set_current_dir(&temporary.0).unwrap();
        let result = root.export_entry(&entry, Path::new("standalone"));
        std::env::set_current_dir(&previous).unwrap();
        result.unwrap();
        assert!(
            temporary
                .0
                .join("standalone/weights/layer.safetensors")
                .is_file()
        );
    }

    #[test]
    fn model_keys_are_lossless_and_publishable_through_the_store() {
        assert_eq!(model_key("org/model").unwrap(), "org_2Fmodel");
        assert_eq!(model_key("org_2Fmodel").unwrap(), "org_5F2Fmodel");
        assert_eq!(model_key("org%2Fmodel").unwrap(), "org_252Fmodel");
        assert_eq!(model_key("qwen3.8-flash").unwrap(), "qwen3.8-flash");
        assert_ne!(model_key("a/b").unwrap(), model_key("a--b").unwrap());
        assert_ne!(model_key("a/b").unwrap(), model_key("a_2Fb").unwrap());
        assert!(model_key("").is_err());
        assert!(model_key(".").is_err());
        assert!(model_key("..").is_err());
        assert!(model_key(&"x".repeat(200)).is_err());

        // Real publication through the store, not just string formatting.
        let temporary = Temporary::new();
        let root = temporary.root();
        let object = install(&root, b"namespaced payload");
        for identity in ["org/model", "org%2Fmodel", "org_2Fmodel"] {
            let key = model_key(identity).unwrap();
            let mut staging = root.stage(&key, "entry").unwrap();
            staging.link(&object, "weights/layer.safetensors").unwrap();
            let entry = staging.commit().unwrap();
            root.publish_entry(&entry).unwrap();
            assert_eq!(
                root.resolve(&key).unwrap().unwrap().id(),
                "entry",
                "identity {identity}"
            );
        }
    }

    #[test]
    #[cfg(unix)]
    fn store_slice_streams_multi_block_payloads_and_holds_its_guard() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let bytes: Vec<u8> = (0..3 * 1024 * 1024 + 123)
            .map(|i| (i % 251) as u8)
            .collect();
        let raw = temporary.0.join("slice.bin");
        fs::write(&raw, &bytes).unwrap();
        let slice = Slice::new(Arc::new(File::open(&raw).unwrap()), 0, bytes.len() as u64).unwrap();

        let object = root.store_slice(None, &slice).unwrap();
        assert_eq!(object.id(), PayloadId::of(&bytes));
        assert_eq!(object.length(), bytes.len() as u64);
        assert!(root.payload(object.id()).unwrap().is_some());
        assert!(
            root.store_slice(Some(PayloadId::of(b"other")), &slice)
                .is_err()
        );
        // A reused slice returns the same verified object without a new write.
        let again = root.store_slice(None, &slice).unwrap();
        assert_eq!(again.path(), object.path());

        // The object guard keeps collection away while the handle is held.
        let held = root.collect().unwrap();
        assert_eq!(held.objects, 0);
        drop(object);
        drop(again);
        assert_eq!(root.collect().unwrap().objects, 1);
    }

    #[test]
    #[cfg(unix)]
    fn legacy_import_streams_large_companions_and_publishes_once() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let source = temporary.0.join("legacy");
        fs::create_dir_all(&source).unwrap();
        let companion: Vec<u8> = (0..2 * 1024 * 1024 + 7).map(|i| (i % 239) as u8).collect();
        let companion_name = "a".repeat(64) + ".safetensors";
        fs::write(source.join(&companion_name), &companion).unwrap();
        // A legacy snapshot primary whose layout names the companion: the importer reads the
        // layout, not a fabricated sidecar index.
        let layout = format!(
            "{{\"version\":1,\"files\":{{\"\":[[{}, \"{companion_name}\"]]}}}}",
            companion.len()
        );
        let header = format!(
            "{{\"__metadata__\":{{\"format\":\"flare-snapshot\"}},\
             \"flare.payload\":{{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[0,0]}},\
             \"flare.metadata\":{{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}},\
             \"flare.layout\":{{\"dtype\":\"U8\",\"shape\":[{}],\"data_offsets\":[4,{}]}}}}",
            layout.len(),
            4 + layout.len()
        );
        let mut header = header.into_bytes();
        // The legacy envelope reader requires the header to end on a 4 KiB boundary.
        while (8 + header.len()) % 4096 != 0 {
            header.push(b' ');
        }
        let mut primary = (header.len() as u64).to_le_bytes().to_vec();
        primary.extend_from_slice(&header);
        primary.extend_from_slice(&[0u8; 4]);
        primary.extend_from_slice(layout.as_bytes());
        fs::write(source.join("model.safetensors"), &primary).unwrap();

        let entry = root
            .import_legacy(&source.join("model.safetensors"), "legacy-model", "entry")
            .unwrap();
        assert_eq!(
            entry
                .open(&companion_name)
                .unwrap()
                .metadata()
                .unwrap()
                .len(),
            companion.len() as u64
        );
        assert_eq!(
            fs::read(entry.directory().join("model.safetensors")).unwrap(),
            primary
        );
        assert_eq!(root.resolve("legacy-model").unwrap().unwrap().id(), "entry");
        drop(entry);
        // Both files are content-addressed legacy objects and stay referenced.
        let collected = root.collect().unwrap();
        assert_eq!(collected.objects, 0);
    }

    #[test]
    fn legacy_containers_use_the_container_digest_namespace() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let container = b"legacy primary bytes".to_vec();
        let digest = PayloadId::of(&container);
        let object = root
            .insert_legacy_container(digest, &container[..])
            .unwrap();
        assert_eq!(
            object.path(),
            root.path().join("blobs/legacy").join(digest.file_name())
        );
        assert_eq!(object.length(), container.len() as u64);
        assert!(root.payload(digest).unwrap().is_none());
        assert_eq!(
            root.legacy_container(digest).unwrap().unwrap().length(),
            container.len() as u64
        );
        assert!(
            root.insert_legacy_container(PayloadId::of(b"different"), &container[..])
                .is_err()
        );
        drop(object);
        let collected = root.collect().unwrap();
        assert_eq!(collected.objects, 1);
        assert_eq!(collected.legacy, 1);
    }

    #[test]
    fn names_and_entry_paths_reject_escapes() {
        let temporary = Temporary::new();
        let root = temporary.root();
        assert!(root.stage("../model", "entry").is_err());
        assert!(root.stage("model", "../entry").is_err());
        assert!(
            root.stage("model", "entry")
                .unwrap()
                .write("../x", b"y")
                .is_err()
        );
        assert!(root.publish("missing", "entry").is_err());
        assert!(root.resolve("unpublished").unwrap().is_none());
        assert!(root.entry("unpublished", "entry").is_err());
    }

    fn publish_fixture(
        directory: &std::path::Path,
        payloads: &[(&str, &[u8])],
        initializers: &[(PayloadId, u64)],
    ) -> crate::runtime_package::Publisher {
        use crate::package::export::{
            self, Destination, PublishDeployment, PublishPayload, PublishRank, PublishRequest,
            PublishResource,
        };
        use crate::package::schema::{wire as pwire, FORMAT_VERSION};
        use crate::runtime_package::wire as runtime;
        use crate::runtime_package::Publisher;
        use ring::signature::{Ed25519KeyPair, KeyPair};

        fs::create_dir_all(directory).unwrap();
        let graph = directory.join("graph.bin");
        fs::write(&graph, b"graph-bytes").unwrap();
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
        let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key: key.public_key().as_ref().try_into().unwrap(),
            development: false,
        };
        let mut published = Vec::new();
        for (name, bytes) in payloads {
            let path = directory.join(format!("payload-{name}"));
            fs::write(&path, bytes).unwrap();
            published.push(PublishPayload {
                role: pwire::PayloadRole::Weights,
                relative: format!("weights/{name}.safetensors"),
                source: export::pin_file(&path, bytes.len() as u64).unwrap(),
                id: PayloadId::of(bytes),
                tensor: None,
            });
        }
        let rank = PublishRank {
            body: pwire::RankBody {
                rank: 0,
                deployment_id: "sm120-tp1".into(),
                context: Some(runtime::Context::default()),
                allocations: vec![pwire::Allocation {
                    name: "weights".into(),
                    nbytes: 32,
                    alignment: 4096,
                    address: Some(0x1000_0000),
                    memory: runtime::Memory::Device as i32,
                    ..Default::default()
                }],
                initializers: initializers
                    .iter()
                    .map(|(id, offset)| pwire::Initializer {
                        allocation: "weights".into(),
                        destination_offset: *offset,
                        source: Some(pwire::PayloadUse {
                            slice: Some(pwire::PayloadSlice {
                                payload_id: id.as_bytes().to_vec(),
                                offset: 0,
                                nbytes: 8,
                            }),
                            role: pwire::PayloadRole::Weights as i32,
                        }),
                    })
                    .collect(),
                graphs: vec![pwire::Graph {
                    name: "decode".into(),
                    resource: 0,
                    node_count: 1,
                    ..Default::default()
                }],
                ..Default::default()
            },
            payloads: published,
            resources: vec![PublishResource {
                kind: pwire::ExecutableKind::Graph,
                path: graph,
                nbytes: 11,
            }],
        };
        let request = PublishRequest {
            package: pwire::PackageBody {
                format_version: FORMAT_VERSION,
                package_id: vec![3; 32],
                model_identity: "fixture".into(),
                hardware_identity: "sm120".into(),
                publisher: Some(pwire::PublisherMetadata {
                    namespace: publisher.namespace.clone(),
                    official_source: "https://huggingface.co/example/model".into(),
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
                body: pwire::DeploymentBody {
                    deployment_id: "sm120-tp1".into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 1,
                    contract: Some(Default::default()),
                    ..Default::default()
                },
                ranks: vec![rank],
                payloads: vec![],
            }],
            package_id: [3; 32],
            public_key: publisher.public_key,
            accepted: publisher.namespace.clone(),
            no_weights: false,
        };
        export::publish(&request, &key, Destination::Directory(directory)).unwrap();
        publisher
    }

    /// Windows refuses to replace a read-only destination, and a published package is
    /// sealed; clear the attribute before the rename.
    fn replace_file(temporary: &std::path::Path, path: &std::path::Path) {
        #[cfg(windows)]
        if path.exists() {
            let mut permissions = fs::metadata(path).unwrap().permissions();
            if permissions.readonly() {
                permissions.set_readonly(false);
                fs::set_permissions(path, permissions).unwrap();
            }
        }
        fs::rename(temporary, path).unwrap();
    }

    fn rewrite_primary(
        directory: &std::path::Path,
        locations: Vec<wire::PayloadLocation>,
        derivations: Option<wire::DerivationSet>,
        drop: &[PayloadId],
    ) {
        use prost::Message;
        let metadata = reader::Metadata::open(directory).unwrap();
        let manifest = metadata.primary().tensor(reader::MANIFEST).unwrap();
        let path = directory.join("model.safetensors");
        let temporary = directory.join("primary.rebuilt");
        let mut output = File::create(&temporary).unwrap();
        let mut writer =
            crate::package::container::Writer::new(&mut output, crate::package::container::PAYLOAD_FORMAT)
                .unwrap();
        writer
            .add(
                reader::MANIFEST,
                manifest.len(),
                crate::package::container::Source::Slice(manifest),
            )
            .unwrap();
        let replaced: BTreeSet<Vec<u8>> = locations
            .iter()
            .map(|location| location.payload_id.clone())
            .collect();
        let dropped: BTreeSet<Vec<u8>> = drop
            .iter()
            .map(|id| id.as_bytes().to_vec())
            .collect();
        let mut preserved: Vec<wire::PayloadLocation> = metadata
            .locations
            .locations
            .iter()
            .filter(|location| {
                !replaced.contains(&location.payload_id)
                    && !dropped.contains(&location.payload_id)
            })
            .cloned()
            .collect();
        preserved.extend(locations);
        let set = wire::LocationSet {
            version: metadata.locations.version,
            locations: preserved,
            available_deployments: metadata.locations.available_deployments.clone(),
        };
        let bytes = Message::encode_to_vec(&set);
        writer
            .add(
                reader::LOCATIONS,
                bytes.len() as u64,
                crate::package::container::Source::Bytes(bytes),
            )
            .unwrap();
        if let Some(derivations) = derivations {
            let bytes = Message::encode_to_vec(&derivations);
            writer
                .add(
                    reader::DERIVATIONS,
                    bytes.len() as u64,
                    crate::package::container::Source::Bytes(bytes),
                )
                .unwrap();
        }
        for name in metadata.primary().tensor_names() {
            if name == reader::MANIFEST
                || name == reader::LOCATIONS
                || name == reader::DERIVATIONS
                || name.starts_with(crate::package::container::ALIGNMENT_PREFIX)
            {
                continue;
            }
            let tensor = metadata.primary().tensor(name).unwrap();
            writer
                .add(name, tensor.len(), crate::package::container::Source::Slice(tensor))
                .unwrap();
        }
        writer.finish().unwrap();
        output.sync_all().unwrap();
        // Windows cannot rename a file that is still open.
        std::mem::drop(output);
        replace_file(&temporary, &path);
    }


    fn rewrite_primary_tensors(
        directory: &std::path::Path,
        locations: Vec<wire::PayloadLocation>,
        tensors: &[(&str, &[u8])],
    ) {
        use prost::Message;
        let metadata = reader::Metadata::open(directory).unwrap();
        let manifest = metadata.primary().tensor(reader::MANIFEST).unwrap();
        let replaced: BTreeSet<Vec<u8>> = locations
            .iter()
            .map(|location| location.payload_id.clone())
            .collect();
        let mut preserved: Vec<wire::PayloadLocation> = metadata
            .locations
            .locations
            .iter()
            .filter(|location| !replaced.contains(&location.payload_id))
            .cloned()
            .collect();
        preserved.extend(locations);
        let path = directory.join("model.safetensors");
        let temporary = directory.join("primary.packed");
        let mut output = File::create(&temporary).unwrap();
        let mut writer =
            crate::package::container::Writer::new(&mut output, crate::package::container::PAYLOAD_FORMAT)
                .unwrap();
        writer
            .add(
                reader::MANIFEST,
                manifest.len(),
                crate::package::container::Source::Slice(manifest),
            )
            .unwrap();
        let set = wire::LocationSet {
            version: metadata.locations.version,
            locations: preserved,
            available_deployments: metadata.locations.available_deployments.clone(),
        };
        let bytes = Message::encode_to_vec(&set);
        writer
            .add(
                reader::LOCATIONS,
                bytes.len() as u64,
                crate::package::container::Source::Bytes(bytes),
            )
            .unwrap();
        for (name, bytes) in tensors {
            writer
                .add(
                    name,
                    bytes.len() as u64,
                    crate::package::container::Source::Bytes(bytes.to_vec()),
                )
                .unwrap();
        }
        writer.finish().unwrap();
        output.sync_all().unwrap();
        // Windows cannot rename a file that is still open.
        drop(output);
        replace_file(&temporary, &path);
    }


    fn publish_two_deployment_fixture(
        directory: &std::path::Path,
        canonical: &[u8],
        derived: &[u8],
    ) -> crate::runtime_package::Publisher {
        use crate::package::export::{
            self, Destination, PublishDeployment, PublishPayload, PublishRank, PublishRequest,
            PublishResource,
        };
        use crate::package::schema::{wire as pwire, FORMAT_VERSION};
        use crate::runtime_package::wire as runtime;
        use crate::runtime_package::Publisher;
        use ring::signature::{Ed25519KeyPair, KeyPair};

        fs::create_dir_all(directory).unwrap();
        let graph = directory.join("graph.bin");
        fs::write(&graph, b"graph-bytes").unwrap();
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
        let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key: key.public_key().as_ref().try_into().unwrap(),
            development: false,
        };
        let canonical_path = directory.join("payload-canonical");
        fs::write(&canonical_path, canonical).unwrap();
        let derived_path = directory.join("payload-derived");
        fs::write(&derived_path, derived).unwrap();
        let payload = |relative: &str, path: &std::path::Path, bytes: &[u8]| PublishPayload {
            role: pwire::PayloadRole::Weights,
            relative: relative.into(),
            source: export::pin_file(path, bytes.len() as u64).unwrap(),
            id: PayloadId::of(bytes),
            tensor: None,
        };
        let rank = |rank: u32, deployment: &str, target: PayloadId, nbytes: u64, payloads: Vec<PublishPayload>| {
            PublishRank {
                body: pwire::RankBody {
                    rank,
                    deployment_id: deployment.into(),
                    context: Some(runtime::Context::default()),
                    allocations: vec![pwire::Allocation {
                        name: "weights".into(),
                        nbytes: 32,
                        alignment: 4096,
                        address: Some(0x1000_0000),
                        memory: runtime::Memory::Device as i32,
                        ..Default::default()
                    }],
                    initializers: vec![pwire::Initializer {
                        allocation: "weights".into(),
                        destination_offset: 0,
                        source: Some(pwire::PayloadUse {
                            slice: Some(pwire::PayloadSlice {
                                payload_id: target.as_bytes().to_vec(),
                                offset: 0,
                                nbytes,
                            }),
                            role: pwire::PayloadRole::Weights as i32,
                        }),
                    }],
                    graphs: vec![pwire::Graph {
                        name: "decode".into(),
                        resource: 0,
                        node_count: 1,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                payloads,
                resources: vec![PublishResource {
                    kind: pwire::ExecutableKind::Graph,
                    path: graph.clone(),
                    nbytes: 11,
                }],
            }
        };
        let canonical_id = PayloadId::of(canonical);
        let derived_id = PayloadId::of(derived);
        let asset_bytes = b"PREassetbytesSUF";
        let asset_offset = 0u64;
        let asset_nbytes = asset_bytes.len() as u64;
        let asset_path = directory.join("payload-asset");
        fs::write(&asset_path, asset_bytes).unwrap();
        let asset_id = PayloadId::of(asset_bytes);
        let asset_payload = PublishPayload {
            role: pwire::PayloadRole::Frontend,
            relative: String::new(),
            source: export::pin_file(&asset_path, asset_bytes.len() as u64).unwrap(),
            id: asset_id,
            tensor: Some("frontend.tokenizer".into()),
        };
        let fourth: Vec<u8> = canonical
            .chunks(4)
            .rev()
            .flat_map(|chunk| chunk.iter().copied())
            .collect();
        let fourth_path = directory.join("payload-fourth");
        fs::write(&fourth_path, &fourth).unwrap();
        let fourth_id = PayloadId::of(&fourth);
        let request = PublishRequest {
            package: pwire::PackageBody {
                format_version: FORMAT_VERSION,
                package_id: vec![9; 32],
                model_identity: "two-deployment".into(),
                hardware_identity: "sm120".into(),
                publisher: Some(pwire::PublisherMetadata {
                    namespace: publisher.namespace.clone(),
                    official_source: "https://huggingface.co/example/model".into(),
                    ..Default::default()
                }),
                frontend: Some(Default::default()),
                sampling_defaults: Some(Default::default()),
                payloads: vec![],
                assets: vec![pwire::FrontendAsset {
                    name: "tokenizer".into(),
                    payload: Some(pwire::PayloadUse {
                        slice: Some(pwire::PayloadSlice {
                            payload_id: asset_id.as_bytes().to_vec(),
                            offset: asset_offset,
                            nbytes: asset_nbytes,
                        }),
                        role: pwire::PayloadRole::Frontend as i32,
                    }),
                    sha256: protection::sha256(asset_bytes).to_vec(),
                }],
                deployments: vec![],
            },
            payloads: vec![asset_payload],
            deployments: vec![
                PublishDeployment {
                    body: pwire::DeploymentBody {
                        deployment_id: "sm120-tp1".into(),
                        architecture: vec![12, 0],
                        tensor_parallel: 1,
                        contract: Some(Default::default()),
                        ..Default::default()
                    },
                    ranks: vec![rank(
                        0,
                        "sm120-tp1",
                        canonical_id,
                        canonical.len() as u64,
                        vec![payload(
                            "weights/canonical.safetensors",
                            &canonical_path,
                            canonical,
                        )],
                    )],
                    payloads: vec![],
                },
                PublishDeployment {
                    body: pwire::DeploymentBody {
                        deployment_id: "sm120-tp2".into(),
                        architecture: vec![12, 0],
                        tensor_parallel: 2,
                        contract: Some(Default::default()),
                        ..Default::default()
                    },
                    ranks: vec![
                        rank(
                            0,
                            "sm120-tp2",
                            derived_id,
                            derived.len() as u64,
                            vec![payload(
                                "weights/derived.safetensors",
                                &derived_path,
                                derived,
                            )],
                        ),
                        rank(
                            1,
                            "sm120-tp2",
                            derived_id,
                            derived.len() as u64,
                            vec![payload(
                                "weights/derived-rank1.safetensors",
                                &derived_path,
                                derived,
                            )],
                        ),
                    ],
                    payloads: vec![],
                },
                PublishDeployment {
                    body: pwire::DeploymentBody {
                        deployment_id: "sm120-tp4".into(),
                        architecture: vec![12, 0],
                        tensor_parallel: 4,
                        contract: Some(Default::default()),
                        ..Default::default()
                    },
                    ranks: (0..4)
                        .map(|rank_index| {
                            rank(
                                rank_index,
                                "sm120-tp4",
                                fourth_id,
                                fourth.len() as u64,
                                if rank_index == 0 {
                                    vec![payload(
                                        "weights/fourth.safetensors",
                                        &fourth_path,
                                        &fourth,
                                    )]
                                } else {
                                    vec![payload(
                                        &format!("weights/fourth-rank{rank_index}.safetensors"),
                                        &fourth_path,
                                        &fourth,
                                    )]
                                },
                            )
                        })
                        .collect(),
                    payloads: vec![],
                },
            ],
            package_id: [9; 32],
            public_key: publisher.public_key,
            accepted: publisher.namespace.clone(),
            no_weights: false,
        };
        export::publish(&request, &key, Destination::Directory(directory)).unwrap();
        publisher
    }

    fn write_two_deployment_hint(directory: &std::path::Path, canonical: PayloadId, derived: PayloadId) {
        let copy = |source_offset: u64, destination_offset: u64| wire::DerivationOperation {
            operation: Some(wire::derivation_operation::Operation::Copy(
                wire::DerivationCopy {
                    source_payload_id: canonical.as_bytes().to_vec(),
                    source_offset,
                    destination_offset,
                    contiguous_bytes: 8,
                    extents: vec![],
                    source_strides: vec![],
                    destination_strides: vec![],
                },
            )),
        };
        rewrite_primary(
            directory,
            vec![],
            Some(wire::DerivationSet {
                version: derivation::DERIVATION_VERSION,
                derivations: vec![wire::Derivation {
                    target_payload_id: derived.as_bytes().to_vec(),
                    operations: vec![copy(0, 8), copy(8, 0)],
                }],
            }),
            &[],
        );
    }
    fn copy(derivation_target: PayloadId, derivation_source: PayloadId) -> derivation::Derivation {
        derivation::Derivation {
            target: derivation_target,
            length: 8,
            operations: vec![derivation::Operation::Copy(derivation::Copy {
                source: derivation_source,
                source_offset: 0,
                destination_offset: 0,
                contiguous: 8,
                outer: Vec::new(),
            })],
        }
    }



    #[test]
    fn companion_names_avoid_primary_ancestor() {
        use crate::package::container::{self, Writer};
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("primary-ancestor");
        let bytes = b"PPPPpppp";
        let id = PayloadId::of(bytes);
        let publisher = publish_fixture(&directory, &[("target", bytes)], &[(id, 0)]);
        let source = directory.join("model.safetensors.safetensors");
        let mut output = File::create(&source).unwrap();
        let mut writer = Writer::new(&mut output, container::PAYLOAD_FORMAT).unwrap();
        writer
            .add("leaf", bytes.len() as u64, container::Source::Bytes(bytes.to_vec()))
            .unwrap();
        writer.finish().unwrap();
        output.sync_all().unwrap();
        rewrite_primary(
            &directory,
            vec![wire::PayloadLocation {
                payload_id: id.as_bytes().to_vec(),
                file: "model.safetensors.safetensors".into(),
                tensor: "leaf".into(),
            }],
            None,
            &[],
        );
        let selection = protection::Selection::Deployment { deployment: 0, rank: None };
        let package = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        let entry = root.install_package(package, "primary-ancestor", "entry").unwrap();
        let installed = reader::Metadata::open(entry.directory()).unwrap();
        let location = installed
            .locations
            .locations
            .iter()
            .find(|location| location.payload_id == id.as_bytes().to_vec())
            .unwrap();
        assert!(location.file.starts_with("companions/"), "{}", location.file);
        let slice = entry.payload(&location.file).unwrap();
        let mut stored = vec![0; slice.len() as usize];
        slice.read_at(0, &mut stored).unwrap();
        assert_eq!(&stored, bytes);
    }

    #[test]
    fn companion_names_avoid_file_directory_ancestor() {
        use crate::package::container::{self, Writer};
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("ancestor");
        let first = b"11111111";
        let second = b"22222222";
        let first_id = PayloadId::of(first);
        let second_id = PayloadId::of(second);
        let publisher = publish_fixture(
            &directory,
            &[("first", first), ("second", second)],
            &[(first_id, 0), (second_id, 8)],
        );
        let first_file = directory.join("weights/container.safetensors");
        fs::create_dir_all(first_file.parent().unwrap()).unwrap();
        let mut output = File::create(&first_file).unwrap();
        let mut writer = Writer::new(&mut output, container::PAYLOAD_FORMAT).unwrap();
        writer
            .add("part", first.len() as u64, container::Source::Bytes(first.to_vec()))
            .unwrap();
        writer.finish().unwrap();
        output.sync_all().unwrap();
        let second_file = directory.join("weights/container/part.safetensors.safetensors");
        fs::create_dir_all(second_file.parent().unwrap()).unwrap();
        let mut output = File::create(&second_file).unwrap();
        let mut writer = Writer::new(&mut output, container::PAYLOAD_FORMAT).unwrap();
        writer
            .add("leaf", second.len() as u64, container::Source::Bytes(second.to_vec()))
            .unwrap();
        writer.finish().unwrap();
        output.sync_all().unwrap();
        rewrite_primary(
            &directory,
            vec![
                wire::PayloadLocation {
                    payload_id: first_id.as_bytes().to_vec(),
                    file: "weights/container.safetensors".into(),
                    tensor: "part".into(),
                },
                wire::PayloadLocation {
                    payload_id: second_id.as_bytes().to_vec(),
                    file: "weights/container/part.safetensors.safetensors".into(),
                    tensor: "leaf".into(),
                },
            ],
            None,
            &[],
        );
        let selection = protection::Selection::Deployment { deployment: 0, rank: None };
        let package = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        let entry = root.install_package(package, "ancestor", "entry").unwrap();
        let installed = reader::Metadata::open(entry.directory()).unwrap();
        for (id, bytes) in [(first_id, first), (second_id, second)] {
            let location = installed
                .locations
                .locations
                .iter()
                .find(|location| location.payload_id == id.as_bytes().to_vec())
                .unwrap();
            let slice = entry.payload(&location.file).unwrap();
            let mut stored = vec![0; slice.len() as usize];
            slice.read_at(0, &mut stored).unwrap();
            assert_eq!(&stored, bytes);
        }
    }




    #[test]
    fn unrelated_cached_payload_is_not_verified_by_selected_plan() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("cone-scope");
        let canonical: Vec<u8> = (0..16).collect();
        let mut derived = canonical[8..16].to_vec();
        derived.extend_from_slice(&canonical[..8]);
        let publisher = publish_two_deployment_fixture(&directory, &canonical, &derived);
        let unrelated = b"unrelated-cached-payload";
        let unrelated_id = PayloadId::of(unrelated);
        root.insert_payload(unrelated_id, unrelated.len() as u64, unrelated.as_slice())
            .unwrap();
        let selection = protection::Selection::Deployment { deployment: 0, rank: None };
        let package = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        let progress = progress::AcquisitionProgress::new();
        let _plan = root
            .plan_acquisition_with(package, &derivation::Limits::default(), Some(&progress))
            .unwrap();
        assert_eq!(progress.snapshot().verification_bytes, 0);
    }
    #[test]
    fn cached_canonical_avoids_remote_alternative_target() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("warm-forward");
        let canonical: Vec<u8> = (0..16).collect();
        let mut derived = canonical[8..16].to_vec();
        derived.extend_from_slice(&canonical[..8]);
        let canonical_id = PayloadId::of(&canonical);
        let derived_id = PayloadId::of(&derived);
        let publisher = publish_two_deployment_fixture(&directory, &canonical, &derived);
        root.insert_payload(canonical_id, canonical.len() as u64, canonical.as_slice())
            .unwrap();
        write_two_deployment_hint(&directory, canonical_id, derived_id);
        let selection = protection::Selection::Deployment { deployment: 1, rank: None };
        let package = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        let plan = root
            .plan_acquisition(package, &derivation::Limits::default())
            .unwrap();
        assert!(
            plan.remote_files()
                .iter()
                .all(|file| !file.contains("weights/derived")),
            "{:?}",
            plan.remote_files()
        );
        let entry = root.install_planned(plan, "warm-forward", "entry").unwrap();
        let installed = reader::Metadata::open(entry.directory()).unwrap();
        let location = installed
            .locations
            .locations
            .iter()
            .find(|location| location.payload_id == derived_id.as_bytes().to_vec())
            .unwrap();
        let slice = entry.payload(&location.file).unwrap();
        let mut stored = vec![0; slice.len() as usize];
        slice.read_at(0, &mut stored).unwrap();
        assert_eq!(stored, derived);
    }

    #[test]
    fn corrupt_cached_canonical_falls_back_to_remote_target() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("corrupt-fallback");
        let canonical: Vec<u8> = (0..16).collect();
        let mut derived = canonical[8..16].to_vec();
        derived.extend_from_slice(&canonical[..8]);
        let canonical_id = PayloadId::of(&canonical);
        let derived_id = PayloadId::of(&derived);
        let publisher = publish_two_deployment_fixture(&directory, &canonical, &derived);
        fs::create_dir_all(root.path().join("blobs").join(canonical_id.file_name())).unwrap();
        write_two_deployment_hint(&directory, canonical_id, derived_id);
        let selection = protection::Selection::Deployment { deployment: 1, rank: None };
        let package = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        let plan = root
            .plan_acquisition(package, &derivation::Limits::default())
            .unwrap();
        assert!(
            plan.remote_files()
                .iter()
                .any(|file| file.contains("weights/derived")),
            "{:?}",
            plan.remote_files()
        );
        let entry = root.install_planned(plan, "corrupt-fallback", "entry").unwrap();
        let installed = reader::Metadata::open(entry.directory()).unwrap();
        let location = installed
            .locations
            .locations
            .iter()
            .find(|location| location.payload_id == derived_id.as_bytes().to_vec())
            .unwrap();
        let slice = entry.payload(&location.file).unwrap();
        let mut stored = vec![0; slice.len() as usize];
        slice.read_at(0, &mut stored).unwrap();
        assert_eq!(stored, derived);
    }
    #[test]
    fn cached_alternative_deployment_recovers_selected_canonical() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("two-deployment");
        let canonical: Vec<u8> = (0..16).collect();
        let mut derived = canonical[8..16].to_vec();
        derived.extend_from_slice(&canonical[..8]);
        let canonical_id = PayloadId::of(&canonical);
        let derived_id = PayloadId::of(&derived);
        let publisher = publish_two_deployment_fixture(&directory, &canonical, &derived);
        root.insert_payload(derived_id, derived.len() as u64, derived.as_slice())
            .unwrap();
        let operations = vec![
            wire::DerivationOperation {
                operation: Some(wire::derivation_operation::Operation::Copy(
                    wire::DerivationCopy {
                        source_payload_id: canonical_id.as_bytes().to_vec(),
                        source_offset: 0,
                        destination_offset: 8,
                        contiguous_bytes: 8,
                        extents: vec![],
                        source_strides: vec![],
                        destination_strides: vec![],
                    },
                )),
            },
            wire::DerivationOperation {
                operation: Some(wire::derivation_operation::Operation::Copy(
                    wire::DerivationCopy {
                        source_payload_id: canonical_id.as_bytes().to_vec(),
                        source_offset: 8,
                        destination_offset: 0,
                        contiguous_bytes: 8,
                        extents: vec![],
                        source_strides: vec![],
                        destination_strides: vec![],
                    },
                )),
            },
        ];
        rewrite_primary(
            &directory,
            vec![],
            Some(wire::DerivationSet {
                version: derivation::DERIVATION_VERSION,
                derivations: vec![wire::Derivation {
                    target_payload_id: derived_id.as_bytes().to_vec(),
                    operations,
                }],
            }),
            &[],
        );
        let selection = protection::Selection::Deployment {
            deployment: 0,
            rank: None,
        };
        let package = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        let progress = progress::AcquisitionProgress::new();
        let mut plan = root
            .plan_acquisition_with(
                package,
                &derivation::Limits::default(),
                Some(&progress),
            )
            .unwrap();
        assert!(
            plan.remote_files()
                .iter()
                .all(|file| !file.contains("weights/canonical")),
            "{:?}",
            plan.remote_files()
        );
        // The acquired alternative-source lease must survive collection while
        // transport/relocation work is in progress.
        root.collect().unwrap();
        plan.relocate_source(&directory.join("model.safetensors")).unwrap();
        let staging = root.stage_named("two-deployment", "gb10-tp1-202610240600").unwrap();
        let entry = root.install_planned_named(plan, staging).unwrap();
        assert_eq!(entry.id(), "gb10-tp1-202610240600");
        let installed = reader::Metadata::open(entry.directory()).unwrap();
        let location = installed
            .locations
            .locations
            .iter()
            .find(|location| location.payload_id == canonical_id.as_bytes().to_vec())
            .unwrap();
        let slice = entry.payload(&location.file).unwrap();
        let mut stored = vec![0; slice.len() as usize];
        slice.read_at(0, &mut stored).unwrap();
        assert_eq!(stored, canonical);
        assert!(!installed
            .locations
            .locations
            .iter()
            .any(|location| location.payload_id == derived_id.as_bytes().to_vec()));
        let snapshot = progress.snapshot();
        assert!(snapshot.materialized_copy_bytes >= canonical.len() as u64);
        assert!(snapshot.verification_bytes > 0);
    }

    #[test]
    fn relocation_rejects_changed_embedded_asset_bytes() {
        use crate::package::container::{self, Writer};
        use prost::Message;
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("relocate-asset");
        let canonical: Vec<u8> = (0..16).collect();
        let mut derived = canonical[8..16].to_vec();
        derived.extend_from_slice(&canonical[..8]);
        let publisher = publish_two_deployment_fixture(&directory, &canonical, &derived);
        let selection = protection::Selection::Deployment { deployment: 0, rank: None };
        let package = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        let progress = progress::AcquisitionProgress::new();
        let mut plan = root
            .plan_acquisition_with(package, &derivation::Limits::default(), Some(&progress))
            .unwrap();
        let transport = temporary.0.join("transport");
        fs::create_dir_all(&transport).unwrap();
        let metadata = reader::Metadata::open(&directory).unwrap();
        let asset_len = metadata
            .primary()
            .tensor("frontend.tokenizer")
            .unwrap()
            .len();
        let mut output = File::create(transport.join("model.safetensors")).unwrap();
        let mut writer = Writer::new(&mut output, container::PAYLOAD_FORMAT).unwrap();
        let manifest = metadata.primary().tensor(reader::MANIFEST).unwrap();
        writer
            .add(reader::MANIFEST, manifest.len(), container::Source::Slice(manifest))
            .unwrap();
        let locations = Message::encode_to_vec(&metadata.locations);
        writer
            .add(reader::LOCATIONS, locations.len() as u64, container::Source::Bytes(locations))
            .unwrap();
        if let Some(derivations) = &metadata.derivations {
            let bytes = Message::encode_to_vec(derivations);
            writer
                .add(reader::DERIVATIONS, bytes.len() as u64, container::Source::Bytes(bytes))
                .unwrap();
        }
        for name in metadata.primary().tensor_names() {
            if name == reader::MANIFEST
                || name == reader::LOCATIONS
                || name == reader::DERIVATIONS
                || name.starts_with(container::ALIGNMENT_PREFIX)
            {
                continue;
            }
            let tensor = metadata.primary().tensor(name).unwrap();
            if name == "frontend.tokenizer" {
                let mut bytes = vec![0; tensor.len() as usize];
                tensor.read_at(0, &mut bytes).unwrap();
                bytes.reverse();
                writer
                    .add(name, bytes.len() as u64, container::Source::Bytes(bytes))
                    .unwrap();
            } else {
                writer
                    .add(name, tensor.len(), container::Source::Slice(tensor))
                    .unwrap();
            }
        }
        writer.finish().unwrap();
        output.sync_all().unwrap();
        plan.relocate_source(&directory.join("model.safetensors"))
            .expect("embedded asset payload should relocate as a whole payload");
        let error = plan
            .relocate_source(&transport.join("model.safetensors"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("embedded asset"), "{error}");
        assert_eq!(progress.snapshot().verification_bytes, 2 * asset_len);
    }

    #[test]
    fn inverse_recovers_dependency_then_forward_selected_target() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("inverse-forward");
        let canonical: Vec<u8> = (0..16).collect();
        let mut derived = canonical[8..16].to_vec();
        derived.extend_from_slice(&canonical[..8]);
        let fourth: Vec<u8> = canonical
            .chunks(4)
            .rev()
            .flat_map(|chunk| chunk.iter().copied())
            .collect();
        let canonical_id = PayloadId::of(&canonical);
        let derived_id = PayloadId::of(&derived);
        let fourth_id = PayloadId::of(&fourth);
        let publisher = publish_two_deployment_fixture(&directory, &canonical, &derived);
        root.insert_payload(derived_id, derived.len() as u64, derived.as_slice())
            .unwrap();
        let copy = |source: PayloadId, _target: PayloadId, source_offset: u64, destination_offset: u64| {
            wire::DerivationOperation {
                operation: Some(wire::derivation_operation::Operation::Copy(
                    wire::DerivationCopy {
                        source_payload_id: source.as_bytes().to_vec(),
                        source_offset,
                        destination_offset,
                        contiguous_bytes: 4,
                        extents: vec![],
                        source_strides: vec![],
                        destination_strides: vec![],
                    },
                )),
            }
        };
        let copy_bytes = |source: PayloadId, source_offset: u64, destination_offset: u64, nbytes: u64| {
            wire::DerivationOperation {
                operation: Some(wire::derivation_operation::Operation::Copy(
                    wire::DerivationCopy {
                        source_payload_id: source.as_bytes().to_vec(),
                        source_offset,
                        destination_offset,
                        contiguous_bytes: nbytes,
                        extents: vec![],
                        source_strides: vec![],
                        destination_strides: vec![],
                    },
                )),
            }
        };
        let derived_hint = wire::Derivation {
            target_payload_id: derived_id.as_bytes().to_vec(),
            operations: vec![
                copy_bytes(canonical_id, 0, 8, 8),
                copy_bytes(canonical_id, 8, 0, 8),
            ],
        };
        let fourth_hint = wire::Derivation {
            target_payload_id: fourth_id.as_bytes().to_vec(),
            operations: vec![
                copy(canonical_id, fourth_id, 0, 12),
                copy(canonical_id, fourth_id, 4, 8),
                copy(canonical_id, fourth_id, 8, 4),
                copy(canonical_id, fourth_id, 12, 0),
            ],
        };
        rewrite_primary(
            &directory,
            vec![],
            Some(wire::DerivationSet {
                version: derivation::DERIVATION_VERSION,
                derivations: vec![derived_hint, fourth_hint],
            }),
            &[],
        );
        let selection = protection::Selection::Deployment {
            deployment: 2,
            rank: None,
        };
        let package = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        let mut plan = root
            .plan_acquisition(package, &derivation::Limits::default())
            .unwrap();
        assert!(
            plan.remote_files()
                .iter()
                .all(|file| !file.contains("weights/fourth")),
            "{:?}",
            plan.remote_files()
        );
        plan.relocate_source(&directory.join("model.safetensors")).unwrap();
        let entry = root.install_planned(plan, "inverse-forward", "entry").unwrap();
        let installed = reader::Metadata::open(entry.directory()).unwrap();
        let location = installed
            .locations
            .locations
            .iter()
            .find(|location| location.payload_id == fourth_id.as_bytes().to_vec())
            .unwrap();
        let slice = entry.payload(&location.file).unwrap();
        let mut stored = vec![0; slice.len() as usize];
        slice.read_at(0, &mut stored).unwrap();
        assert_eq!(stored, fourth);
    }

    #[test]
    fn optional_cache_arrival_after_plan_does_not_count_source_read() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("optional-arrival");
        let source = b"TTTTttttSSSSssss".to_vec();
        let target = source[..8].to_vec();
        let target_id = PayloadId::of(&target);
        let source_id = PayloadId::of(&source);
        let graph_id = PayloadId::of(b"graph-bytes");
        let publisher = publish_fixture(
            &directory,
            &[("target", &target), ("source", &source)],
            &[(target_id, 0)],
        );
        root.insert_payload(graph_id, 11, b"graph-bytes".as_slice())
            .unwrap();
        rewrite_primary(
            &directory,
            vec![
                wire::PayloadLocation {
                    payload_id: target_id.as_bytes().to_vec(),
                    file: "weights/target.safetensors".into(),
                    tensor: COMPANION_TENSOR.into(),
                },
                wire::PayloadLocation {
                    payload_id: source_id.as_bytes().to_vec(),
                    file: "weights/source.safetensors".into(),
                    tensor: COMPANION_TENSOR.into(),
                },
            ],
            Some(wire::DerivationSet {
                version: derivation::DERIVATION_VERSION,
                derivations: vec![wire::Derivation {
                    target_payload_id: target_id.as_bytes().to_vec(),
                    operations: vec![wire::DerivationOperation {
                        operation: Some(wire::derivation_operation::Operation::Copy(
                            wire::DerivationCopy {
                                source_payload_id: source_id.as_bytes().to_vec(),
                                source_offset: 0,
                                destination_offset: 0,
                                contiguous_bytes: 8,
                                extents: vec![],
                                source_strides: vec![],
                                destination_strides: vec![],
                            },
                        )),
                    }],
                }],
            }),
            &[],
        );
        let selection = protection::Selection::Deployment { deployment: 0, rank: None };
        let package = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        let graph_slice = {
            let slice = package.verified()[0].ranks()[0].resources[0]
                .stored
                .as_ref()
                .unwrap();
            let graph_id = PayloadId::from_bytes(slice.payload_id.as_slice().try_into().unwrap());
            let payloads = package
                .metadata()
                .payloads(package.verified()[0].catalog(), [slice.payload_id.as_slice()])
                .unwrap();
            let source = payloads.get(&slice.payload_id).unwrap().clone();
            root.insert_payload(
                graph_id,
                source.len(),
                SliceReader { slice: source.clone(), offset: 0 },
            )
            .unwrap();
            source
        };
        let _ = graph_slice;
        let progress = progress::AcquisitionProgress::new();
        let plan = root
            .plan_acquisition_with(package, &derivation::Limits::default(), Some(&progress))
            .unwrap();
        root.insert_payload(source_id, source.len() as u64, source.as_slice())
            .unwrap();
        let before = progress.snapshot();
        let entry = root.install_planned(plan, "optional-arrival", "entry").unwrap();
        let after = progress.snapshot();
        assert_eq!(
            after.local_source_bytes - before.local_source_bytes,
            target.len() as u64
        );
        assert!(entry.payload("weights/source.safetensors").is_ok());
    }
    #[test]
    fn ready_target_retains_cache_only_source() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("cache-only");
        let source = b"TTTTttttSSSSssss".to_vec();
        let target = source[..8].to_vec();
        let target_id = PayloadId::of(&target);
        let source_id = PayloadId::of(&source);
        let publisher = publish_fixture(
            &directory,
            &[("target", &target), ("source", &source)],
            &[(target_id, 0)],
        );
        root.insert_payload(source_id, source.len() as u64, source.as_slice())
            .unwrap();
        rewrite_primary(
            &directory,
            vec![wire::PayloadLocation {
                payload_id: target_id.as_bytes().to_vec(),
                file: "weights/target.safetensors".into(),
                tensor: COMPANION_TENSOR.into(),
            }],
            Some(wire::DerivationSet {
                version: derivation::DERIVATION_VERSION,
                derivations: vec![wire::Derivation {
                    target_payload_id: target_id.as_bytes().to_vec(),
                    operations: vec![wire::DerivationOperation {
                        operation: Some(wire::derivation_operation::Operation::Copy(
                            wire::DerivationCopy {
                                source_payload_id: source_id.as_bytes().to_vec(),
                                source_offset: 0,
                                destination_offset: 0,
                                contiguous_bytes: 8,
                                extents: vec![],
                                source_strides: vec![],
                                destination_strides: vec![],
                            },
                        )),
                    }],
                }],
            }),
            &[source_id],
        );
        let selection = protection::Selection::Deployment { deployment: 0, rank: None };
        let package = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        let entry = root.install_package(package, "cache-only", "entry").unwrap();
        let installed = reader::Metadata::open(entry.directory()).unwrap();
        let location = installed
            .locations
            .locations
            .iter()
            .find(|location| location.payload_id == source_id.as_bytes().to_vec())
            .unwrap();
        assert!(location.file.starts_with("companions/"), "{}", location.file);
        root.collect().unwrap();
        assert!(root.payload(source_id).unwrap().is_some());
        assert!(root.payload(target_id).unwrap().is_some());
    }

    #[test]
    fn ready_target_retains_multi_hop_available_source() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("multi-hop");
        let target = b"TTTTtttt";
        let intermediate = b"IIIIiiii";
        let canonical = b"CCCCcccc";
        let target_id = PayloadId::of(target);
        let intermediate_id = PayloadId::of(intermediate);
        let canonical_id = PayloadId::of(canonical);
        let publisher = publish_fixture(
            &directory,
            &[
                ("target", target),
                ("intermediate", intermediate),
                ("canonical", canonical),
            ],
            &[(target_id, 0)],
        );
        let copy = |source: PayloadId, target: PayloadId| wire::Derivation {
            target_payload_id: target.as_bytes().to_vec(),
            operations: vec![wire::DerivationOperation {
                operation: Some(wire::derivation_operation::Operation::Copy(
                    wire::DerivationCopy {
                        source_payload_id: source.as_bytes().to_vec(),
                        source_offset: 0,
                        destination_offset: 0,
                        contiguous_bytes: 8,
                        extents: vec![],
                        source_strides: vec![],
                        destination_strides: vec![],
                    },
                )),
            }],
        };
        rewrite_primary(
            &directory,
            vec![
                wire::PayloadLocation {
                    payload_id: target_id.as_bytes().to_vec(),
                    file: "weights/target.safetensors".into(),
                    tensor: COMPANION_TENSOR.into(),
                },
                wire::PayloadLocation {
                    payload_id: canonical_id.as_bytes().to_vec(),
                    file: "weights/canonical.safetensors".into(),
                    tensor: COMPANION_TENSOR.into(),
                },
            ],
            Some(wire::DerivationSet {
                version: derivation::DERIVATION_VERSION,
                derivations: vec![
                    copy(intermediate_id, target_id),
                    copy(canonical_id, intermediate_id),
                ],
            }),
            &[intermediate_id],
        );
        let selection = protection::Selection::Deployment { deployment: 0, rank: None };
        let package = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        let entry = root.install_package(package, "multi-hop", "entry").unwrap();
        let installed = reader::Metadata::open(entry.directory()).unwrap();
        assert!(
            installed
                .locations
                .locations
                .iter()
                .any(|location| location.payload_id == canonical_id.as_bytes().to_vec())
        );
        root.collect().unwrap();
        assert!(root.payload(canonical_id).unwrap().is_some());
        assert!(root.payload(intermediate_id).unwrap().is_none());
    }

    #[test]
    fn invalid_optional_hints_do_not_retain_or_fail_ready_targets() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("invalid-optional");
        let target = b"AAAAaaaa";
        let target2 = b"BBBBbbbb";
        let source = b"CCCCcccc";
        let source2 = b"DDDDdddd";
        let target_id = PayloadId::of(target);
        let target2_id = PayloadId::of(target2);
        let source_id = PayloadId::of(source);
        let source2_id = PayloadId::of(source2);
        let publisher = publish_fixture(
            &directory,
            &[
                ("target", target),
                ("target2", target2),
                ("source", source),
                ("source2", source2),
            ],
            &[(target_id, 0), (target2_id, 8)],
        );
        let copy_source = |source: PayloadId| wire::derivation_operation::Operation::Copy(
            wire::DerivationCopy {
                source_payload_id: source.as_bytes().to_vec(),
                source_offset: 0,
                destination_offset: 0,
                contiguous_bytes: 4,
                extents: vec![],
                source_strides: vec![],
                destination_strides: vec![],
            },
        );
        let over = derivation::Limits::default().max_operations + 1;
        let mut huge = Vec::with_capacity(over);
        for _ in 0..over {
            huge.push(wire::DerivationOperation {
                operation: Some(copy_source(source2_id)),
            });
        }
        rewrite_primary(
            &directory,
            vec![
                wire::PayloadLocation {
                    payload_id: target_id.as_bytes().to_vec(),
                    file: "weights/target.safetensors".into(),
                    tensor: COMPANION_TENSOR.into(),
                },
                wire::PayloadLocation {
                    payload_id: target2_id.as_bytes().to_vec(),
                    file: "weights/target2.safetensors".into(),
                    tensor: COMPANION_TENSOR.into(),
                },
                wire::PayloadLocation {
                    payload_id: source_id.as_bytes().to_vec(),
                    file: "weights/source.safetensors".into(),
                    tensor: COMPANION_TENSOR.into(),
                },
                wire::PayloadLocation {
                    payload_id: source2_id.as_bytes().to_vec(),
                    file: "weights/source2.safetensors".into(),
                    tensor: COMPANION_TENSOR.into(),
                },
            ],
            Some(wire::DerivationSet {
                version: derivation::DERIVATION_VERSION,
                derivations: vec![
                    wire::Derivation {
                        target_payload_id: target_id.as_bytes().to_vec(),
                        operations: vec![wire::DerivationOperation {
                            operation: Some(copy_source(source_id)),
                        }],
                    },
                    wire::Derivation {
                        target_payload_id: target2_id.as_bytes().to_vec(),
                        operations: huge,
                    },
                ],
            }),
            &[],
        );
        let selection = protection::Selection::Deployment { deployment: 0, rank: None };
        let package = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        let entry = root.install_package(package, "invalid-optional", "entry").unwrap();
        assert!(entry.payload("weights/target.safetensors").is_ok());
        assert!(entry.payload("weights/target2.safetensors").is_ok());
        root.collect().unwrap();
        assert!(root.payload(source_id).unwrap().is_none());
        assert!(root.payload(source2_id).unwrap().is_none());
    }

    fn over_limit_hint_set() -> wire::DerivationSet {
        let count = derivation::Limits::default().max_derivations + 1;
        wire::DerivationSet {
            version: derivation::DERIVATION_VERSION,
            derivations: (0..count)
                .map(|_| wire::Derivation {
                    target_payload_id: vec![0; 32],
                    operations: Vec::new(),
                })
                .collect(),
        }
    }

    fn package_source<'a>(
        directory: &std::path::Path,
        publisher: &'a crate::runtime_package::Publisher,
    ) -> PackageSource<'a> {
        PackageSource::open(
            directory,
            std::slice::from_ref(publisher),
            protection::Selection::Deployment {
                deployment: 0,
                rank: None,
            },
        )
        .unwrap()
    }

    #[test]
    fn warm_cached_target_ignores_over_limit_hints_and_installs_default() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("over-limit-warm");
        let target = vec![b'W'; 64];
        let target_id = PayloadId::of(&target);
        let publisher = publish_fixture(
            &directory,
            &[("target", target.as_slice())],
            &[(target_id, 0)],
        );
        root.insert_payload(target_id, target.len() as u64, target.as_slice())
            .unwrap();
        root.install_package(
            package_source(&directory, &publisher),
            "warm-model",
            "warm-a",
        )
        .unwrap();
        assert_eq!(
            root.ref_entry("warm-model").unwrap().as_deref(),
            Some("warm-a")
        );
        let locations = reader::Metadata::open(&directory)
            .unwrap()
            .locations
            .locations;
        rewrite_primary(&directory, locations, Some(over_limit_hint_set()), &[]);
        let progress = progress::AcquisitionProgress::new();
        let _plan = root
            .plan_acquisition_with(
                package_source(&directory, &publisher),
                &derivation::Limits::default(),
                Some(&progress),
            )
            .unwrap();
        assert_eq!(
            root.ref_entry("warm-model").unwrap().as_deref(),
            Some("warm-a"),
            "planning must not update the default"
        );
        let entry = root
            .install_package(
                package_source(&directory, &publisher),
                "warm-model",
                "warm-b",
            )
            .unwrap();
        assert_eq!(
            root.ref_entry("warm-model").unwrap().as_deref(),
            Some(entry.id()),
            "successful warm installation must update the default"
        );
        assert!(
            progress.snapshot().verification_bytes >= target.len() as u64,
            "the cached required target must still be hashed"
        );
    }

    #[test]
    fn cold_missing_target_with_over_limit_hints_fails_without_default_mutation() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let prior_directory = temporary.0.join("over-limit-prior");
        let prior_target = b"PPPPpppp";
        let prior_id = PayloadId::of(prior_target);
        let prior_publisher = publish_fixture(
            &prior_directory,
            &[("target", prior_target)],
            &[(prior_id, 0)],
        );
        root.install_package(
            package_source(&prior_directory, &prior_publisher),
            "cold-model",
            "prior",
        )
        .unwrap();
        assert_eq!(
            root.ref_entry("cold-model").unwrap().as_deref(),
            Some("prior")
        );

        let directory = temporary.0.join("over-limit-cold");
        let target = b"CCCCcccc";
        let target_id = PayloadId::of(target);
        let publisher = publish_fixture(&directory, &[("target", target)], &[(target_id, 0)]);
        rewrite_primary(
            &directory,
            Vec::new(),
            Some(over_limit_hint_set()),
            &[target_id],
        );
        let error = match root.install_package(
            package_source(&directory, &publisher),
            "cold-model",
            "cold",
        ) {
            Ok(_) => panic!("an underivable cold target must fail"),
            Err(error) => error,
        };
        assert!(!error.to_string().is_empty());
        assert_eq!(
            root.ref_entry("cold-model").unwrap().as_deref(),
            Some("prior"),
            "a failed cold acquisition must leave the default unchanged"
        );
    }

    #[test]
    fn direct_located_target_with_over_limit_hints_uses_network_path() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("over-limit-located");
        let target = b"LLLLllll";
        let target_id = PayloadId::of(target);
        let publisher = publish_fixture(&directory, &[("target", target)], &[(target_id, 0)]);
        rewrite_primary(
            &directory,
            reader::Metadata::open(&directory)
                .unwrap()
                .locations
                .locations,
            Some(over_limit_hint_set()),
            &[],
        );
        let plan = root
            .plan_acquisition(
                package_source(&directory, &publisher),
                &derivation::Limits::default(),
            )
            .unwrap();
        assert!(
            plan.remote_files()
                .iter()
                .any(|file| file == "weights/target.safetensors"),
            "the direct located target must remain fetchable: {:?}",
            plan.remote_files()
        );
        let entry = root
            .install_planned(plan, "over-limit-located", "entry")
            .unwrap();
        let slice = entry.payload("weights/target.safetensors").unwrap();
        let mut stored = vec![0; slice.len() as usize];
        slice.read_at(0, &mut stored).unwrap();
        assert_eq!(stored, target);
    }

    #[test]
    fn corrupt_cached_required_target_is_not_reused_with_over_limit_hints() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("over-limit-corrupt");
        let target = b"RRRRrrrr";
        let target_id = PayloadId::of(target);
        let publisher = publish_fixture(&directory, &[("target", target)], &[(target_id, 0)]);
        // A directory at the immutable object path makes the cached hit
        // unreadable and must not satisfy the required target.
        fs::create_dir_all(root.path().join("blobs").join(target_id.file_name())).unwrap();
        rewrite_primary(
            &directory,
            reader::Metadata::open(&directory)
                .unwrap()
                .locations
                .locations,
            Some(over_limit_hint_set()),
            &[],
        );
        let plan = root
            .plan_acquisition(
                package_source(&directory, &publisher),
                &derivation::Limits::default(),
            )
            .unwrap();
        assert!(
            plan.remote_files()
                .iter()
                .any(|file| file == "weights/target.safetensors"),
            "a corrupt cached required target must fall back to its located source"
        );
        // Remove the deliberately corrupt object so the fallback source can be
        // installed under its immutable identity.
        fs::remove_dir_all(root.path().join("blobs").join(target_id.file_name())).unwrap();
        let entry = root
            .install_planned(plan, "over-limit-corrupt", "entry")
            .unwrap();
        let slice = entry.payload("weights/target.safetensors").unwrap();
        let mut stored = vec![0; slice.len() as usize];
        slice.read_at(0, &mut stored).unwrap();
        assert_eq!(stored, target);
    }

    #[test]
    fn optional_publication_failure_is_reported() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("optional-io");
        let target = b"TTTTtttt";
        let source = b"SSSSssss";
        let target_id = PayloadId::of(target);
        let source_id = PayloadId::of(source);
        let publisher = publish_fixture(
            &directory,
            &[("target", target), ("source", source)],
            &[(target_id, 0)],
        );
        rewrite_primary(
            &directory,
            vec![
                wire::PayloadLocation {
                    payload_id: target_id.as_bytes().to_vec(),
                    file: "weights/target.safetensors".into(),
                    tensor: COMPANION_TENSOR.into(),
                },
                wire::PayloadLocation {
                    payload_id: source_id.as_bytes().to_vec(),
                    file: "weights/source.safetensors".into(),
                    tensor: COMPANION_TENSOR.into(),
                },
            ],
            Some(wire::DerivationSet {
                version: derivation::DERIVATION_VERSION,
                derivations: vec![wire::Derivation {
                    target_payload_id: target_id.as_bytes().to_vec(),
                    operations: vec![wire::DerivationOperation {
                        operation: Some(wire::derivation_operation::Operation::Copy(
                            wire::DerivationCopy {
                                source_payload_id: source_id.as_bytes().to_vec(),
                                source_offset: 0,
                                destination_offset: 0,
                                contiguous_bytes: 8,
                                extents: vec![],
                                source_strides: vec![],
                                destination_strides: vec![],
                            },
                        )),
                    }],
                }],
            }),
            &[],
        );
        let prior = root.stage_named("optional-io", "prior").unwrap().commit().unwrap();
        root.publish_entry(&prior).unwrap();
        fs::create_dir_all(root.path().join("blobs").join(source_id.file_name())).unwrap();
        let selection = protection::Selection::Deployment { deployment: 0, rank: None };
        let package = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        let error = match root.install_package(package, "optional-io", "entry") {
            Ok(_) => panic!("optional publication failure must not be swallowed"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("collides with a directory"), "{error}");
        assert_eq!(
            root.ref_entry(&model_key("optional-io").unwrap()).unwrap().as_deref(),
            Some(prior.id())
        );
    }

    #[test]
    fn path_conflicts_find_descendants_before_punctuation_siblings() {
        let used = BTreeSet::from([
            "a-b.safetensors".to_owned(),
            "a/child.safetensors".to_owned(),
        ]);
        assert!(path_conflicts("a", &used));
        let used = BTreeSet::from([
            "a.safetensors-x".to_owned(),
            "a.safetensors/child.safetensors".to_owned(),
        ]);
        assert!(path_conflicts("a.safetensors", &used));
    }

    #[test]
    fn arbitrary_packed_tensors_normalize_to_friendly_companions() {
        let identity = PayloadId::of(b"arbitrary packed payload");
        assert_eq!(
            normalized_payload_companion("", "part0", identity),
            "packed/part0.safetensors"
        );
        assert_eq!(
            normalized_payload_companion("", "weights.part0", identity),
            "weights/part0.safetensors"
        );
        assert_eq!(
            normalized_payload_companion("", "../unsafe name", identity),
            "packed/_unsafe_name.safetensors"
        );
    }

    #[test]
    fn optional_repeated_sources_open_once_per_unique_id() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("repeated-source");
        let target = b"TTTTTTTT";
        let source = b"SSSSSSSS";
        let target_id = PayloadId::of(target);
        let source_id = PayloadId::of(source);
        let publisher = publish_fixture(
            &directory,
            &[("target", target), ("source", source)],
            &[(target_id, 0)],
        );
        let operations = (0..8u64)
            .map(|offset| wire::DerivationOperation {
                operation: Some(wire::derivation_operation::Operation::Copy(
                    wire::DerivationCopy {
                        source_payload_id: source_id.as_bytes().to_vec(),
                        source_offset: 0,
                        destination_offset: offset,
                        contiguous_bytes: 1,
                        extents: vec![],
                        source_strides: vec![],
                        destination_strides: vec![],
                    },
                )),
            })
            .collect();
        rewrite_primary(
            &directory,
            vec![
                wire::PayloadLocation {
                    payload_id: target_id.as_bytes().to_vec(),
                    file: "weights/target.safetensors".into(),
                    tensor: COMPANION_TENSOR.into(),
                },
                wire::PayloadLocation {
                    payload_id: source_id.as_bytes().to_vec(),
                    file: "weights/source.safetensors".into(),
                    tensor: COMPANION_TENSOR.into(),
                },
            ],
            Some(wire::DerivationSet {
                version: derivation::DERIVATION_VERSION,
                derivations: vec![wire::Derivation {
                    target_payload_id: target_id.as_bytes().to_vec(),
                    operations,
                }],
            }),
            &[],
        );
        let selection = protection::Selection::Deployment { deployment: 0, rank: None };
        let package = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        let placed: BTreeMap<Vec<u8>, (String, String)> = package
            .metadata()
            .locations
            .locations
            .iter()
            .map(|location| {
                (
                    location.payload_id.clone(),
                    (location.file.clone(), location.tensor.clone()),
                )
            })
            .collect();
        let required = BTreeSet::from([target_id.as_bytes().to_vec()]);
        PAYLOAD_OPENS.with(|count| count.set(0));
        let retained = root.optional_derivation_sources(&package, &required, &placed);
        assert_eq!(
            retained.get(&source_id.as_bytes().to_vec()),
            Some(&(source.len() as u64))
        );
        assert!(PAYLOAD_OPENS.with(|count| count.get()) <= 1);
    }

    #[test]
    fn verify_ready_hashes_embedded_primary_payloads() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let first = temporary.0.join("verify-embedded");
        let canonical = b"CCCCcccc";
        let mut derived = canonical[4..].to_vec();
        derived.extend_from_slice(&canonical[..4]);
        let publisher = publish_two_deployment_fixture(&first, canonical, &derived);
        let selection = protection::Selection::Deployment { deployment: 0, rank: None };
        let entry = root
            .install_package(
                PackageSource::open(&first, std::slice::from_ref(&publisher), selection).unwrap(),
                "verify-embedded-model",
                "entry",
            )
            .unwrap();
        let source = PackageSource::open(&first, std::slice::from_ref(&publisher), selection)
            .unwrap();
        root.verify_ready(&entry, &source).unwrap();

        let locations = reader::Metadata::open(entry.directory())
            .unwrap()
            .locations
            .locations;
        rewrite_primary_tensors(
            entry.directory(),
            locations,
            &[("frontend.tokenizer", b"corrupt")],
        );
        let error = root.verify_ready(&entry, &source).unwrap_err().to_string();
        assert!(
            error.contains("payload") || error.contains("digest"),
            "embedded primary corruption was not detected: {error}"
        );
    }

    #[test]
    fn verify_ready_with_reports_hashed_payload_bytes() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("verify-progress");
        let target = b"TTTTtttt";
        let target_id = PayloadId::of(target);
        let publisher = publish_fixture(&directory, &[("target", target)], &[(target_id, 0)]);
        let selection = protection::Selection::Deployment { deployment: 0, rank: None };
        let entry = root
            .install_package(
                PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
                    .unwrap(),
                "verify-progress-model",
                "entry",
            )
            .unwrap();
        let source = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        let progress = progress::AcquisitionProgress::new();
        root.verify_ready_with(&entry, &source, Some(&progress)).unwrap();
        let once = progress.snapshot().verification_bytes;
        assert!(once >= target.len() as u64, "{once}");
        root.verify_ready_with(&entry, &source, Some(&progress)).unwrap();
        assert_eq!(progress.snapshot().verification_bytes, 2 * once);
    }

    #[test]
    fn verified_cache_hit_is_counted_once_and_not_rehashed_on_install() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("planned-hit-progress");
        let canonical = b"CCCCcccc";
        let canonical_id = PayloadId::of(canonical);
        let publisher = publish_fixture(
            &directory,
            &[("canonical", canonical)],
            &[(canonical_id, 0)],
        );
        let package = package_source(&directory, &publisher);
        let graph = package.verified()[0].ranks()[0].resources[0]
            .stored
            .as_ref()
            .unwrap();
        let graph_id = PayloadId::from_bytes(graph.payload_id.as_slice().try_into().unwrap());
        let graph_payload = package
            .metadata()
            .payloads(package.verified()[0].catalog(), [graph.payload_id.as_slice()])
            .unwrap();
        let graph_slice = graph_payload.get(&graph.payload_id).unwrap().clone();
        let graph_len = graph_slice.len();
        root.insert_payload(canonical_id, canonical.len() as u64, canonical.as_slice())
            .unwrap();
        root.insert_payload(
            graph_id,
            graph_len,
            SliceReader { slice: graph_slice, offset: 0 },
        )
        .unwrap();
        let progress = progress::AcquisitionProgress::new();
        let plan = root
            .plan_acquisition_with(
                package_source(&directory, &publisher),
                &derivation::Limits::default(),
                Some(&progress),
            )
            .unwrap();
        let verified = canonical.len() as u64 + graph_len;
        assert_eq!(progress.snapshot().verification_bytes, verified);
        let _entry = root.install_planned(plan, "planned-hit", "entry").unwrap();
        assert_eq!(
            progress.snapshot().verification_bytes,
            verified,
            "installation must reuse the verified payload bytes"
        );
        let slice = root.payload(canonical_id).unwrap().unwrap().slice().clone();
        let digest_progress = progress::AcquisitionProgress::new();
        assert_eq!(slice.digest_with(Some(&digest_progress)).unwrap(), canonical_id);
        assert_eq!(
            digest_progress.snapshot().verification_bytes,
            canonical.len() as u64
        );
    }

    #[test]
    fn verify_ready_uses_the_installed_external_tensor_locator() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("verify-external-tensor");
        let target = b"TTTTtttt";
        let target_id = PayloadId::of(target);
        let publisher = publish_fixture(&directory, &[("target", target)], &[(target_id, 0)]);
        let selection = protection::Selection::Deployment { deployment: 0, rank: None };
        let entry = root
            .install_package(
                PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
                    .unwrap(),
                "verify-external-tensor-model",
                "entry",
            )
            .unwrap();
        let source = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        root.verify_ready(&entry, &source).unwrap();

        let mut locations = reader::Metadata::open(entry.directory())
            .unwrap()
            .locations
            .locations;
        let location = locations
            .iter_mut()
            .find(|location| location.payload_id == target_id.as_bytes().to_vec())
            .unwrap();
        location.tensor = "other".into();
        rewrite_primary(entry.directory(), vec![location.clone()], None, &[]);
        let error = root.verify_ready(&entry, &source).unwrap_err().to_string();
        assert!(
            error.contains("tensor") || error.contains("other"),
            "the tampered installed tensor locator was not followed: {error}"
        );
    }

    #[test]
    fn verify_ready_rejects_a_different_signed_envelope() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let first = temporary.0.join("verify-envelope-first");
        let second = temporary.0.join("verify-envelope-second");
        let canonical = b"EEEEeeee";
        let mut derived = canonical[4..].to_vec();
        derived.extend_from_slice(&canonical[..4]);
        let publisher = publish_two_deployment_fixture(&first, canonical, &derived);
        let other_publisher = publish_two_deployment_fixture(&second, canonical, &derived);
        let selection = protection::Selection::Deployment { deployment: 0, rank: None };
        let entry = root
            .install_package(
                PackageSource::open(&first, std::slice::from_ref(&publisher), selection).unwrap(),
                "verify-envelope-model",
                "entry",
            )
            .unwrap();
        let source = PackageSource::open(&first, std::slice::from_ref(&publisher), selection)
            .unwrap();
        root.verify_ready(&entry, &source).unwrap();

        let primary = entry.directory().join("model.safetensors");
        fs::remove_file(&primary).unwrap();
        fs::copy(second.join("model.safetensors"), &primary).unwrap();
        let error = root.verify_ready(&entry, &source).unwrap_err().to_string();
        assert!(
            error.contains("envelope") || error.contains("signed"),
            "different signed envelope was not rejected: {error}"
        );
        let _ = other_publisher;
    }

    #[test]
    fn arbitrary_packed_tensors_install_with_friendly_companions() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("arbitrary-packed");
        let first = b"11111111";
        let second = b"22222222";
        let first_id = PayloadId::of(first);
        let second_id = PayloadId::of(second);
        let publisher = publish_fixture(
            &directory,
            &[("first", first), ("second", second)],
            &[(first_id, 0), (second_id, 8)],
        );
        rewrite_primary_tensors(
            &directory,
            vec![
                wire::PayloadLocation {
                    payload_id: first_id.as_bytes().to_vec(),
                    file: String::new(),
                    tensor: "part0".into(),
                },
                wire::PayloadLocation {
                    payload_id: second_id.as_bytes().to_vec(),
                    file: String::new(),
                    tensor: "part1".into(),
                },
            ],
            &[("part0", first), ("part1", second)],
        );
        let selection = protection::Selection::Deployment { deployment: 0, rank: None };
        let package = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        let entry = root
            .install_package(package, "arbitrary-packed", "entry")
            .unwrap();
        for (relative, bytes) in [
            ("packed/part0.safetensors", first),
            ("packed/part1.safetensors", second),
        ] {
            let slice = entry.payload(relative).unwrap();
            let mut stored = vec![0; slice.len() as usize];
            slice.read_at(0, &mut stored).unwrap();
            assert_eq!(&stored, bytes);
        }
    }
    #[test]
    fn sharded_container_payloads_get_distinct_companions() {
        use crate::package::container::{self, Writer};
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("sharded");
        let first = b"AAAAaaaa";
        let second = b"BBBBbbbb";
        let first_id = PayloadId::of(first);
        let second_id = PayloadId::of(second);
        let publisher = publish_fixture(
            &directory,
            &[("first", first), ("second", second)],
            &[(first_id, 0), (second_id, 8)],
        );
        let shared = directory.join("weights/container.safetensors");
        fs::create_dir_all(shared.parent().unwrap()).unwrap();
        let mut output = File::create(&shared).unwrap();
        let mut writer = Writer::new(&mut output, container::PAYLOAD_FORMAT).unwrap();
        writer
            .add("part0", first.len() as u64, container::Source::Bytes(first.to_vec()))
            .unwrap();
        writer
            .add("part1", second.len() as u64, container::Source::Bytes(second.to_vec()))
            .unwrap();
        writer.finish().unwrap();
        output.sync_all().unwrap();
        rewrite_primary(
            &directory,
            vec![
                wire::PayloadLocation {
                    payload_id: first_id.as_bytes().to_vec(),
                    file: "weights/container.safetensors".into(),
                    tensor: "part0".into(),
                },
                wire::PayloadLocation {
                    payload_id: second_id.as_bytes().to_vec(),
                    file: "weights/container.safetensors".into(),
                    tensor: "part1".into(),
                },
            ],
            None,
            &[],
        );
        let selection = protection::Selection::Deployment {
            deployment: 0,
            rank: None,
        };
        let package = PackageSource::open(
            &directory,
            std::slice::from_ref(&publisher),
            selection,
        )
        .unwrap();
        let entry = root.install_package(package, "sharded", "entry").unwrap();
        for (relative, bytes) in [
            ("weights/container/part0.safetensors", first),
            ("weights/container/part1.safetensors", second),
        ] {
            let slice = entry.payload(relative).unwrap();
            let mut stored = vec![0; slice.len() as usize];
            slice.read_at(0, &mut stored).unwrap();
            assert_eq!(&stored, bytes);
        }
    }

    #[test]
    fn ready_targets_retain_optional_sources_without_requiring_hints() {
        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("optional");
        let target = b"TTTTtttt";
        let source = b"SSSSssss";
        let target2 = b"UUUUuuuu";
        let unused = b"XXXXxxxx";
        let target_id = PayloadId::of(target);
        let source_id = PayloadId::of(source);
        let target2_id = PayloadId::of(target2);
        let unused_id = PayloadId::of(unused);
        let publisher = publish_fixture(
            &directory,
            &[
                ("target", target),
                ("source", source),
                ("target2", target2),
                ("unused", unused),
            ],
            &[(target_id, 0), (target2_id, 8)],
        );
        let locations = |name: &str, id: PayloadId| wire::PayloadLocation {
            payload_id: id.as_bytes().to_vec(),
            file: format!("weights/{name}.safetensors"),
            tensor: COMPANION_TENSOR.into(),
        };
        let unusable = wire::Derivation {
            target_payload_id: target2_id.as_bytes().to_vec(),
            operations: vec![wire::DerivationOperation {
                operation: Some(wire::derivation_operation::Operation::Copy(
                    wire::DerivationCopy {
                        source_payload_id: PayloadId::of(b"missing-source").as_bytes().to_vec(),
                        source_offset: 0,
                        destination_offset: 0,
                        contiguous_bytes: 8,
                        extents: vec![],
                        source_strides: vec![],
                        destination_strides: vec![],
                    },
                )),
            }],
        };
        let unselected = copy(PayloadId::of(b"unused-target"), unused_id);
        rewrite_primary(
            &directory,
            vec![
                locations("target", target_id),
                locations("source", source_id),
                locations("target2", target2_id),
                locations("unused", unused_id),
            ],
            Some(wire::DerivationSet {
                version: derivation::DERIVATION_VERSION,
                derivations: vec![
                    wire::Derivation {
                        target_payload_id: target_id.as_bytes().to_vec(),
                        operations: vec![wire::DerivationOperation {
                            operation: Some(wire::derivation_operation::Operation::Copy(
                                wire::DerivationCopy {
                                    source_payload_id: source_id.as_bytes().to_vec(),
                                    source_offset: 0,
                                    destination_offset: 0,
                                    contiguous_bytes: 8,
                                    extents: vec![],
                                    source_strides: vec![],
                                    destination_strides: vec![],
                                },
                            )),
                        }],
                    },
                    unusable,
                    wire::Derivation {
                        target_payload_id: unselected.target.as_bytes().to_vec(),
                        operations: vec![],
                    },
                ],
            }),
            &[],
        );
        let selection = protection::Selection::Deployment {
            deployment: 0,
            rank: None,
        };
        let package = PackageSource::open(
            &directory,
            std::slice::from_ref(&publisher),
            selection,
        )
        .unwrap();
        let entry = root.install_package(package, "optional", "entry").unwrap();
        // The ready target still loads even though its hint is unusable.
        let target_slice = entry.payload("weights/target.safetensors").unwrap();
        let mut stored = vec![0; target_slice.len() as usize];
        target_slice.read_at(0, &mut stored).unwrap();
        assert_eq!(&stored, target);
        let source_slice = entry.payload("weights/source.safetensors").unwrap();
        let mut stored = vec![0; source_slice.len() as usize];
        source_slice.read_at(0, &mut stored).unwrap();
        assert_eq!(&stored, source);
        let package = PackageSource::open(
            &directory,
            std::slice::from_ref(&publisher),
            selection,
        )
        .unwrap();
        root.verify_ready(&entry, &package).unwrap();
        root.collect().unwrap();
        root.verify_ready(&entry, &package).unwrap();
        assert!(root.payload(source_id).unwrap().is_some());
        assert!(root.payload(PayloadId::of(b"missing-source")).unwrap().is_none());
        assert!(root.payload(unused_id).unwrap().is_none());
        assert!(!entry.directory().join("weights/unused.safetensors").exists());
    }
    #[test]
    fn selected_derivation_sources_stay_linked_through_collection() {
        use crate::package::conversion::emit_derivations;
        use crate::package::export::{
            self, Destination, PublishDeployment, PublishPayload, PublishRank, PublishRequest,
            PublishResource,
        };
        use crate::package::schema::{wire as pwire, FORMAT_VERSION};
        use crate::runtime_package::wire as runtime;
        use crate::runtime_package::Publisher;
        use ring::signature::{Ed25519KeyPair, KeyPair};

        let temporary = Temporary::new();
        let root = temporary.root();
        let directory = temporary.0.join("fixture");
        fs::create_dir_all(&directory).unwrap();
        let graph = directory.join("graph.bin");
        fs::write(&graph, b"graph-bytes").unwrap();
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
        let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key: key.public_key().as_ref().try_into().unwrap(),
            development: false,
        };
        let source = |name: &str, bytes: &[u8]| {
            let path = directory.join(format!("payload-{name}"));
            fs::write(&path, bytes).unwrap();
            PublishPayload {
                role: pwire::PayloadRole::Weights,
                relative: format!("weights/{name}.safetensors"),
                source: export::pin_file(&path, bytes.len() as u64).unwrap(),
                id: PayloadId::of(bytes),
                tensor: None,
            }
        };
        let source0 = b"AAAAaaaa".to_vec();
        let target0 = source0[..4].to_vec();
        let source1 = b"BBBBbbbb".to_vec();
        let target1 = source1[..4].to_vec();
        let unused_source = b"CCCCcccc".to_vec();
        let unused_target = unused_source[..4].to_vec();
        let source0_id = PayloadId::of(&source0);
        let source1_id = PayloadId::of(&source1);
        let unused_source_id = PayloadId::of(&unused_source);
        let initializer = |id: PayloadId, nbytes: u64| pwire::Initializer {
            allocation: "device-0".into(),
            destination_offset: 0,
            source: Some(pwire::PayloadUse {
                slice: Some(pwire::PayloadSlice {
                    payload_id: id.as_bytes().to_vec(),
                    offset: 0,
                    nbytes,
                }),
                role: pwire::PayloadRole::Weights as i32,
            }),
        };
        let rank = |rank: u32, deployment: &str, target: PayloadId, nbytes: u64, payloads| {
            PublishRank {
                body: pwire::RankBody {
                    rank,
                    deployment_id: deployment.into(),
                    context: Some(runtime::Context::default()),
                    allocations: vec![pwire::Allocation {
                        name: "device-0".into(),
                        nbytes: 16,
                        alignment: 4096,
                        address: Some(0x1000_0000),
                        memory: runtime::Memory::Device as i32,
                        ..Default::default()
                    }],
                    initializers: vec![initializer(target, nbytes)],
                    graphs: vec![pwire::Graph {
                        name: "decode".into(),
                        resource: 0,
                        node_count: 1,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                payloads,
                resources: vec![PublishResource {
                    kind: pwire::ExecutableKind::Graph,
                    path: graph.clone(),
                    nbytes: 11,
                }],
            }
        };
        let request = PublishRequest {
            package: pwire::PackageBody {
                format_version: FORMAT_VERSION,
                package_id: vec![7; 32],
                model_identity: "derive".into(),
                hardware_identity: "sm120".into(),
                publisher: Some(pwire::PublisherMetadata {
                    namespace: publisher.namespace.clone(),
                    official_source: "https://huggingface.co/example/model".into(),
                    ..Default::default()
                }),
                frontend: Some(Default::default()),
                sampling_defaults: Some(Default::default()),
                payloads: vec![],
                assets: vec![],
                deployments: vec![],
            },
            payloads: vec![],
            deployments: vec![
                PublishDeployment {
                    body: pwire::DeploymentBody {
                        deployment_id: "sm120-tp2".into(),
                        architecture: vec![12, 0],
                        tensor_parallel: 2,
                        contract: Some(Default::default()),
                        ..Default::default()
                    },
                    ranks: vec![
                        rank(0, "sm120-tp2", PayloadId::of(&target0), 4, vec![
                            source("source0", &source0),
                            source("target0", &target0),
                        ]),
                        rank(1, "sm120-tp2", PayloadId::of(&target1), 4, vec![
                            source("source1", &source1),
                            source("target1", &target1),
                        ]),
                    ],
                    payloads: vec![],
                },
                PublishDeployment {
                    body: pwire::DeploymentBody {
                        deployment_id: "sm120-tp1".into(),
                        architecture: vec![12, 0],
                        tensor_parallel: 1,
                        contract: Some(Default::default()),
                        ..Default::default()
                    },
                    ranks: vec![rank(0, "sm120-tp1", PayloadId::of(&unused_target), 4, vec![
                        source("unused-source", &unused_source),
                        source("unused-target", &unused_target),
                    ])],
                    payloads: vec![],
                },
            ],
            package_id: [7; 32],
            public_key: publisher.public_key,
            accepted: publisher.namespace.clone(),
            no_weights: false,
        };
        export::publish(&request, &key, Destination::Directory(&directory)).unwrap();
        let copy = |source: PayloadId, target: PayloadId| derivation::Derivation {
            target,
            length: 4,
            operations: vec![derivation::Operation::Copy(derivation::Copy {
                source,
                source_offset: 0,
                destination_offset: 0,
                contiguous: 4,
                outer: Vec::new(),
            })],
        };
        emit_derivations(
            &directory,
            &[
                copy(source0_id, PayloadId::of(&target0)),
                copy(source1_id, PayloadId::of(&target1)),
                copy(unused_source_id, PayloadId::of(&unused_target)),
            ],
            &derivation::Limits::default(),
        )
        .unwrap();

        let selection = protection::Selection::Deployment {
            deployment: 0,
            rank: None,
        };
        let package = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        let entry = root.install_package(package, "derive", "entry").unwrap();
        for (name, identity) in [
            ("weights/source0.safetensors", source0_id),
            ("weights/source1.safetensors", source1_id),
        ] {
            let slice = entry.payload(name).unwrap();
            let mut bytes = vec![0; slice.len() as usize];
            slice.read_at(0, &mut bytes).unwrap();
            let expected = if identity == source0_id { &source0 } else { &source1 };
            assert_eq!(&bytes, expected);
            assert!(root.payload(identity).unwrap().is_some());
        }
        let installed = reader::Metadata::open(entry.directory()).unwrap();
        for identity in [source0_id, source1_id, unused_source_id] {
            let located = installed
                .locations
                .locations
                .iter()
                .any(|location| location.payload_id == identity.as_bytes().to_vec());
            assert_eq!(located, identity != unused_source_id, "location for {identity}");
        }
        let package = PackageSource::open(&directory, std::slice::from_ref(&publisher), selection)
            .unwrap();
        root.verify_ready(&entry, &package).unwrap();
        root.collect().unwrap();
        root.verify_ready(&entry, &package).unwrap();
        // Derivation-only sources stay referenced by the entry and survive collection;
        // an unselected deployment's payload and an unreachable hint do not.
        for identity in [source0_id, source1_id] {
            assert!(root.payload(identity).unwrap().is_some(), "collected {identity}");
        }
        assert!(root.payload(unused_source_id).unwrap().is_none());
        assert!(
            !entry
                .directory()
                .join("weights/unused-source.safetensors")
                .exists()
        );
        drop(entry);
    }

}

#[cfg(test)]
mod selection_tests {
    use super::*;

    fn scratch() -> (std::path::PathBuf, Root) {
        let path = std::env::temp_dir().join(format!("helios-cache-select-{}", token()));
        let _ = fs::remove_dir_all(&path);
        let root = Root::open(&path).unwrap();
        (path, root)
    }

    fn publish_dir(root: &Root, model: &str, entry: &str) {
        fs::create_dir_all(root.path().join("models").join(model).join(entry)).unwrap();
    }

    #[test]
    fn selection_and_removal_respect_the_default() {
        let (path, root) = scratch();
        publish_dir(&root, "model", "v1");
        publish_dir(&root, "model", "v2");
        root.publish("model", "v1").unwrap();
        assert_eq!(root.ref_entry("model").unwrap().as_deref(), Some("v1"));
        assert!(root.remove("model", "v1", false).is_err());
        root.publish("model", "v2").unwrap();
        root.remove("model", "v1", false).unwrap();
        assert!(root.remove("model", "v2", false).is_err());
        root.remove("model", "v2", true).unwrap();
        assert_eq!(root.ref_entry("model").unwrap(), None);
        assert!(root.versions("model").unwrap().is_empty());
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    #[cfg(unix)]
    fn failed_default_clear_keeps_the_selected_entry() {
        use std::os::unix::fs::PermissionsExt;

        let (path, root) = scratch();
        publish_dir(&root, "model", "v1");
        root.publish("model", "v1").unwrap();
        let entry_dir = root.path().join("models/model/v1");
        let refs = root.path().join("refs");
        let restore = fs::metadata(&refs).unwrap().permissions();
        fs::set_permissions(&refs, fs::Permissions::from_mode(0o555)).unwrap();

        // The ref cannot be cleared, so nothing may be deleted.
        let error = root.remove("model", "v1", true).unwrap_err();
        assert!(!error.to_string().is_empty());
        assert!(entry_dir.is_dir(), "entry vanished before its default cleared");
        assert_eq!(root.ref_entry("model").unwrap().as_deref(), Some("v1"));

        fs::set_permissions(&refs, restore).unwrap();
        root.remove("model", "v1", true).unwrap();
        assert!(!entry_dir.exists());
        assert_eq!(root.ref_entry("model").unwrap(), None);
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn leased_selected_entry_keeps_its_default() {
        let (path, root) = scratch();
        publish_dir(&root, "model", "v1");
        root.publish("model", "v1").unwrap();
        let entry = root.entry("model", "v1").unwrap();
        let error = root.remove("model", "v1", true).unwrap_err();
        assert!(error.to_string().contains("leased"), "{error}");
        assert_eq!(root.ref_entry("model").unwrap().as_deref(), Some("v1"));
        assert!(root.path().join("models/model/v1").is_dir());
        drop(entry);
        root.remove("model", "v1", true).unwrap();
        assert_eq!(root.ref_entry("model").unwrap(), None);
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn leases_block_removal_and_are_reported() {
        let (path, root) = scratch();
        publish_dir(&root, "model", "v1");
        let entry = root.entry("model", "v1").unwrap();
        assert!(root.is_leased("model", "v1").unwrap());
        assert!(root.remove("model", "v1", true).is_err());
        drop(entry);
        assert!(!root.is_leased("model", "v1").unwrap());
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn source_refs_report_resolved_metadata_without_leasing_or_existence() {
        let (path, root) = scratch();
        publish_dir(&root, "model", "v1");
        {
            let entry = root.entry("model", "v1").unwrap();
            root.publish_source_ref("repo@gb202-tp1-202610170923", "commit-1", &entry, [9; 32])
                .unwrap();
        }
        let reference = root
            .read_source_ref("repo@gb202-tp1-202610170923")
            .unwrap()
            .unwrap();
        assert_eq!(reference.revision(), "commit-1");
        assert_eq!(reference.model(), "model");
        assert_eq!(reference.entry(), "v1");
        assert_eq!(reference.expected_envelope(), Some([9; 32]));
        // The resolved commit stays readable after the entry is evicted, and reading it leases
        // nothing.
        fs::remove_dir_all(root.path().join("models/model/v1")).unwrap();
        let evicted = root.read_source_ref("repo@gb202-tp2-202610170923").unwrap();
        assert!(evicted.is_none());
        let still = root
            .read_source_ref("repo@gb202-tp1-202610170923")
            .unwrap()
            .unwrap();
        assert_eq!(still.revision(), "commit-1");
        assert!(
            root.resolve_source_ref("repo@gb202-tp1-202610170923")
                .unwrap()
                .is_none()
        );
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn source_refs_read_legacy_records_as_unbound_and_validate_versioned_digests() {
        let (path, root) = scratch();
        publish_dir(&root, "model", "entry");
        let entry = root.entry("model", "entry").unwrap();
        let selector = "repository/revision/filename";
        let path = path
            .join("sources")
            .join(PayloadId::of(selector.as_bytes()).to_string());
        fs::write(
            &path,
            serde_json::to_vec(&(selector, "commit", "model", "entry")).unwrap(),
        )
        .unwrap();
        let legacy = root.read_source_ref(selector).unwrap().unwrap();
        assert_eq!(legacy.expected_envelope(), None);

        root.publish_source_ref(selector, "commit", &entry, [3; 32])
            .unwrap();
        let bound = root.read_source_ref(selector).unwrap().unwrap();
        assert_eq!(bound.expected_envelope(), Some([3; 32]));
        let record: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(record["version"], 2);
        assert_eq!(
            record["envelope_sha256"],
            PayloadId::from_bytes([3; 32]).to_hex()
        );

        let mut unsupported = record.clone();
        unsupported["version"] = serde_json::json!(3);
        fs::write(&path, serde_json::to_vec(&unsupported).unwrap()).unwrap();
        assert!(root.read_source_ref(selector).is_err());

        let mut malformed = record;
        malformed["envelope_sha256"] = serde_json::json!("not-a-digest");
        fs::write(&path, serde_json::to_vec(&malformed).unwrap()).unwrap();
        assert!(root.read_source_ref(selector).is_err());
        drop(entry);
        let _ = fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }

    #[test]
    fn capture_payload_accepts_a_caller_thread_reader() {
        // An Rc-backed reader is not Send: streaming hashes on the caller's thread.
        let directory = std::env::temp_dir().join(format!("flare-cache-rc-reader-{}", token()));
        let _ = fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let root = Root::open(&directory).unwrap();
        let bytes: Vec<u8> = (0..65536u32).map(|value| value as u8).collect();
        let expected = PayloadId::of(&bytes);
        let shared = std::rc::Rc::new(bytes);
        let reader_state = std::rc::Rc::clone(&shared);
        struct RcReader(std::rc::Rc<Vec<u8>>, usize);
        impl std::io::Read for RcReader {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                let remaining = self.0.len() - self.1;
                let count = remaining.min(output.len());
                output[..count].copy_from_slice(&self.0[self.1..self.1 + count]);
                self.1 += count;
                Ok(count)
            }
        }
        let object = root
            .capture_payload(reader_state.len() as u64, RcReader(reader_state, 0))
            .unwrap();
        assert_eq!(object.id(), expected);
        assert_eq!(object.length(), shared.len() as u64);
        let _ = std::fs::remove_dir_all(&directory);
        drop(shared);
    }

    #[test]
    fn named_staging_reserves_ids_and_commits_without_deadlock() {
        let (path, root) = scratch();
        publish_dir(&root, "model", "candidate");
        let first = root.stage_named("model", "candidate").unwrap();
        assert_eq!(first.entry, "candidate-2");
        // A simultaneous reservation must choose a different id while the first is live.
        let second = root.stage_named("model", "candidate").unwrap();
        assert_eq!(second.entry, "candidate-3");
        // Committing must hand the held exclusive reservation to the entry as a shared lease on
        // the same descriptor; a fresh descriptor would block forever.
        let first = first.commit().unwrap();
        assert_eq!(first.id(), "candidate-2");
        assert!(root.is_leased("model", "candidate-2").unwrap());
        let second = second.commit().unwrap();
        assert_eq!(second.id(), "candidate-3");
        drop(first);
        drop(second);
        assert!(!root.is_leased("model", "candidate-2").unwrap());
        root.remove("model", "candidate-3", false).unwrap();
        assert_eq!(
            root.versions("model").unwrap(),
            vec!["candidate".to_string(), "candidate-2".to_string()]
        );
        // An aborted reservation releases its name.
        let aborted = root.stage_named("model", "candidate").unwrap();
        assert_eq!(aborted.entry, "candidate-3");
        aborted.abort();
        let reused = root.stage_named("model", "candidate").unwrap();
        assert_eq!(reused.entry, "candidate-3");
        drop(reused);
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn model_keys_decode_back_to_their_identity() {
        for identity in [
            "owner/model",
            "model",
            "a/b@C",
            // A literal escape-looking identity must not decode as an escape.
            "a_2Fb",
            "a_5F2Fb",
            "a--b",
            // Non-ASCII identities survive because bytes are reassembled as UTF-8.
            "模型/阶段",
            "mødel/naïve-ä",
            "emoji/🚀-flash",
        ] {
            let key = model_key(identity).unwrap();
            assert!(
                key.is_ascii(),
                "escaped key must stay filesystem-safe: {key:?}"
            );
            assert_eq!(
                model_identity(&key).unwrap(),
                identity,
                "roundtrip {identity:?}"
            );
        }
        // Malformed escapes, undecodable bytes, and non-canonical keys are rejected.
        assert!(model_identity("owner_model").is_err());
        assert!(model_identity("_2").is_err());
        assert!(model_identity("_4").is_err());
        assert!(model_identity("_ZZ").is_err());
        assert!(
            model_identity("_ff").is_err(),
            "lowercase escape is not canonical"
        );
        assert!(
            model_identity("_80").is_err(),
            "invalid UTF-8 byte is rejected"
        );
        assert!(
            model_identity("_C3").is_err(),
            "truncated UTF-8 sequence is rejected"
        );
    }

    fn copy_tree(source: &Path, target: &Path) {
        fs::create_dir_all(target).unwrap();
        for entry in fs::read_dir(source).unwrap() {
            let entry = entry.unwrap();
            let destination = target.join(entry.file_name());
            if entry.metadata().unwrap().is_dir() {
                copy_tree(&entry.path(), &destination);
            } else {
                fs::copy(entry.path(), destination).unwrap();
            }
        }
    }

    #[test]
    fn summary_counts_rank_payloads_and_marks_deployments_ready() {
        use crate::package::export::{
            self, Destination, PublishDeployment, PublishPayload, PublishRank, PublishRequest,
            PublishResource,
        };
        use crate::package::schema::wire as pwire;
        use crate::package::{PayloadId, schema};
        use ring::signature::KeyPair;

        let (path, root) = scratch();
        let work = path.join("work");
        fs::create_dir_all(&work).unwrap();
        let weight = b"rank scoped weight bytes".to_vec();
        let cubin = b"fake cubin".to_vec();
        let graph = b"fake graph".to_vec();
        let weight_path = work.join("w.prepared");
        fs::write(&weight_path, &weight).unwrap();
        let cubin_path = work.join("k.plain");
        fs::write(&cubin_path, &cubin).unwrap();
        let graph_path = work.join("g.plain");
        fs::write(&graph_path, &graph).unwrap();
        let id = PayloadId::of(&weight);
        let rng = ring::rand::SystemRandom::new();
        let pkcs8 = ring::signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = ring::signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let rank = pwire::RankBody {
            rank: 0,
            deployment_id: "gb202-tp1".into(),
            context: Some(crate::runtime_package::wire::Context::default()),
            allocations: vec![pwire::Allocation {
                name: "device-0".into(),
                nbytes: weight.len() as u64,
                alignment: 256,
                address: Some(0x1000),
                memory: 0,
                sizing: None,
                relocations: vec![],
            }],
            initializers: vec![pwire::Initializer {
                allocation: "device-0".into(),
                destination_offset: 0,
                source: Some(pwire::PayloadUse {
                    slice: Some(pwire::PayloadSlice {
                        payload_id: id.as_bytes().to_vec(),
                        offset: 0,
                        nbytes: weight.len() as u64,
                    }),
                    role: pwire::PayloadRole::Weights as i32,
                }),
            }],
            modules: vec![pwire::Module {
                name: "kernel".into(),
                cubin: 0,
                globals: vec![],
            }],
            functions: vec![pwire::Function {
                module: "kernel".into(),
                symbol: "decode".into(),
                parameter_sizes: vec![],
                attributes: vec![],
            }],
            graphs: vec![pwire::Graph {
                name: "decode".into(),
                flags: 0,
                resource: 1,
                node_count: 1,
                edge_count: 0,
                functions: vec![0],
            }],
            ..Default::default()
        };
        let request = PublishRequest {
            package: pwire::PackageBody {
                format_version: schema::FORMAT_VERSION,
                package_id: [9; 32].to_vec(),
                model_identity: "owner/model".into(),
                hardware_identity: "gb202".into(),
                publisher: Some(pwire::PublisherMetadata {
                    namespace: "example".into(),
                    official_source: "https://huggingface.co/example/model".into(),
                    development: false,
                    license_name: "Example".into(),
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
                body: pwire::DeploymentBody {
                    deployment_id: "gb202-tp1".into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 1,
                    contract: Some(crate::runtime_package::wire::ServingContract {
                        model_contract: Some(crate::runtime_package::wire::ModelContract::default()),
                        pages: 8,
                        state_rows: 2,
                        ..Default::default()
                    }),
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
                        role: pwire::PayloadRole::Weights,
                        relative: "weights/rank-0/weight.safetensors".into(),
                        source: export::pin_file(&weight_path, weight.len() as u64).unwrap(),
                        id,
                        tensor: None,
                    }],
                    resources: vec![
                        PublishResource {
                            kind: pwire::ExecutableKind::Cubin,
                            path: cubin_path,
                            nbytes: cubin.len() as u64,
                        },
                        PublishResource {
                            kind: pwire::ExecutableKind::Graph,
                            path: graph_path,
                            nbytes: graph.len() as u64,
                        },
                    ],
                }],
            }],
            package_id: [9; 32],
            public_key,
            accepted: "example".into(),
            no_weights: false,
        };
        let published = work.join("package");
        fs::create_dir_all(&published).unwrap();
        export::publish(&request, &key, Destination::Directory(&published)).unwrap();
        let key = model_key("owner/model").unwrap();
        let staging = root.stage_named(&key, "gb202-tp1-202610170923").unwrap();
        copy_tree(&published, staging.directory());
        let entry = staging.commit().unwrap();
        let summary = root.summarize_entry(&entry).unwrap();
        assert_eq!(summary.form, "payload");
        assert_eq!(summary.model_identity, "owner/model");
        assert_eq!(summary.deployments.len(), 1);
        assert!(summary.deployments[0].ready);
        assert_eq!(summary.deployments[0].pages, Some(8));
        // Rank payloads (weight + graph) and the deployment payload count toward logical bytes.
        assert!(summary.payloads >= 2, "payloads: {}", summary.payloads);
        assert!(summary.logical_bytes >= weight.len() as u64);
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn removing_a_missing_entry_is_an_error_and_keeps_the_default() {
        let (path, root) = scratch();
        publish_dir(&root, "model", "v1");
        root.publish("model", "v1").unwrap();
        assert!(root.remove("model", "absent", false).is_err());
        assert!(root.remove("model", "absent", true).is_err());
        assert_eq!(root.ref_entry("model").unwrap().as_deref(), Some("v1"));
        assert_eq!(root.versions("model").unwrap(), vec!["v1".to_string()]);
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn collection_never_evicts_published_entries() {
        let (path, root) = scratch();
        publish_dir(&root, "model", "v1");
        root.publish("model", "v1").unwrap();
        let collected = root.collect().unwrap();
        assert_eq!(collected.objects, 0);
        assert!(root.path().join("models/model/v1").is_dir());
        assert!(root.entry("model", "v1").is_ok());
        assert_eq!(root.ref_entry("model").unwrap().as_deref(), Some("v1"));
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn entry_for_path_leases_only_contained_entries() {
        let (path, root) = scratch();
        publish_dir(&root, "model", "v1");
        fs::write(root.path().join("models/model/v1/model.safetensors"), b"x").unwrap();
        let primary = root.path().join("models/model/v1/model.safetensors");
        let entry = root.entry_for_path(&primary).unwrap().unwrap();
        assert_eq!(entry.id(), "v1");
        drop(entry);
        assert!(
            root.entry_for_path(&root.path().join("models/model"))
                .unwrap()
                .is_none()
        );
        let outside = path.join("outside.safetensors");
        fs::write(&outside, b"x").unwrap();
        assert!(root.entry_for_path(&outside).unwrap().is_none());
        assert!(
            root.entry_for_path(
                &root
                    .path()
                    .join("models/../models/model/v1/model.safetensors")
            )
            .unwrap()
            .is_some()
        );
        // A symlink inside an entry cannot alias a path outside the managed tree.
        #[cfg(unix)]
        let outside_dir = path.join("elsewhere");
        #[cfg(unix)]
        {
            fs::create_dir_all(&outside_dir).unwrap();
            fs::write(outside_dir.join("model.safetensors"), b"x").unwrap();
            std::os::unix::fs::symlink(&outside_dir, root.path().join("models/model/v1/link"))
                .unwrap();
            assert!(
                root.entry_for_path(&root.path().join("models/model/v1/link/model.safetensors"))
                    .unwrap()
                    .is_none()
            );
        }
        // The lease taken through the path blocks removal until it is dropped.
        let leased = root.entry_for_path(&primary).unwrap().unwrap();
        assert!(
            root.remove("model", "v1", true).is_err(),
            "leased entry must not be removable"
        );
        drop(leased);
        root.remove("model", "v1", true).unwrap();
        assert!(root.entry_for_path(&primary).unwrap().is_none());
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn legacy_import_normalizes_a_custom_primary_basename() {
        let (path, root) = scratch();
        let source = path.join("source");
        fs::create_dir_all(&source).unwrap();
        let companion: Vec<u8> = vec![0x5a; 4096];
        let companion_name = "b".repeat(64) + ".safetensors";
        fs::write(source.join(&companion_name), &companion).unwrap();
        let layout = format!(
            "{{\"version\":1,\"files\":{{\"\":[[{}, \"{companion_name}\"]]}}}}",
            companion.len()
        );
        let header = format!(
            "{{\"__metadata__\":{{\"format\":\"flare-snapshot\"}},\"flare.payload\":{{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[0,0]}},\"flare.metadata\":{{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}},\"flare.layout\":{{\"dtype\":\"U8\",\"shape\":[{}],\"data_offsets\":[4,{}]}}}}",
            layout.len(),
            4 + layout.len()
        );
        let mut bytes = header.into_bytes();
        while (8 + bytes.len()) % 4096 != 0 {
            bytes.push(b' ');
        }
        let mut primary = (bytes.len() as u64).to_le_bytes().to_vec();
        primary.extend_from_slice(&bytes);
        primary.extend_from_slice(&[0u8; 4]);
        primary.extend_from_slice(layout.as_bytes());
        let custom = source.join("custom-name.safetensors");
        fs::write(&custom, &primary).unwrap();
        let entry = root
            .import_legacy(&custom, "legacy-model", "entry")
            .unwrap();
        assert!(entry.directory().join("model.safetensors").is_file());
        assert!(!entry.directory().join("custom-name.safetensors").exists());
        assert!(entry.directory().join(&companion_name).is_file());
        assert_eq!(fs::read(custom).unwrap(), primary, "source stays intact");
        assert_eq!(
            root.primary(&entry).unwrap(),
            entry.directory().join("model.safetensors")
        );
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn signed_legacy_custom_primary_imports_and_summarizes() {
        let (path, root) = scratch();
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/legacy-custom");
        let source = path.join("source");
        fs::create_dir_all(&source).unwrap();
        for entry in fs::read_dir(&fixture).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name();
            let name = if name == "model.safetensors" {
                "custom-name.safetensors".into()
            } else {
                name
            };
            fs::copy(entry.path(), source.join(name)).unwrap();
        }
        let entry = root
            .import_legacy(
                &source.join("custom-name.safetensors"),
                "legacy-model",
                "entry",
            )
            .unwrap();
        assert!(entry.directory().join("model.safetensors").is_file());
        assert!(!entry.directory().join("custom-name.safetensors").exists());
        assert_eq!(
            root.primary(&entry).unwrap(),
            entry.directory().join("model.safetensors")
        );
        let summary = root.summarize_entry(&entry).unwrap();
        assert_eq!(summary.form, "legacy");
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn primary_falls_back_to_one_unique_signed_container_and_rejects_ambiguity() {
        let (path, root) = scratch();
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/legacy-custom/model.safetensors");
        let bytes = fs::read(&fixture).unwrap();
        let staging = root.stage_named("model", "custom").unwrap();
        fs::write(staging.directory().join("custom-name.safetensors"), &bytes).unwrap();
        let entry = staging.commit().unwrap();
        assert_eq!(
            root.primary(&entry).unwrap(),
            entry.directory().join("custom-name.safetensors")
        );
        // A second signed container makes discovery ambiguous rather than picking one.
        fs::write(entry.directory().join("other.safetensors"), &bytes).unwrap();
        assert!(root.primary(&entry).is_err(), "ambiguous primaries must be rejected");
        let _ = fs::remove_dir_all(path);
    }


}