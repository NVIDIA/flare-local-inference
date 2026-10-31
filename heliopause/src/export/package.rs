/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rank assembly and immutable package publication.
use super::archive::{self, Container, Layout};
use super::protection;
use super::{
    ExportResult as Result, FilePart, FileSource, RankOutput, invalid, wire,
};
use prost::Message;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Write},
    path::Path,
    path::PathBuf,
    time::SystemTime,
};

pub(crate) fn align(value: u64) -> Result<u64> {
    value
        .checked_add(4095)
        .map(|v| v & !4095)
        .ok_or_else(|| invalid("payload extent overflow"))
}

#[derive(Clone, Debug)]
pub(crate) enum SpanKey {
    Resource(usize, String),
    Asset(usize),
    Table(usize, Option<usize>, bool),
}
#[derive(Clone, Debug)]
pub(crate) struct Role {
    pub kind: u32,
    pub owner: Option<usize>,
    pub span: wire::Span,
}
pub(crate) fn span_mut<'a>(index: &'a mut wire::ModelIndex, key: &SpanKey) -> &'a mut wire::Span {
    match key {
        SpanKey::Resource(rank, name) => index.ranks[*rank].resources.get_mut(name).unwrap(),
        SpanKey::Asset(i) => index.assets[*i].payload.as_mut().unwrap(),
        SpanKey::Table(i, shard, scales) => {
            let table = &mut index.engram.as_mut().unwrap().tables[*i];
            match (shard, scales) {
                (Some(j), true) => table.shards[*j].scales.get_or_insert_default(),
                (Some(j), false) => table.shards[*j].weights.get_or_insert_default(),
                (None, true) => table.scales.get_or_insert_default(),
                (None, false) => table.weights.get_or_insert_default(),
            }
        }
    }
}
pub(crate) fn roles(index: &wire::ModelIndex) -> Result<Vec<Role>> {
    let mut result = Vec::new();
    for (owner, rank) in index.ranks.iter().enumerate() {
        let mut kinds = BTreeMap::new();
        for (kind, names) in [
            (
                2,
                rank.modules
                    .iter()
                    .map(|m| m.file.as_str())
                    .chain(rank.graphs.iter().map(|g| g.resource.as_str()))
                    .collect::<Vec<_>>(),
            ),
            (
                3,
                rank.modules
                    .iter()
                    .flat_map(|m| m.globals.iter().map(|g| g.file.as_str()))
                    .collect(),
            ),
            (
                4,
                rank.allocations
                    .iter()
                    .filter_map(|a| a.file.as_deref())
                    .collect(),
            ),
        ] {
            for name in names {
                if !rank.resources.contains_key(name)
                    || kinds
                        .insert(name, kind)
                        .is_some_and(|previous| previous != kind)
                {
                    return Err(invalid("missing or conflicting resource classification"));
                }
            }
        }
        if kinds.len() != rank.resources.len() {
            return Err(invalid("unclassified package resource"));
        }
        for (name, kind) in kinds {
            result.push(Role {
                kind,
                owner: Some(owner),
                span: rank.resources[name].clone(),
            });
        }
        if rank
            .allocations
            .iter()
            .flat_map(|a| &a.data_spans)
            .any(|s| !s.filename.is_empty())
        {
            return Err(invalid("allocation destination spans cannot name a file"));
        }
    }
    for asset in &index.assets {
        result.push(Role {
            kind: 3,
            owner: None,
            span: asset
                .payload
                .clone()
                .ok_or_else(|| invalid("asset lacks payload"))?,
        });
    }
    if let Some(engram) = &index.engram {
        for table in &engram.tables {
            if table.shards.is_empty() {
                for scales in [false, true] {
                    result.push(Role {
                        kind: 4,
                        owner: None,
                        span: if scales {
                            table.scales.clone()
                        } else {
                            table.weights.clone()
                        }
                        .ok_or_else(|| invalid("embedding table lacks payload plane"))?,
                    });
                }
            } else {
                if table.weights.is_some() || table.scales.is_some() {
                    return Err(invalid(
                        "partitioned embedding tables cannot also hold shared planes",
                    ));
                }
                for shard in &table.shards {
                    if shard.rank as usize >= index.ranks.len() {
                        return Err(invalid("embedding partition owner outside package ranks"));
                    }
                    for scales in [false, true] {
                        result.push(Role {
                            kind: 4,
                            owner: Some(shard.rank as usize),
                            span: if scales {
                                shard.scales.clone()
                            } else {
                                shard.weights.clone()
                            }
                            .ok_or_else(|| invalid("embedding partition lacks payload plane"))?,
                        });
                    }
                }
            }
        }
    }
    for role in &result {
        let span = &role.span;
        if !span.filename.is_empty() {
            archive::validate_filename(&span.filename)?;
            if index.version != 5 || role.kind != 4 {
                return Err(invalid(
                    "only stored data in model index 5 may reference a companion",
                ));
            }
        }
        if span.nbytes == 0 || span.offset.checked_add(span.nbytes).is_none() {
            return Err(invalid("empty or overflowing package resource"));
        }
    }
    let mut ordered: Vec<_> = result.iter().map(|r| &r.span).collect();
    ordered.sort_by_key(|s| (&s.filename, s.offset));
    for pair in ordered.windows(2) {
        if pair[0].filename == pair[1].filename && pair[0].offset + pair[0].nbytes > pair[1].offset
        {
            return Err(invalid("overlapping package resources"));
        }
    }
    Ok(result)
}
pub(crate) fn validate_index(index: &wire::ModelIndex, length: u64) -> Result<()> {
    if !matches!(index.version, 4 | 5) || index.ranks.is_empty() {
        return Err(invalid("unsupported model index"));
    }
    let mut spans: Vec<&wire::Span> = index
        .ranks
        .iter()
        .flat_map(|r| r.resources.values())
        .collect();
    spans.extend(index.assets.iter().filter_map(|a| a.payload.as_ref()));
    if let Some(engram) = &index.engram {
        for table in &engram.tables {
            if table.shards.is_empty() {
                spans.extend(table.weights.iter());
                spans.extend(table.scales.iter());
            } else {
                if table.weights.is_some() || table.scales.is_some() {
                    return Err(invalid(
                        "partitioned embedding tables cannot also hold shared planes",
                    ));
                }
                for shard in &table.shards {
                    spans.extend(shard.weights.iter());
                    spans.extend(shard.scales.iter());
                }
            }
        }
    }
    for span in spans {
        if !span.filename.is_empty() {
            archive::validate_filename(&span.filename)?;
            if index.version != 5 {
                return Err(invalid("companion references require model index 5"));
            }
        } else if span
            .offset
            .checked_add(span.nbytes)
            .is_none_or(|end| end > length)
        {
            return Err(invalid("resource exceeds model package payload"));
        }
    }
    if index
        .ranks
        .iter()
        .flat_map(|r| &r.allocations)
        .flat_map(|a| &a.data_spans)
        .any(|s| !s.filename.is_empty())
    {
        return Err(invalid("allocation destination spans cannot name a file"));
    }
    Ok(())
}

#[derive(Deserialize, Serialize)]
struct Sources {
    index: wire::ModelIndex,
    payload: Vec<FilePart>,
    /// Stable semantic weight groups resolved during capture.
    #[serde(default)]
    weight_payload: Option<super::WeightPayloadSources>,
    /// Platform identity detected on the preparing device.
    #[serde(default)]
    hardware: Option<String>,
}
pub(crate) fn write_sources(path: &Path, rank: &RankOutput) -> Result<()> {
    let length = validate_payload(&rank.payload)?;
    validate_index(&rank.index, length)?;
    let mut output = BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    );
    serde_json::to_writer(
        &mut output,
        &Sources {
            index: rank.index.clone(),
            payload: rank.payload.clone(),
            weight_payload: rank.weight_payload.clone(),
            hardware: rank.hardware.clone(),
        },
    )?;
    output.flush()?;
    output.get_ref().sync_all()?;
    Ok(())
}
/// Payload identities one staged rank source requires from the package cache.
pub(crate) fn source_payload_ids(path: &Path) -> Result<Vec<helios_core::package::PayloadId>> {
    let record: Sources = serde_json::from_reader(BufReader::new(File::open(path)?))?;
    let mut ids = Vec::new();
    if let Some(sources) = record.weight_payload.as_ref() {
        if let Some(engram) = &sources.engram {
            for source in super::weight_reuse::engram_uses(engram) {
                let slice = source.slice.as_ref().ok_or_else(|| invalid("cached Engram plane has no slice"))?;
                ids.push(helios_core::package::PayloadId::from_bytes(slice.payload_id.as_slice().try_into()?));
            }
        }
        for group in sources
            .groups
            .iter()
            .chain(std::iter::once(&sources.initial_state))
        {
            if group.payload_id.is_empty() {
                continue;
            }
            let bytes: [u8; 32] = group
                .payload_id
                .as_slice()
                .try_into()
                .map_err(|_| invalid("staged payload id is not 32 bytes"))?;
            ids.push(helios_core::package::PayloadId::from_bytes(bytes));
        }
    }
    Ok(ids)
}

pub(crate) fn read_sources(path: &Path) -> Result<RankOutput> {
    let record: Sources = serde_json::from_reader(BufReader::new(File::open(path)?))?;
    validate_index(&record.index, validate_payload(&record.payload)?)?;
    Ok(RankOutput {
        hardware: record.hardware,
        payload_leases: Default::default(),
        index: record.index,
        payload: record.payload,
        allocations: vec![],
        module_globals: vec![],
        external: vec![],
        staging: None,
        weight_payload: record.weight_payload,
    })
}
pub(crate) fn validate_payload(parts: &[FilePart]) -> Result<u64> {
    let mut end = 0;
    for part in parts {
        if part.offset < end || part.source.offset.checked_add(part.source.nbytes).is_none() {
            return Err(invalid(
                "package payload ranges must be ordered and disjoint",
            ));
        }
        end = part
            .offset
            .checked_add(part.source.nbytes)
            .ok_or_else(|| invalid("payload length overflow"))?;
    }
    Ok(end)
}
pub(crate) fn locate(parts: &[FilePart], span: &wire::Span) -> Result<Vec<FileSource>> {
    let end = span
        .offset
        .checked_add(span.nbytes)
        .ok_or_else(|| invalid("resource length overflow"))?;
    let first = parts
        .partition_point(|p| p.offset <= span.offset)
        .checked_sub(1)
        .ok_or_else(|| invalid("rank export resource lies outside its recorded sources"))?;
    if span.nbytes == 0 {
        let part = &parts[first];
        if span.offset
            > part
                .offset
                .checked_add(part.source.nbytes)
                .ok_or_else(|| invalid("payload length overflow"))?
        {
            return Err(invalid(
                "rank export resource lies outside its recorded sources",
            ));
        }
        return Ok(vec![FileSource {
            path: part.source.path.clone(),
            offset: part.source.offset + span.offset - part.offset,
            nbytes: 0,
        }]);
    }
    let mut cursor = span.offset;
    let mut found = Vec::new();
    for part in &parts[first..] {
        let stop = part
            .offset
            .checked_add(part.source.nbytes)
            .ok_or_else(|| invalid("payload length overflow"))?;
        if part.offset > cursor || cursor >= stop {
            break;
        }
        let size = end.min(stop) - cursor;
        found.push(FileSource {
            path: part.source.path.clone(),
            offset: part.source.offset + cursor - part.offset,
            nbytes: size,
        });
        cursor += size;
        if cursor == end {
            return Ok(found);
        }
    }
    Err(invalid(
        "rank export resource lies outside its recorded sources",
    ))
}

pub(crate) fn combine(
    ranks: &[RankOutput],
    peer_memory: Vec<wire::PeerMemory>,
    cluster: Option<wire::Cluster>,
) -> Result<(wire::ModelIndex, Vec<FilePart>)> {
    let hardware = ranks
        .iter()
        .filter_map(|rank| rank.hardware.as_deref())
        .collect::<std::collections::BTreeSet<_>>();
    if hardware.len() > 1 {
        return Err(invalid("rank exports disagree about the preparing device"));
    }
    let mut index = wire::ModelIndex {
        version: 4,
        peer_memory,
        ..Default::default()
    };
    let mut storage = Vec::new();
    let mut execution = Vec::new();
    for (rank, source) in ranks.iter().enumerate() {
        if source.index.ranks.len() != 1 {
            return Err(invalid("rank export must contain one resource bundle"));
        }
        index.ranks.push(source.index.ranks[0].clone());
        let bundle = &index.ranks[rank];
        let stored: BTreeSet<_> = bundle
            .allocations
            .iter()
            .filter_map(|a| a.file.as_ref())
            .collect();
        let mut names: Vec<_> = bundle.resources.keys().collect();
        names.sort();
        for name in names {
            let key = SpanKey::Resource(rank, name.clone());
            let pieces = locate(&source.payload, &bundle.resources[name])?;
            if stored.contains(name) {
                storage.push((key, pieces));
            } else {
                execution.push((key, pieces));
            }
        }
        if rank == 0 {
            index.contract = source.index.contract.clone();
            index.rebuild = source.index.rebuild.clone();
            index.sampling_defaults = source.index.sampling_defaults.clone();
            index.frontend = source.index.frontend.clone();
            index.engram = source.index.engram.clone();
            index.assets = source.index.assets.clone();
            if let Some(engram) = &index.engram {
                for (i, table) in engram.tables.iter().enumerate() {
                    let shards = if table.shards.is_empty() {
                        vec![None]
                    } else {
                        (0..table.shards.len()).map(Some).collect()
                    };
                    for shard in shards {
                        for scales in [false, true] {
                            let key = SpanKey::Table(i, shard, scales);
                            let span = match (shard, scales) {
                                (Some(j), true) => table.shards[j].scales.clone(),
                                (Some(j), false) => table.shards[j].weights.clone(),
                                (None, true) => table.scales.clone(),
                                (None, false) => table.weights.clone(),
                            }
                            .unwrap_or_default();
                            storage.push((key, locate(&source.payload, &span)?));
                        }
                    }
                }
            }
            for (i, asset) in index.assets.iter().enumerate() {
                execution.push((
                    SpanKey::Asset(i),
                    locate(
                        &source.payload,
                        asset
                            .payload
                            .as_ref()
                            .ok_or_else(|| invalid("asset lacks payload"))?,
                    )?,
                ));
            }
        } else if source.index.engram.is_some() {
            return Err(invalid(
                "offloaded embedding tables must be stored once on rank zero",
            ));
        }
    }
    let mut payload = Vec::new();
    let mut cursor = 0;
    for (key, pieces) in storage.into_iter().chain(execution) {
        cursor = align(cursor)?;
        let span = span_mut(&mut index, &key);
        span.offset = cursor;
        span.filename.clear();
        for source in pieces {
            let size = source.nbytes;
            payload.push(FilePart {
                offset: cursor,
                source,
            });
            cursor = cursor
                .checked_add(size)
                .ok_or_else(|| invalid("payload extent overflow"))?;
        }
    }
    if let Some(cluster) = cluster {
        if index.ranks.len() != cluster.world as usize {
            return Err(invalid(
                "a multi-host package holds every rank of its cluster",
            ));
        }
        index.cluster = Some(cluster);
    }
    validate_index(&index, cursor)?;
    Ok((index, payload))
}

use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
fn py_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}
fn decode_payload(value: &str) -> Result<Vec<FilePart>> {
    let tuples: Vec<(u64, String, u64, u64)> = serde_json::from_str(value)?;
    Ok(tuples
        .into_iter()
        .map(|(offset, path, source_offset, nbytes)| FilePart {
            offset,
            source: FileSource {
                path: path.into(),
                offset: source_offset,
                nbytes,
            },
        })
        .collect())
}
#[pyfunction]
#[pyo3(signature=(path,ranks,peer_memory,publisher,cluster,cache=None))]
fn export_payload(
    py: Python<'_>,
    path: String,
    ranks: Vec<String>,
    peer_memory: String,
    publisher: Option<String>,
    cluster: Option<String>,
    cache: Option<String>,
) -> PyResult<String> {
    py.detach(|| {
        let ranks: Result<Vec<_>> = ranks.iter().map(|p| read_sources(Path::new(p))).collect();
        let peers: Vec<wire::PeerMemory> = serde_json::from_str(&peer_memory)?;
        let cluster: Option<wire::Cluster> =
            cluster.as_deref().map(serde_json::from_str).transpose()?;
        let publisher = publisher
            .as_deref()
            .map(serde_json::from_str::<protection::PublisherRequest>)
            .transpose()?
            .map(|p| p.publisher())
            .transpose()?
            .ok_or_else(|| invalid("payload export requires a publisher identity"))?;
        let result = publish_payload(
            Path::new(&path),
            &ranks?,
            peers,
            cluster,
            &publisher,
            cache.map(PathBuf::from),
            false,
            "directory",
        )?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(serde_json::to_string(&result)?)
    })
    .map_err(py_error)
}
#[pyfunction]
#[pyo3(signature=(path,ranks,peer_memory,publisher,cluster,cache,no_weights=false,form="directory"))]
fn export_combine(
    py: Python<'_>,
    path: String,
    ranks: Vec<String>,
    peer_memory: String,
    publisher: Option<String>,
    cluster: Option<String>,
    cache: Option<String>,
    no_weights: bool,
    form: &str,
) -> PyResult<String> {
    py.detach(|| {
        let sources: Result<Vec<_>> = ranks.iter().map(|p| read_sources(Path::new(p))).collect();
        let sources = sources?;
        let peers: Vec<wire::PeerMemory> = serde_json::from_str(&peer_memory)?;
        let cluster: Option<wire::Cluster> =
            cluster.as_deref().map(serde_json::from_str).transpose()?;
        let publisher = publisher
            .as_deref()
            .map(serde_json::from_str::<protection::PublisherRequest>)
            .transpose()?
            .map(|p| p.publisher())
            .transpose()?;
        let publisher = publisher.ok_or_else(|| invalid("export requires a publisher identity"))?;
        let result = publish_payload(
            Path::new(&path),
            &sources,
            peers,
            cluster,
            &publisher,
            cache.map(PathBuf::from),
            no_weights,
            form,
        )?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(serde_json::to_string(&result)?)
    })
    .map_err(py_error)
}
#[pyfunction]
fn export_write_sources(
    py: Python<'_>,
    path: String,
    index: Vec<u8>,
    payload: String,
    hardware: Option<String>,
    weight_payload: Option<String>,
) -> PyResult<()> {
    py.detach(|| {
        let index = wire::ModelIndex::decode(index.as_slice())?;
        let payload = decode_payload(&payload)?;
        // Rewrites carry the captured side metadata forward: export naming needs the
        // real capture platform, and declared weight groups need their payload ids.
        let weight_payload = match weight_payload {
            Some(value) => Some(serde_json::from_str(&value)?),
            None => None,
        };
        write_sources(
            Path::new(&path),
            &RankOutput {
                hardware,
                payload_leases: Default::default(),
                index,
                payload,
                allocations: vec![],
                module_globals: vec![],
                external: vec![],
                staging: None,
                weight_payload,
            },
        )
    })
    .map_err(py_error)
}
#[pyfunction]
fn export_read_sources(py: Python<'_>, path: String) -> PyResult<String> {
    py.detach(|| {
        let rank = read_sources(Path::new(&path))?;
        let payload: Vec<_> = rank
            .payload
            .iter()
            .map(|p| {
                (
                    p.offset,
                    p.source.path.to_string_lossy().into_owned(),
                    p.source.offset,
                    p.source.nbytes,
                )
            })
            .collect();
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(serde_json::to_string(
            &serde_json::json!({
                "index":rank.index.encode_to_vec(),
                "payload":payload,
                "hardware":rank.hardware,
                "weight_payload":rank.weight_payload,
            }),
        )?)
    })
    .map_err(py_error)
}
/// Publish captured ranks of one tensor-parallel deployment in the payload
/// package format. The directory canonical form is produced; packed form is a
/// keyless container conversion of it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn publish_payload(
    path: &Path,
    ranks: &[RankOutput],
    peer_memory: Vec<wire::PeerMemory>,
    cluster: Option<wire::Cluster>,
    publisher: &protection::Publisher,
    cache: Option<PathBuf>,
    no_weights: bool,
    form: &str,
) -> Result<serde_json::Value> {
    if !matches!(form, "directory" | "packed") {
        return Err(invalid(format!("unknown package form {form:?}")));
    }
    use helios_core::package::{export as payload_export, schema::wire as pwire};
    if ranks.is_empty() {
        return Err(invalid("payload export requires at least one rank"));
    }
    // An empty destination publishes into the shared package cache only.
    let cache_only = path.as_os_str().is_empty();
    // The payload form is a directory at exactly the requested path. The
    // caller chooses directory output; this never rewrites the destination.
    if !cache_only && path.exists() {
        return Err(invalid("payload export destination already exists"));
    }
    if cache_only && cache.is_none() {
        return Err(invalid("cache-only export requires the package cache"));
    }
    if no_weights && cache.is_none() {
        return Err(invalid("no-weights export requires the package cache"));
    }
    if !cache_only {
        const RESERVED: &str = "payload export destination already exists";
        if path.exists() {
            return Err(invalid(RESERVED));
        }
    }
    let parent = if cache_only {
        cache.as_deref().unwrap_or(Path::new("."))
    } else {
        path.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
    };
    fs::create_dir_all(parent)?;
    let staging_directory = tempfile::Builder::new()
        .prefix(".flare-payload-")
        .tempdir_in(parent)?;
    let staging = staging_directory.path().to_path_buf();
    let progress = helios_core::startup::StartupProgress::stderr_rows();
    let result = (|| -> Result<serde_json::Value> {
        let assembly = progress.phase("Assemble package payloads", Some(ranks.len() as u64));
        let (index, payload) = combine(ranks, peer_memory, cluster.clone())?;
        let rank_count = u32::try_from(index.ranks.len()).map_err(|_| invalid("too many ranks"))?;
        let architecture = index.ranks[0].architecture.clone();
        let platform = ranks
            .iter()
            .find_map(|rank| rank.hardware.clone())
            .ok_or_else(|| {
                invalid("export naming requires the capture device identity (gb10 or gb202)")
            })?;
        let deployment_id = cluster
            .as_ref()
            .map(|c| c.export_id.clone())
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| format!("{platform}-tp{rank_count}"));
        let mut package_payloads = Vec::new();
        let mut assets = Vec::new();
        for (position, asset) in index.assets.iter().enumerate() {
            let span = asset
                .payload
                .as_ref()
                .ok_or_else(|| invalid("frontend asset lacks a payload"))?;
            let (id, nbytes, source) = payload_source(
                &payload,
                span,
                &staging,
                &format!("frontend-{position}.payload"),
            )?;
            let tensor = format!(
                "frontend.{}",
                match asset.name.as_str() {
                    "tokenizer" => "tokenizer".to_string(),
                    "chat_template" | "chat-template" => "chat-template".to_string(),
                    other => other.replace('_', "-"),
                }
            );
            assets.push(pwire::FrontendAsset {
                name: asset.name.clone(),
                payload: Some(pwire::PayloadUse {
                    slice: Some(pwire::PayloadSlice {
                        payload_id: id.as_bytes().to_vec(),
                        offset: 0,
                        nbytes,
                    }),
                    role: pwire::PayloadRole::Frontend as i32,
                }),
                sha256: hex_bytes(&asset.sha256)?,
            });
            package_payloads.push(payload_export::PublishPayload {
                role: pwire::PayloadRole::Frontend,
                relative: String::new(),
                source,
                id,
                tensor: Some(tensor),
            });
        }
        let package_id = random_id()?;
        let package = pwire::PackageBody {
            format_version: helios_core::package::schema::FORMAT_VERSION,
            package_id: package_id.to_vec(),
            model_identity: index
                .contract
                .as_ref()
                .map(|c| c.model_id.clone())
                .filter(|id| !id.is_empty())
                .unwrap_or_else(|| "model".into()),
            hardware_identity: platform.clone(),
            publisher: Some(pwire::PublisherMetadata {
                namespace: publisher.namespace.clone(),
                official_source: publisher.official_source.clone(),
                development: publisher.development,
                license_name: publisher
                    .license
                    .as_ref()
                    .map(|l| l.name.clone())
                    .unwrap_or_default(),
                license_link: publisher
                    .license
                    .as_ref()
                    .map(|l| l.link.clone())
                    .unwrap_or_default(),
                // Rank topology belongs to the deployment, never to publisher identity: a
                // publisher identity that encoded the rank count would prevent combining
                // independently exported tensor-parallel sizes into one package.
                ranks: 0,
            }),
            frontend: index.frontend.clone(),
            sampling_defaults: index.sampling_defaults.clone(),
            payloads: vec![],
            assets,
            deployments: vec![],
        };
        let cache_root = match cache.as_deref() {
            Some(path) => Some(helios_core::package::cache::Root::open(path)?),
            None => None,
        };
        let mut leases = Vec::new();
        let mut engram_payloads = Vec::new();
        let reused_engram = ranks[0].weight_payload.as_ref().and_then(|sources| sources.engram.as_ref());
        let engram = if let Some(engram) = reused_engram {
            Some(cached_engram(engram, cache_root.as_ref().ok_or_else(|| invalid("Engram reuse requires the package cache"))?,
                &deployment_id, &mut engram_payloads, &mut leases)?)
        } else { index
            .engram
            .as_ref()
            .map(|engram| {
                package_engram(
                    engram,
                    &payload,
                    &staging,
                    &deployment_id,
                    &mut engram_payloads,
                )
            })
            .transpose()? };
        let needs_cache = ranks.iter().any(|rank| {
            rank.weight_payload.as_ref().is_some_and(|sources| {
                sources
                    .groups
                    .iter()
                    .chain(std::iter::once(&sources.initial_state))
                    .any(|group| !group.payload_id.is_empty())
            })
        });
        if needs_cache && cache_root.is_none() {
            return Err(invalid(
                "declared capture payloads require the package cache; use --model-cache",
            ));
        }
        // Cache object leases stay alive until publication completes, so GC cannot unlink a declared
        // payload between lookup and its slice reaching the publisher.
        let mut rank_plans = Vec::with_capacity(ranks.len());
        for (position, rank) in ranks.iter().enumerate() {
            assembly.detail(format!("rank {position}"));
            rank_plans.push(rank_plan(
                rank,
                position as u32,
                rank_count,
                &index.peer_memory,
                &staging,
                &deployment_id,
                cache_root.as_ref(),
                &mut leases,
            )?);
            assembly.advance(1);
        }
        let deployment = pwire::DeploymentBody {
            deployment_id: deployment_id.clone(),
            architecture,
            tensor_parallel: rank_count,
            contract: index.contract.clone(),
            cluster: index.cluster.clone(),
            rebuild: index.rebuild.clone(),
            engram,
            payloads: vec![],
            ranks: vec![],
        };
        let staged_payloads =
            package_payloads.len() + rank_plans.iter().map(|r| r.payloads.len()).sum::<usize>();
        let request = payload_export::PublishRequest {
            package,
            payloads: package_payloads,
            deployments: vec![payload_export::PublishDeployment {
                body: deployment,
                ranks: rank_plans,
                payloads: engram_payloads,
            }],
            package_id,
            public_key: publisher.public_key().try_into().unwrap(),
            accepted: publisher.namespace.clone(),
            no_weights,
        };
        assembly.finish(format!("{rank_count} ranks"));
        let reuse = ranks.iter().any(|rank| rank.weight_payload.as_ref()
            .is_some_and(|sources| !sources.cached.is_empty() || sources.engram.is_some()));
        let cached_weights: Vec<_> = if reuse {
            let mut ids = BTreeSet::new();
            for rank in ranks {
                if let Some(sources) = &rank.weight_payload {
                    for descriptor in &sources.cached {
                        ids.insert(helios_core::package::PayloadId::from_bytes(descriptor.id.as_slice().try_into()?));
                    }
                    if let Some(engram) = &sources.engram {
                        for source in super::weight_reuse::engram_uses(engram) {
                            let slice = source.slice.as_ref().ok_or_else(|| invalid("cached Engram plane has no slice"))?;
                            ids.insert(helios_core::package::PayloadId::from_bytes(slice.payload_id.as_slice().try_into()?));
                        }
                    }
                }
            }
            leases.iter().filter(|object| ids.contains(&object.id())).cloned().collect()
        } else { Vec::new() };
        let cached_sources: Vec<_> = cached_weights.iter().map(|object|
            super::io::Source::from_file(object.file().clone(), object.slice().offset(), object.length()))
            .collect::<Result<_>>()?;
        if let Some(cache) = &cache {
            let root = helios_core::package::cache::Root::open(cache)?;
            // The cache owns model-key encoding: an unusable identity is an
            // error, never a lossy rewrite that could merge two models.
            let model = helios_core::package::cache::model_key(&request.package.model_identity)?;
            let candidate = helios_core::package::naming::deployment_candidate(
                Some(&platform),
                &[rank_count],
                SystemTime::now(),
            )
            .ok_or_else(|| invalid("export naming requires a supported platform"))?;
            let mut staging_entry = root.stage_named(&model, &candidate)?;
            let published = if reuse { payload_export::publish_reusing_cached_weights(
                &request, publisher.signing_key(),
                payload_export::Destination::Cache(&root, &mut staging_entry), &cached_weights,
            )? } else { payload_export::publish(
                &request,
                publisher.signing_key(),
                payload_export::Destination::Cache(&root, &mut staging_entry),
            )? };
            for source in &cached_sources {
                if !source.identity().same_content(&super::io::Identity::of(&source.file)?) {
                    return Err(invalid("cached weight source changed during publication"));
                }
            }
            let commit = progress.phase("Commit package", None);
            let payload_ids: Vec<String> = published
                .payloads
                .iter()
                .map(|payload| payload.id.to_string())
                .collect();
            let placements: Vec<serde_json::Value> = published
                .payloads
                .iter()
                .filter_map(|payload| {
                    payload.tensor.as_ref().map(|tensor| {
                        serde_json::json!({"id": payload.id.to_string(), "tensor": tensor})
                    })
                })
                .collect();
            let pending = staging_entry.entry().to_string();
            let committed = match staging_entry.commit() {
                Ok(entry) => entry,
                Err(error) => {
                    // commit can fail after the entry rename (directory sync); that tree exists
                    // but is unselected and not collectable. Report and remove it best-effort.
                    let cleanup = root.remove(&model, &pending, false);
                    return Err(match cleanup {
                        Ok(()) => invalid(error.to_string()),
                        Err(cleanup) => invalid(format!(
                            "{error}; unselected cache entry {model}@{pending} could not be removed: {cleanup}"
                        )),
                    });
                }
            };
            let entry_id = committed.id().to_string();
            let model_key = committed.model().to_string();
            // Standalone output precedes this export's own default publication. A failure here
            // means the default was never touched by this export; remove only our own entry and
            // never clear or rewrite a ref someone else may have selected concurrently.
            if !cache_only {
                let standalone = (|| -> Result<()> {
                    if form == "packed" {
                        let directory = staging.join("packed");
                        root.export_entry(&committed, &directory)?;
                        let packed = staging.join("model.safetensors");
                        helios_core::package::conversion::pack(&directory, &packed)?;
                        fs::hard_link(&packed, path).map_err(|error| {
                            invalid(format!(
                                "publish packed package at {}: {error}",
                                path.display()
                            ))
                        })?;
                        // The package is published; a stale temporary is not a failure.
                        let _ = fs::remove_file(&packed);
                        sync_parent(path)?;
                    } else {
                        root.export_entry(&committed, path)?;
                    }
                    Ok(())
                })();
                if let Err(error) = standalone {
                    drop(committed);
                    return match root.remove(&model_key, &entry_id, false) {
                        Ok(()) => Err(error),
                        Err(cleanup) => Err(invalid(format!(
                            "{error}; the failed export left cache entry {model_key}@{entry_id}, which could not be removed: {cleanup}; remove it with `flare models remove`"
                        ))),
                    };
                }
            }
            // The entry is complete; this is the only step that selects it. `write_ref` renames
            // under the publication and model locks, so a failure after the rename may already be
            // visible. Do not guess or roll back another caller's selection.
            if let Err(error) = root.publish_entry(&committed) {
                drop(committed);
                return Err(invalid(format!(
                    "publishing the default for completed cache entry {model_key}@{entry_id} failed: {error}; the entry is retained and the ref update may already be visible; verify with `flare models list`"
                )));
            }
            commit.finish(committed.directory().display().to_string());
            return Ok(serde_json::json!({
                "format": format!("payload-{form}"),
                "destination": if cache_only {
                    "cache".to_string()
                } else {
                    format!("cache-and-{form}")
                },
                "path": (!cache_only).then(|| path.display().to_string()),
                "entry_path": committed.directory().display().to_string(),
                "model": committed.model(),
                "entry": committed.id(),
                "payloads": staged_payloads,
                "payload_ids": payload_ids,
                "placements": placements,
            }));
        }
        if cache_only {
            return Err(invalid("cache-only export requires the package cache"));
        }
        // Without a configured cache the tree is built in the private staging
        // workspace and published with the shared atomic no-replace rename.
        let package_directory = staging.join("package");
        fs::create_dir_all(&package_directory)?;
        let published = payload_export::publish(
            &request,
            publisher.signing_key(),
            payload_export::Destination::Directory(&package_directory),
        )?;
        let commit = progress.phase("Commit package", None);
        let payload_ids: Vec<String> = published
            .payloads
            .iter()
            .map(|payload| payload.id.to_string())
            .collect();
        let placements: Vec<serde_json::Value> = published
            .payloads
            .iter()
            .filter_map(|payload| {
                payload.tensor.as_ref().map(
                    |tensor| serde_json::json!({"id": payload.id.to_string(), "tensor": tensor}),
                )
            })
            .collect();
        if form == "packed" {
            let packed = staging.join("model.safetensors");
            helios_core::package::conversion::pack(&package_directory, &packed)?;
            fs::hard_link(&packed, path).map_err(|error| {
                invalid(format!(
                    "publish packed package at {}: {error}",
                    path.display()
                ))
            })?;
            // The package is published; a stale temporary is not a failure.
            let _ = fs::remove_file(&packed);
            helios_core::package::export::sync_destination_parent(path)?;
        } else {
            helios_core::package::export::publish_staged_directory(&package_directory, path)?;
        }
        commit.finish(path.display().to_string());
        Ok(serde_json::json!({
            "format": format!("payload-{form}"),
            "path": path.display().to_string(),
            "manifest_bytes": published.manifest.len(),
            "payloads": published.payloads.len(),
            "payload_ids": payload_ids,
            "placements": placements,
        }))
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

type Materialized = (helios_core::package::PayloadId, u64, PathBuf);

const MAX_PLANE_WORKERS: usize = 8;

/// Resolve each `(span, staging name)` plane with [`payload_source`] on parallel readers, largest
/// first. Each plane is hashed by one synchronous reader, so hashing the planes one at a time
/// leaves the device mostly idle. Returns the sources in the order of `planes`; the first error
/// stops scheduling and is returned once running planes finish.
fn plane_sources(
    payload: &[FilePart],
    planes: &[(&wire::Span, String)],
    staging: &Path,
) -> Result<
    Vec<(
        helios_core::package::PayloadId,
        u64,
        helios_core::package::container::Slice,
    )>,
> {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let workers = std::thread::available_parallelism()
        .map_or(1, |workers| workers.get())
        .min(MAX_PLANE_WORKERS)
        .min(planes.len());
    let mut order: Vec<usize> = (0..planes.len()).collect();
    order.sort_by_key(|&index| std::cmp::Reverse(planes[index].0.nbytes));
    let cursor = AtomicUsize::new(0);
    let failure = Mutex::new(None);
    let resolved = Mutex::new((0..planes.len()).map(|_| None).collect::<Vec<_>>());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                if failure.lock().unwrap().is_some() {
                    break;
                }
                let Some(&index) = order.get(cursor.fetch_add(1, Ordering::Relaxed)) else {
                    break;
                };
                let (span, name) = &planes[index];
                match payload_source(payload, span, staging, name) {
                    Ok(source) => resolved.lock().unwrap()[index] = Some(source),
                    Err(error) => {
                        failure.lock().unwrap().get_or_insert(error);
                        break;
                    }
                }
            });
        }
    });
    if let Some(error) = failure.into_inner().unwrap() {
        return Err(error);
    }
    Ok(resolved
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|source| source.expect("every plane resolved"))
        .collect())
}

fn payload_source(
    payload: &[FilePart],
    span: &wire::Span,
    staging: &Path,
    name: &str,
) -> Result<(
    helios_core::package::PayloadId,
    u64,
    helios_core::package::container::Slice,
)> {
    use helios_core::package::{container::Slice, export::pin_file};
    let pieces = locate(payload, span)?;
    if let [piece] = pieces.as_slice() {
        let source = Slice::new(
            std::sync::Arc::new(File::open(&piece.path)?),
            piece.offset,
            piece.nbytes,
        )?;
        return Ok((source.digest()?, span.nbytes, source));
    }
    let (id, nbytes, path) = materialize(payload, span, staging, name)?;
    Ok((id, nbytes, pin_file(&path, nbytes)?))
}

fn materialize(
    payload: &[FilePart],
    span: &wire::Span,
    staging: &Path,
    name: &str,
) -> Result<Materialized> {
    use sha2::{Digest, Sha256};
    let pieces = locate(payload, span)?;
    let path = staging.join(name);
    let mut file = File::create(&path)?;
    let mut hasher = Sha256::new();
    for piece in pieces {
        let mut source = File::open(&piece.path)?;
        use std::io::Seek;
        source.seek(std::io::SeekFrom::Start(piece.offset))?;
        let mut remaining = piece.nbytes;
        let mut buffer = vec![0u8; 1 << 20];
        while remaining > 0 {
            use std::io::Read;
            let chunk = remaining.min(buffer.len() as u64) as usize;
            source.read_exact(&mut buffer[..chunk])?;
            hasher.update(&buffer[..chunk]);
            std::io::Write::write_all(&mut file, &buffer[..chunk])?;
            remaining -= chunk as u64;
        }
    }
    file.sync_all()?;
    let digest: [u8; 32] = hasher.finalize().into();
    Ok((
        helios_core::package::PayloadId::from_bytes(digest),
        span.nbytes,
        path,
    ))
}

/// Translate declared provenance exactly: source tensors, their ordered
/// half-open per-axis ranges, concatenation axis, declared packing geometry, and
/// the statistic domain. Absent declarations stay absent rather than being
/// approximated from names.
fn prepared_provenance(
    view: &super::WeightPayloadView,
) -> (
    Vec<helios_core::package::schema::wire::WeightSource>,
    Option<helios_core::package::schema::wire::PackingParameters>,
    String,
) {
    use helios_core::package::schema::wire as pwire;
    let sources = view
        .sources
        .iter()
        .map(|source| pwire::WeightSource {
            tensor: source.tensor.clone(),
            ranges: source
                .ranges
                .iter()
                .map(|(begin, end)| pwire::TensorRange {
                    begin: *begin,
                    end: *end,
                })
                .collect(),
            concatenation_axis: source.concatenation_axis.unwrap_or(0),
            piece_encoding: source.piece_encoding.clone(),
            pieces: source
                .pieces
                .iter()
                .map(|piece| pwire::WeightPiece {
                    source_offset: piece.source_offset,
                    destination_offset: piece.destination_offset,
                    contiguous_bytes: piece.contiguous_bytes,
                    extents: piece.extents.clone(),
                    source_strides: piece.source_strides.clone(),
                    destination_strides: piece.destination_strides.clone(),
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    // Prefer declared geometry; a bare format name stays format-only rather than
    // having geometry inferred from it.
    let packing = match &view.packing_parameters {
        Some(parameters) => Some(pwire::PackingParameters {
            format: parameters.format.clone(),
            block_size: parameters.block_size,
            group_size: parameters.group_size,
            scale_atom_rows: parameters.scale_atom_rows,
            tile_order: parameters.tile_order,
            row_rotation: parameters.row_rotation,
            locality_tile: parameters.locality_tile,
        }),
        None => (!view.packing.is_empty()).then(|| pwire::PackingParameters {
            format: view.packing.clone(),
            ..Default::default()
        }),
    };
    (sources, packing, view.statistics_domain.clone())
}

fn hex_bytes(text: &str) -> Result<Vec<u8>> {
    if text.len() % 2 != 0 {
        return Err(invalid("invalid hexadecimal digest"));
    }
    (0..text.len())
        .step_by(2)
        .map(|index| {
            u8::from_str_radix(&text[index..index + 2], 16)
                .map_err(|_| invalid("invalid hexadecimal digest"))
        })
        .collect()
}

/// True when one semantic group's pieces on one allocation cover the range.
fn group_covers(
    spans: &[super::WeightPayloadSourceSpan],
    allocation: &str,
    start: u64,
    end: u64,
) -> bool {
    let mut pieces: Vec<(u64, u64)> = spans
        .iter()
        .filter(|span| span.allocation == allocation)
        .map(|span| {
            (
                span.destination_offset,
                span.destination_offset + span.nbytes,
            )
        })
        .collect();
    pieces.sort_unstable();
    let mut cursor = start;
    for (piece_start, piece_stop) in pieces {
        if piece_start > cursor {
            break;
        }
        cursor = cursor.max(piece_stop);
        if cursor >= end {
            return true;
        }
    }
    false
}

fn sync_parent(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    {
        let directory =
            File::open(parent).map_err(|error| invalid(format!("open export parent: {error}")))?;
        directory
            .sync_all()
            .map_err(|error| invalid(format!("sync export parent: {error}")))?;
    }
    Ok(())
}

fn random_id() -> Result<[u8; 32]> {
    let mut bytes = [0u8; 32];
    let rng = ring::rand::SystemRandom::new();
    ring::rand::SecureRandom::fill(&rng, &mut bytes)
        .map_err(|_| invalid("package identity generation failed"))?;
    Ok(bytes)
}

fn safe_group(name: &str) -> Result<String> {
    let safe = !name.is_empty()
        && name.len() <= 96
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    if safe {
        Ok(name.to_string())
    } else {
        Err(invalid("unsafe weight group name"))
    }
}

fn cached_engram(
    engram: &helios_core::package::schema::wire::Engram,
    root: &helios_core::package::cache::Root,
    deployment: &str,
    installed: &mut Vec<helios_core::package::export::PublishPayload>,
    leases: &mut Vec<helios_core::package::cache::PayloadObject>,
) -> Result<helios_core::package::schema::wire::Engram> {
    use helios_core::package::{PayloadId, export::PublishPayload, schema::wire as pwire};
    for (index, table) in engram.tables.iter().enumerate() {
        let mut planes = vec![
            (
                format!("weights/engram-{index}-weights.safetensors"),
                &table.weights,
            ),
            (
                format!("weights/engram-{index}-scales.safetensors"),
                &table.scales,
            ),
        ];
        for (position, shard) in table.shards.iter().enumerate() {
            planes.extend([(format!("deployments/{deployment}/engram-{index}-shard-{position}-weights.safetensors"), &shard.weights),
                (format!("deployments/{deployment}/engram-{index}-shard-{position}-scales.safetensors"), &shard.scales)]);
        }
        for (relative, source) in planes {
            let Some(source) = source else { continue };
            let slice = source
                .slice
                .as_ref()
                .ok_or_else(|| invalid("cached Engram plane has no slice"))?;
            let id = PayloadId::from_bytes(slice.payload_id.as_slice().try_into()?);
            let object = root
                .payload(id)?
                .ok_or_else(|| invalid(format!("missing cached Engram payload {id}")))?;
            if slice
                .offset
                .checked_add(slice.nbytes)
                .is_none_or(|end| end > object.length())
            {
                return Err(invalid("cached Engram plane exceeds its payload object"));
            }
            installed.push(PublishPayload {
                role: pwire::PayloadRole::Weights,
                relative,
                source: object.slice().clone(),
                id,
                tensor: None,
            });
            leases.push(object);
        }
    }
    Ok(engram.clone())
}
fn package_engram(
    engram: &wire::Engram,
    payload: &[FilePart],
    staging: &Path,
    deployment: &str,
    installed: &mut Vec<helios_core::package::export::PublishPayload>,
) -> Result<helios_core::package::schema::wire::Engram> {
    use helios_core::package::schema::wire as pwire;
    let use_of = |id: &helios_core::package::PayloadId, nbytes: u64| pwire::PayloadUse {
        slice: Some(pwire::PayloadSlice {
            payload_id: id.as_bytes().to_vec(),
            offset: 0,
            nbytes,
        }),
        role: pwire::PayloadRole::Weights as i32,
    };
    // Validate every table and list its planes first, so the planes can be hashed together.
    let mut planes = Vec::new();
    for (index, table) in engram.tables.iter().enumerate() {
        if table.shards.is_empty() {
            planes.push((
                table
                    .weights
                    .as_ref()
                    .ok_or_else(|| invalid("engram table lacks weights"))?,
                format!("engram-{index}-w.payload"),
            ));
            planes.push((
                table
                    .scales
                    .as_ref()
                    .ok_or_else(|| invalid("engram table lacks scales"))?,
                format!("engram-{index}-s.payload"),
            ));
            continue;
        }
        if table.weights.is_some() || table.scales.is_some() {
            return Err(invalid(
                "partitioned engram table cannot also carry shared planes",
            ));
        }
        let mut shard_ranks = BTreeSet::new();
        for (position, shard) in table.shards.iter().enumerate() {
            if !shard_ranks.insert(shard.rank) {
                return Err(invalid(format!(
                    "engram table repeats shard rank {}",
                    shard.rank
                )));
            }
            planes.push((
                shard
                    .weights
                    .as_ref()
                    .ok_or_else(|| invalid("engram shard lacks weights"))?,
                format!("engram-{index}-shard-{position}-w.payload"),
            ));
            planes.push((
                shard
                    .scales
                    .as_ref()
                    .ok_or_else(|| invalid("engram shard lacks scales"))?,
                format!("engram-{index}-shard-{position}-s.payload"),
            ));
        }
    }
    let mut sources = plane_sources(payload, &planes, staging)?.into_iter();
    let mut next = || sources.next().expect("one source per listed plane");
    let mut tables = Vec::new();
    for (index, table) in engram.tables.iter().enumerate() {
        let (weights, scales, shards) = if table.shards.is_empty() {
            let weights = next();
            let scales = next();
            for (plane, materialized) in [("weights", &weights), ("scales", &scales)] {
                installed.push(helios_core::package::export::PublishPayload {
                    role: pwire::PayloadRole::Weights,
                    relative: format!("weights/engram-{index}-{plane}.safetensors"),
                    source: materialized.2.clone(),
                    id: materialized.0,
                    tensor: None,
                });
            }
            (
                Some(use_of(&weights.0, weights.1)),
                Some(use_of(&scales.0, scales.1)),
                Vec::new(),
            )
        } else {
            let mut shards = Vec::with_capacity(table.shards.len());
            for (position, shard) in table.shards.iter().enumerate() {
                let weights = next();
                let scales = next();
                for (plane, materialized) in [("weights", &weights), ("scales", &scales)] {
                    installed.push(helios_core::package::export::PublishPayload {
                        role: pwire::PayloadRole::Weights,
                        relative: format!(
                            "deployments/{deployment}/engram-{index}-shard-{position}-{plane}.safetensors"
                        ),
                        source: materialized.2.clone(),
                        id: materialized.0,
                        tensor: None,
                    });
                }
                shards.push(pwire::EngramTableShard {
                    rank: shard.rank,
                    first_row: shard.first_row,
                    rows: shard.rows,
                    weights: Some(use_of(&weights.0, weights.1)),
                    scales: Some(use_of(&scales.0, scales.1)),
                });
            }
            (None, None, shards)
        };
        tables.push(pwire::EngramTable {
            layer: table.layer,
            rows: table.rows,
            embedding_width: table.embedding_width,
            scale_block: table.scale_block,
            weights,
            scales,
            hash: table.hash.as_ref().map(|hash| pwire::EngramHash {
                method: hash.method,
                orders: hash.orders.clone(),
                multipliers: hash.multipliers.clone(),
                primes: hash.primes.clone(),
                offsets: hash.offsets.clone(),
            }),
            weight_buffer: table.weight_buffer.clone(),
            scale_buffer: table.scale_buffer.clone(),
            hash_buffer: table.hash_buffer.clone(),
            ready_event: table.ready_event.clone(),
            encoding: table.encoding,
            request_buffer: table.request_buffer.clone(),
            global_scale: table.global_scale,
            shards,
        });
    }
    Ok(pwire::Engram {
        token_map: engram.token_map.clone(),
        pad_token_id: engram.pad_token_id,
        compressed_vocabulary: engram.compressed_vocabulary,
        tables,
    })
}

/// Accumulates one rank's initial-state bytes into a single payload file.
struct StateFile {
    path: PathBuf,
    file: File,
    hasher: sha2::Sha256,
    offset: u64,
}

impl StateFile {
    fn new(path: PathBuf) -> Result<Self> {
        let file = File::create(&path)?;
        Ok(Self {
            path,
            file,
            hasher: sha2::Digest::new(),
            offset: 0,
        })
    }

    fn append(&mut self, payload: &[FilePart], span: &wire::Span) -> Result<(u64, u64)> {
        use sha2::Digest;
        let start = self.offset;
        for piece in locate(payload, span)? {
            use std::io::{Read, Seek, SeekFrom, Write};
            let mut source = File::open(&piece.path)?;
            source.seek(SeekFrom::Start(piece.offset))?;
            let mut remaining = piece.nbytes;
            let mut buffer = vec![0u8; 1 << 20];
            while remaining > 0 {
                let chunk = remaining.min(buffer.len() as u64) as usize;
                source.read_exact(&mut buffer[..chunk])?;
                self.hasher.update(&buffer[..chunk]);
                self.file.write_all(&buffer[..chunk])?;
                remaining -= chunk as u64;
            }
            self.offset = self
                .offset
                .checked_add(piece.nbytes)
                .ok_or_else(|| invalid("initial-state length overflow"))?;
        }
        Ok((start, self.offset - start))
    }

    fn finish(self) -> Result<(helios_core::package::PayloadId, u64, PathBuf)> {
        use sha2::Digest;
        self.file.sync_all()?;
        let digest: [u8; 32] = self.hasher.finalize().into();
        Ok((
            helios_core::package::PayloadId::from_bytes(digest),
            self.offset,
            self.path,
        ))
    }
}

/// Convert one captured rank into a package rank plan.
///
/// Semantic weight groups keep their declared storage class: stable groups are
/// canonical (`weights/<group>` at TP1), tensor-parallel-specific
/// (`weights/tp<N>/rank-R/<group>` above TP1, so distinct exported sizes cannot
/// share one relative path), tuned
/// groups are deployment/rank scoped. Every other retained byte — unclassified
/// captures, residual state, module globals — is appended to one per-rank
/// initial-state payload with signed slices.
fn rank_plan(
    rank: &RankOutput,
    rank_index: u32,
    tensor_parallel: u32,
    peers: &[wire::PeerMemory],
    staging: &Path,
    deployment_id: &str,
    cache: Option<&helios_core::package::cache::Root>,
    leases: &mut Vec<helios_core::package::cache::PayloadObject>,
) -> Result<helios_core::package::export::PublishRank> {
    use helios_core::package::{export as payload_export, schema::wire as pwire};
    let bundle = &rank.index.ranks[0];
    let mut allocations = Vec::with_capacity(bundle.allocations.len());
    for allocation in &bundle.allocations {
        allocations.push(pwire::Allocation {
            name: allocation.name.clone(),
            nbytes: allocation.nbytes,
            alignment: allocation.alignment,
            address: allocation.address,
            memory: allocation.memory,
            sizing: allocation.sizing.clone(),
            relocations: allocation.relocations.clone(),
        });
    }
    let mut payloads = Vec::new();
    let mut initializers = Vec::new();
    let mut state = StateFile::new(staging.join(format!("rank-{rank_index}-state.payload")))?;
    let canonical = tensor_parallel == 1;
    // A cached initial-state object and locally staged module globals are two payloads with the
    // same role; they need distinct canonical names or one would overwrite the other.
    let mut cached_state = false;

    if let Some(sources) = rank.weight_payload.as_ref() {
        for group in sources
            .groups
            .iter()
            .chain(std::iter::once(&sources.initial_state))
        {
            let tuned = group.storage.as_str() == "tuned" || group.name.ends_with(":tuned");
            let base = group.name.strip_suffix(":tuned").unwrap_or(&group.name);
            let base = safe_group(base)?;
            let is_state = std::ptr::eq(group, &sources.initial_state);
            let total: u64 = group.spans.iter().try_fold(0u64, |sum, span| {
                sum.checked_add(span.nbytes)
                    .ok_or_else(|| invalid("weight group overflow"))
            })?;
            if !group.payload_id.is_empty() {
                // Declared capture publishes each group as one shared cache object. `source_offset`
                // is relative to that object and already accounts for owner-anchored zero elision,
                // so no staging file is written or re-read here.
                let root = cache.ok_or_else(|| {
                    invalid("declared capture payloads require the package cache")
                })?;
                let id_bytes: [u8; 32] = group
                    .payload_id
                    .as_slice()
                    .try_into()
                    .map_err(|_| invalid("declared payload identity is not 32 bytes"))?;
                let id = helios_core::package::PayloadId::from_bytes(id_bytes);
                let object = root.payload(id)?.ok_or_else(|| {
                    invalid(format!(
                        "declared payload {id} is absent from the package cache"
                    ))
                })?;
                if object.length() != group.nbytes {
                    return Err(invalid(format!(
                        "declared payload {id} length {} differs from group {} length {}",
                        object.length(),
                        group.name,
                        group.nbytes
                    )));
                }
                let role = if is_state {
                    pwire::PayloadRole::InitialState
                } else if tuned {
                    pwire::PayloadRole::TunedWeights
                } else {
                    pwire::PayloadRole::Weights
                };
                if is_state {
                    cached_state = true;
                }
                let relative = if is_state {
                    format!(
                        "deployments/{deployment_id}/rank-{rank_index}/initial-state.safetensors"
                    )
                } else if tuned {
                    format!("weights/tuned/{deployment_id}/rank-{rank_index}/{base}.safetensors")
                } else if canonical {
                    format!("weights/{base}.safetensors")
                } else {
                    format!("weights/tp{tensor_parallel}/rank-{rank_index}/{base}.safetensors")
                };
                payloads.push(payload_export::PublishPayload {
                    role,
                    relative,
                    source: object.slice().clone(),
                    id,
                    tensor: None,
                });
                leases.push(object);
                if group.spans.is_empty() {
                    return Err(invalid(format!(
                        "declared payload {} has no spans",
                        group.name
                    )));
                }
                for span in &group.spans {
                    if span.nbytes == 0 {
                        continue;
                    }
                    let end = span
                        .source_offset
                        .checked_add(span.nbytes)
                        .ok_or_else(|| invalid("declared payload span overflow"))?;
                    if end > group.nbytes {
                        return Err(invalid(format!(
                            "declared payload {} span exceeds its stored bytes",
                            group.name
                        )));
                    }
                    initializers.push(pwire::Initializer {
                        allocation: span.allocation.clone(),
                        destination_offset: span.destination_offset,
                        source: Some(pwire::PayloadUse {
                            slice: Some(pwire::PayloadSlice {
                                payload_id: group.payload_id.clone(),
                                offset: span.source_offset,
                                nbytes: span.nbytes,
                            }),
                            role: role as i32,
                        }),
                    });
                }
                continue;
            }
            if is_state {
                for span in &group.spans {
                    if span.nbytes == 0 {
                        continue;
                    }
                    let wire_span = wire::Span {
                        offset: span.source_offset,
                        nbytes: span.nbytes,
                        filename: String::new(),
                    };
                    let (offset, nbytes) = state.append(&rank.payload, &wire_span)?;
                    initializers.push(pwire::Initializer {
                        allocation: span.allocation.clone(),
                        destination_offset: span.destination_offset,
                        source: Some(pwire::PayloadUse {
                            slice: Some(pwire::PayloadSlice {
                                payload_id: Vec::new(),
                                offset,
                                nbytes,
                            }),
                            role: pwire::PayloadRole::InitialState as i32,
                        }),
                    });
                }
                continue;
            }
            if total == 0 || group.spans.is_empty() {
                continue;
            }
            let (id, nbytes, path) = materialize_payload_spans(
                &rank.payload,
                &group.spans,
                staging,
                &format!("rank-{rank_index}-{base}.payload"),
            )?;
            if nbytes == 0 {
                let _ = fs::remove_file(&path);
                continue;
            }
            let relative = if tuned {
                format!("weights/tuned/{deployment_id}/rank-{rank_index}/{base}.safetensors")
            } else if canonical {
                format!("weights/{base}.safetensors")
            } else {
                format!("weights/tp{tensor_parallel}/rank-{rank_index}/{base}.safetensors")
            };
            let role = if tuned {
                pwire::PayloadRole::TunedWeights
            } else {
                pwire::PayloadRole::Weights
            };
            payloads.push(payload_export::PublishPayload {
                role,
                relative,
                source: payload_export::pin_file(&path, nbytes)?,
                id,
                tensor: None,
            });
            let mut offset = 0u64;
            for span in &group.spans {
                if span.nbytes == 0 {
                    continue;
                }
                initializers.push(pwire::Initializer {
                    allocation: span.allocation.clone(),
                    destination_offset: span.destination_offset,
                    source: Some(pwire::PayloadUse {
                        slice: Some(pwire::PayloadSlice {
                            payload_id: id.as_bytes().to_vec(),
                            offset,
                            nbytes: span.nbytes,
                        }),
                        role: role as i32,
                    }),
                });
                offset += span.nbytes;
            }
        }
    } else {
        for allocation in &bundle.allocations {
            let Some(file) = allocation.file.as_ref() else {
                continue;
            };
            let span = bundle
                .resources
                .get(file)
                .ok_or_else(|| invalid("allocation names an absent resource"))?;
            let (offset, nbytes) = state.append(&rank.payload, span)?;
            if nbytes == 0 {
                continue;
            }
            let mut payload_offset = offset;
            for data_span in &allocation.data_spans {
                if data_span.nbytes == 0 {
                    continue;
                }
                initializers.push(pwire::Initializer {
                    allocation: allocation.name.clone(),
                    destination_offset: data_span.offset,
                    source: Some(pwire::PayloadUse {
                        slice: Some(pwire::PayloadSlice {
                            payload_id: Vec::new(),
                            offset: payload_offset,
                            nbytes: data_span.nbytes,
                        }),
                        role: pwire::PayloadRole::InitialState as i32,
                    }),
                });
                payload_offset += data_span.nbytes;
            }
        }
    }

    let mut resources = Vec::new();
    let mut modules = Vec::new();
    for (index, module) in bundle.modules.iter().enumerate() {
        let span = bundle
            .resources
            .get(&module.file)
            .ok_or_else(|| invalid("module names an absent resource"))?;
        let path = materialize_span(
            &rank.payload,
            span,
            staging,
            &format!("rank{rank_index}-cubin-{index}.plain"),
        )?;
        resources.push(payload_export::PublishResource {
            kind: pwire::ExecutableKind::Cubin,
            path,
            nbytes: span.nbytes,
        });
        let mut globals = Vec::new();
        for (global_index, global) in module.globals.iter().enumerate() {
            let span = bundle
                .resources
                .get(&global.file)
                .ok_or_else(|| invalid("module global names an absent resource"))?;
            let (offset, nbytes) = state.append(&rank.payload, span)?;
            if nbytes != global.nbytes {
                return Err(invalid(
                    "module global length differs from its stored bytes",
                ));
            }
            let _ = global_index;
            globals.push(pwire::ModuleGlobal {
                symbol: global.symbol.clone(),
                nbytes: global.nbytes,
                contents: Some(pwire::PayloadUse {
                    slice: Some(pwire::PayloadSlice {
                        payload_id: Vec::new(),
                        offset,
                        nbytes,
                    }),
                    role: pwire::PayloadRole::InitialState as i32,
                }),
                relocations: global.relocations.clone(),
            });
        }
        modules.push(pwire::Module {
            name: module.name.clone(),
            cubin: index as u32,
            globals,
        });
    }
    let mut graphs = Vec::new();
    let cubin_count = resources.len() as u32;
    for (index, graph) in bundle.graphs.iter().enumerate() {
        let span = bundle
            .resources
            .get(&graph.resource)
            .ok_or_else(|| invalid("graph names an absent resource"))?;
        let path = materialize_span(
            &rank.payload,
            span,
            staging,
            &format!("rank{rank_index}-graph-{index}.plain"),
        )?;
        resources.push(payload_export::PublishResource {
            kind: pwire::ExecutableKind::Graph,
            path,
            nbytes: span.nbytes,
        });
        graphs.push(pwire::Graph {
            name: graph.name.clone(),
            flags: graph.flags,
            resource: cubin_count + index as u32,
            node_count: graph.node_count,
            edge_count: graph.edge_count,
            functions: graph.functions.clone(),
        });
    }

    // One initial-state payload per rank carries residual state and globals.
    let (state_id, state_nbytes, state_path) = state.finish()?;
    if state_nbytes > 0 {
        payloads.push(payload_export::PublishPayload {
            role: pwire::PayloadRole::InitialState,
            relative: if cached_state {
                format!("deployments/{deployment_id}/rank-{rank_index}/initial-state-globals.safetensors")
            } else {
                format!("deployments/{deployment_id}/rank-{rank_index}/initial-state.safetensors")
            },
            source: payload_export::pin_file(&state_path, state_nbytes)?,
            id: state_id,
            tensor: None,
        });
        for initializer in &mut initializers {
            if let Some(use_) = initializer.source.as_mut()
                && let Some(slice) = use_.slice.as_mut()
                && slice.payload_id.is_empty()
            {
                slice.payload_id = state_id.as_bytes().to_vec();
            }
        }
        for module in &mut modules {
            for global in &mut module.globals {
                if let Some(use_) = global.contents.as_mut()
                    && let Some(slice) = use_.slice.as_mut()
                {
                    slice.payload_id = state_id.as_bytes().to_vec();
                }
            }
        }
    } else {
        let _ = fs::remove_file(&state_path);
    }

    let peers = selected_peers(peers, rank_index)
        .into_iter()
        .map(|peer| wire::PeerMemory {
            owner: peer.owner,
            address: peer.address,
            nbytes: peer.nbytes,
            readers: peer.readers,
        })
        .collect();
    let prepared = match bundle.rebuild_weights.as_ref() {
        Some(catalog) => Some(convert_prepared(
            catalog,
            bundle,
            rank.weight_payload.as_ref(),
        )?),
        None => None,
    };
    let functions = bundle
        .functions
        .iter()
        .map(|function| pwire::Function {
            module: function.module.clone(),
            symbol: function.symbol.clone(),
            parameter_sizes: function.parameter_sizes.clone(),
            attributes: function.attributes.clone(),
        })
        .collect();
    Ok(payload_export::PublishRank {
        body: pwire::RankBody {
            rank: rank_index,
            deployment_id: deployment_id.to_string(),
            context: bundle.context.clone(),
            reservations: bundle.reservations.clone(),
            allocations,
            initializers,
            payloads: vec![],
            resources: vec![],
            modules,
            functions,
            graphs,
            events: bundle.events.clone(),
            peers,
            execution: bundle.execution.clone(),
            roce: bundle.roce.clone(),
            prepared_weights: prepared,
        },
        payloads,
        resources,
    })
}

/// Select peer records the rank owns or reads.
fn selected_peers(peers: &[wire::PeerMemory], rank: u32) -> Vec<wire::PeerMemory> {
    peers
        .iter()
        .filter(|peer| peer.owner == rank || peer.readers.contains(&rank))
        .cloned()
        .collect()
}

fn materialize_payload_spans(
    payload: &[FilePart],
    spans: &[super::WeightPayloadSourceSpan],
    staging: &Path,
    name: &str,
) -> Result<Materialized> {
    use sha2::{Digest, Sha256};
    let path = staging.join(name);
    let mut file = File::create(&path)?;
    let mut hasher = Sha256::new();
    let mut total = 0u64;
    for span in spans {
        let span_value = wire::Span {
            offset: span.source_offset,
            nbytes: span.nbytes,
            filename: String::new(),
        };
        for piece in locate(payload, &span_value)? {
            use std::io::{Read, Seek, SeekFrom, Write};
            let mut source = File::open(&piece.path)?;
            source.seek(SeekFrom::Start(piece.offset))?;
            let mut remaining = piece.nbytes;
            let mut buffer = vec![0u8; 1 << 20];
            while remaining > 0 {
                let chunk = remaining.min(buffer.len() as u64) as usize;
                source.read_exact(&mut buffer[..chunk])?;
                hasher.update(&buffer[..chunk]);
                file.write_all(&buffer[..chunk])?;
                remaining -= chunk as u64;
            }
            total += piece.nbytes;
        }
    }
    file.sync_all()?;
    let digest: [u8; 32] = hasher.finalize().into();
    Ok((
        helios_core::package::PayloadId::from_bytes(digest),
        total,
        path,
    ))
}

fn materialize_span(
    payload: &[FilePart],
    span: &wire::Span,
    staging: &Path,
    name: &str,
) -> Result<PathBuf> {
    let (_, _, path) = materialize(payload, span, staging, name)?;
    Ok(path)
}

pub(crate) fn convert_prepared(
    catalog: &wire::RebuildWeights,
    bundle: &wire::BundleIndex,
    groups: Option<&super::WeightPayloadSources>,
) -> Result<helios_core::package::schema::wire::PreparedWeightCatalog> {
    use helios_core::package::schema::wire as pwire;
    let index_of: std::collections::BTreeMap<&str, u64> = bundle
        .allocations
        .iter()
        .enumerate()
        .map(|(i, a)| (a.name.as_str(), i as u64))
        .collect();
    let mut views = Vec::new();
    for tensor in &catalog.tensors {
        let storage = tensor
            .storage
            .as_ref()
            .ok_or_else(|| invalid("prepared weight lacks storage"))?;
        let allocation = *index_of
            .get(storage.allocation.as_str())
            .ok_or_else(|| invalid("prepared weight names an absent allocation"))?;
        let extent = tensor
            .shape
            .iter()
            .zip(&tensor.stride)
            .try_fold(1u64, |value, (&size, &stride)| {
                (size - 1)
                    .checked_mul(stride)
                    .and_then(|span| value.checked_add(span))
            })
            .and_then(|value| value.checked_mul(u64::from(tensor.element_bytes)))
            .ok_or_else(|| invalid("prepared weight extent overflow"))?;
        // Prefer the declared semantic group for this view; retained-coverage
        // inference exists only for older captures without declarations.
        let mut group = groups
            .and_then(|groups| {
                groups
                    .views
                    .iter()
                    .find(|view| view.name == tensor.name)
                    .map(|view| view.group.clone())
            })
            .unwrap_or_default();
        let declared_group = !group.is_empty();
        // Coverage remains the fallback that must not introduce equivalence:
        // it only applies when the declaration named no group for this view.
        let mut matched: Option<&str> = None;
        let mut ambiguous = false;
        if let (false, Some(groups)) = (declared_group, groups) {
            let end = storage
                .offset
                .checked_add(extent)
                .ok_or_else(|| invalid("prepared weight extent overflow"))?;
            for candidate in groups.groups.iter() {
                if group_covers(
                    &candidate.spans,
                    storage.allocation.as_str(),
                    storage.offset,
                    end,
                ) {
                    if matched.replace(candidate.name.as_str()).is_some() {
                        ambiguous = true;
                    }
                }
            }
        }
        if !declared_group {
            group = match (matched, ambiguous) {
                (Some(name), false) => name.to_string(),
                _ => String::new(),
            };
        }
        let recorded =
            groups.and_then(|groups| groups.views.iter().find(|view| view.name == tensor.name));
        let (sources, packing, statistics_domain) =
            recorded.map(prepared_provenance).unwrap_or_default();
        views.push(pwire::PreparedWeightView {
            name: tensor.name.clone(),
            group,
            dtype: tensor.dtype.clone(),
            element_bytes: tensor.element_bytes,
            shape: tensor.shape.clone(),
            stride: tensor.stride.clone(),
            layout_type: tensor.layout_type.clone(),
            allocation,
            offset: storage.offset,
            nbytes: extent,
            packing,
            sources,
            padding: vec![],
            statistics_domain,
        });
    }
    Ok(pwire::PreparedWeightCatalog {
        version: catalog.version,
        views,
        unavailable: catalog.unavailable.clone(),
    })
}

pub(crate) fn read_index(path: &Path) -> Result<(wire::ModelIndex, u64)> {
    read_index_container(Container::open(path)?)
}
fn read_index_container(container: Container) -> Result<(wire::ModelIndex, u64)> {
    let index = if container.format == "flare-staging" {
        wire::ModelIndex::decode(container.envelope.as_slice())?
    } else {
        let envelope = wire::ProtectedIndex::decode(container.envelope.as_slice())?;
        if envelope.version != 6 || envelope.groups.len() != 3 || envelope.ranks.is_empty() {
            return Err(invalid("invalid protected package envelope"));
        }
        let metadata = wire::PublisherMetadata::decode(envelope.groups[0].body.as_slice())?;
        let mut index = metadata
            .index
            .ok_or_else(|| invalid("missing signed model index"))?;
        for group in envelope.ranks {
            let rank = wire::RankProtection::decode(group.body.as_slice())?;
            index
                .ranks
                .push(rank.bundle.ok_or_else(|| invalid("missing rank bundle"))?);
        }
        index
    };
    let length = if let Some(layout) = &container.layout {
        layout.files[""].iter().map(|(n, _)| n).sum()
    } else {
        container.payload_length
    };
    validate_index(&index, length)?;
    Ok((index, container.base))
}
#[pyfunction]
fn export_read_index<'py>(py: Python<'py>, path: String) -> PyResult<(Bound<'py, PyBytes>, u64)> {
    let (bytes, base) = py
        .detach(|| read_index(Path::new(&path)).map(|(index, base)| (index.encode_to_vec(), base)))
        .map_err(py_error)?;
    Ok((PyBytes::new(py, &bytes), base))
}
fn container_json(container: Container) -> Result<String> {
    Ok(serde_json::to_string(
        &serde_json::json!({"header":container.header_json,"base":container.base,"envelope":container.envelope,"layout":container.layout}),
    )?)
}
#[pyfunction]
fn export_read_container(py: Python<'_>, path: String) -> PyResult<String> {
    py.detach(|| container_json(Container::open(Path::new(&path))?))
        .map_err(py_error)
}
#[pyfunction]
fn export_read_container_fd(py: Python<'_>, descriptor: i64) -> PyResult<String> {
    py.detach(|| {
        container_json(Container::from_file(std::sync::Arc::new(duplicate_file(
            descriptor,
        )?))?)
    })
    .map_err(py_error)
}
fn duplicate_file(descriptor: i64) -> Result<File> {
    #[cfg(unix)]
    {
        use std::os::fd::FromRawFd;
        let descriptor = i32::try_from(descriptor)?;
        let duplicate = unsafe { libc::fcntl(descriptor, libc::F_DUPFD_CLOEXEC, 0) };
        if duplicate < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(unsafe { File::from_raw_fd(duplicate) })
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::FromRawHandle;
        use windows_sys::Win32::{
            Foundation::{DUPLICATE_SAME_ACCESS, DuplicateHandle},
            System::Threading::GetCurrentProcess,
        };
        unsafe extern "C" {
            fn _get_osfhandle(fd: i32) -> isize;
        }
        let handle = unsafe { _get_osfhandle(i32::try_from(descriptor)?) };
        if handle == -1 {
            return Err(invalid("invalid package file descriptor"));
        }
        let process = unsafe { GetCurrentProcess() };
        let mut duplicate = std::ptr::null_mut();
        if unsafe {
            DuplicateHandle(
                process,
                handle as *mut _,
                process,
                &mut duplicate,
                0,
                0,
                DUPLICATE_SAME_ACCESS,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(unsafe { File::from_raw_handle(duplicate) })
    }
}
#[pyfunction]
fn export_read_payload<'py>(
    py: Python<'py>,
    path: String,
    filename: String,
    offset: u64,
    nbytes: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = py
        .detach(|| {
            let mut files = archive::PackageSources::open(Path::new(&path))?;
            let mut buffer = vec![0; nbytes];
            files.read_at(&filename, offset, &mut buffer)?;
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(buffer)
        })
        .map_err(py_error)?;
    Ok(PyBytes::new(py, &bytes))
}
#[pyfunction]
fn export_payload_length(py: Python<'_>, path: String, filename: String) -> PyResult<u64> {
    py.detach(|| archive::PackageSources::open(Path::new(&path))?.logical_length(&filename))
        .map_err(py_error)
}
#[pyfunction]
fn export_container_header<'py>(
    py: Python<'py>,
    length: u64,
    index_size: u64,
    format: String,
) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = archive::container_header(length, index_size, &format).map_err(py_error)?;
    Ok(PyBytes::new(py, &bytes))
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativePackageFiles>()?;
    module.add_function(wrap_pyfunction!(export_locate_sources, module)?)?;
    module.add_function(wrap_pyfunction!(export_roles, module)?)?;
    module.add_function(wrap_pyfunction!(export_validate_layout, module)?)?;
    module.add_function(wrap_pyfunction!(export_signature_message, module)?)?;
    module.add_function(wrap_pyfunction!(export_validate_publisher, module)?)?;
    module.add_function(wrap_pyfunction!(export_validate_license, module)?)?;
    super::protection::register(module)?;
    module.add_function(wrap_pyfunction!(export_payload, module)?)?;
    module.add_function(wrap_pyfunction!(export_combine, module)?)?;
    module.add_function(wrap_pyfunction!(export_write_sources, module)?)?;
    module.add_function(wrap_pyfunction!(export_read_sources, module)?)?;
    module.add_function(wrap_pyfunction!(export_read_index, module)?)?;
    module.add_function(wrap_pyfunction!(export_read_index_fd, module)?)?;
    module.add_function(wrap_pyfunction!(export_read_container, module)?)?;
    module.add_function(wrap_pyfunction!(export_read_container_fd, module)?)?;
    module.add_function(wrap_pyfunction!(export_read_payload, module)?)?;
    module.add_function(wrap_pyfunction!(export_payload_length, module)?)?;
    module.add_function(wrap_pyfunction!(export_container_header, module)?)?;
    Ok(())
}
#[pyfunction]
fn export_roles(py: Python<'_>, index: Vec<u8>) -> PyResult<String> {
    py.detach(||{let index=wire::ModelIndex::decode(index.as_slice())?;let roles=roles(&index)?;let values:Vec<_>=roles.into_iter().map(|r|serde_json::json!([r.kind,r.owner,{"offset":r.span.offset,"nbytes":r.span.nbytes,"filename":r.span.filename}])).collect();Ok::<_,Box<dyn std::error::Error+Send+Sync>>(serde_json::to_string(&values)?)}).map_err(py_error)
}
#[pyfunction]
fn export_validate_layout(layout: String) -> PyResult<()> {
    let layout: Layout = serde_json::from_str(&layout).map_err(py_error)?;
    layout.validate().map_err(py_error)
}
#[pyfunction]
fn export_signature_message<'py>(
    py: Python<'py>,
    package: Vec<u8>,
    signer: Vec<u8>,
    kind: u32,
    body: Vec<u8>,
) -> Bound<'py, PyBytes> {
    PyBytes::new(
        py,
        &protection::signature_message(&package, &signer, kind, &body),
    )
}
#[pyfunction]
fn export_validate_publisher(namespace: String, source: Option<String>) -> PyResult<()> {
    match source {
        Some(source) => protection::validate_source(&namespace, &source),
        None => protection::validate_namespace(&namespace),
    }
    .map_err(py_error)
}
#[pyfunction]
fn export_validate_license(name: String, link: String) -> PyResult<()> {
    protection::validate_license(&wire::License { name, link }).map_err(py_error)
}

#[pyclass]
struct NativePackageFiles {
    files: std::sync::Mutex<Option<archive::PackageSources>>,
}
#[pymethods]
impl NativePackageFiles {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        Ok(Self {
            files: std::sync::Mutex::new(Some(
                archive::PackageSources::open(Path::new(&path)).map_err(py_error)?,
            )),
        })
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            self.files
                .lock()
                .map_err(|_| invalid("package reader lock poisoned"))?
                .take();
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
        })
        .map_err(py_error)
    }
    fn length(&self, py: Python<'_>, filename: String) -> PyResult<u64> {
        py.detach(|| {
            self.files
                .lock()
                .map_err(|_| invalid("package reader lock poisoned"))?
                .as_mut()
                .ok_or_else(|| invalid("package reader is closed"))?
                .logical_length(&filename)
        })
        .map_err(py_error)
    }
    fn read_at<'py>(
        &self,
        py: Python<'py>,
        filename: String,
        offset: u64,
        nbytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let data = py
            .detach(|| {
                let mut files = self
                    .files
                    .lock()
                    .map_err(|_| invalid("package reader lock poisoned"))?;
                let files = files
                    .as_mut()
                    .ok_or_else(|| invalid("package reader is closed"))?;
                let length = files.logical_length(&filename)?;
                let count = (nbytes as u64).min(length.saturating_sub(offset)) as usize;
                let mut bytes = vec![0; count];
                if count > 0 {
                    files.read_at(&filename, offset, &mut bytes)?;
                }
                Ok::<_, Box<dyn std::error::Error + Send + Sync>>(bytes)
            })
            .map_err(py_error)?;
        Ok(PyBytes::new(py, &data))
    }
}
#[pyfunction]
fn export_locate_sources(
    py: Python<'_>,
    path: String,
    offset: u64,
    nbytes: u64,
) -> PyResult<String> {
    py.detach(|| {
        let rank = read_sources(Path::new(&path))?;
        let sources = locate(
            &rank.payload,
            &wire::Span {
                offset,
                nbytes,
                filename: String::new(),
            },
        )?;
        let values: Vec<_> = sources
            .iter()
            .map(|s| (&s.path, s.offset, s.nbytes))
            .collect();
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(serde_json::to_string(&values)?)
    })
    .map_err(py_error)
}

#[pyfunction]
fn export_read_index_fd<'py>(
    py: Python<'py>,
    descriptor: i64,
) -> PyResult<(Bound<'py, PyBytes>, u64)> {
    let (bytes, base) = py
        .detach(|| {
            let (index, base) = read_index_container(Container::from_file(std::sync::Arc::new(
                duplicate_file(descriptor)?,
            ))?)?;
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>((index.encode_to_vec(), base))
        })
        .map_err(py_error)?;
    Ok((PyBytes::new(py, &bytes), base))
}

#[cfg(test)]
mod payload_export_tests {
    use super::super::{
        WeightPayloadGroup, WeightPayloadSource, WeightPayloadSources, WeightPayloadView,
    };
    use super::*;
    use helios_core::package::reader::Metadata;

    fn staged_rank(root: &Path) -> RankOutput {
        staged_rank_seeded(root, 0x11)
    }

    fn staged_rank_seeded(root: &Path, fill: u8) -> RankOutput {
        let weight = vec![fill; 64];
        let cubin = b"cubin-bytes".to_vec();
        let graph = b"graph-bytes".to_vec();
        for (name, bytes) in [
            ("storage-0.bin", &weight),
            ("cubin.bin", &cubin),
            ("graph.bin", &graph),
        ] {
            std::fs::write(root.join(name), bytes).unwrap();
        }
        let mut bundle = wire::BundleIndex {
            architecture: vec![12, 0],
            context: Some(Default::default()),
            allocations: vec![wire::Allocation {
                name: "device-0".into(),
                nbytes: weight.len() as u64,
                alignment: 256,
                file: Some("storage-0.bin".into()),
                data_spans: vec![wire::Span {
                    offset: 0,
                    nbytes: weight.len() as u64,
                    filename: String::new(),
                }],
                address: Some(0x1000),
                ..Default::default()
            }],
            modules: vec![wire::Module {
                name: "kernel".into(),
                file: "cubin.bin".into(),
                globals: vec![],
            }],
            functions: vec![wire::Function {
                module: "kernel".into(),
                symbol: "decode".into(),
                ..Default::default()
            }],
            graphs: vec![wire::GraphIndex {
                name: "decode".into(),
                resource: "graph.bin".into(),
                node_count: 1,
                functions: vec![0],
                ..Default::default()
            }],
            ..Default::default()
        };
        bundle.resources.insert(
            "storage-0.bin".into(),
            wire::Span {
                offset: 0,
                nbytes: weight.len() as u64,
                filename: String::new(),
            },
        );
        bundle.resources.insert(
            "cubin.bin".into(),
            wire::Span {
                offset: 4096,
                nbytes: cubin.len() as u64,
                filename: String::new(),
            },
        );
        bundle.resources.insert(
            "graph.bin".into(),
            wire::Span {
                offset: 8192,
                nbytes: graph.len() as u64,
                filename: String::new(),
            },
        );
        let payload = vec![
            FilePart {
                offset: 0,
                source: FileSource {
                    path: root.join("storage-0.bin"),
                    offset: 0,
                    nbytes: weight.len() as u64,
                },
            },
            FilePart {
                offset: 4096,
                source: FileSource {
                    path: root.join("cubin.bin"),
                    offset: 0,
                    nbytes: cubin.len() as u64,
                },
            },
            FilePart {
                offset: 8192,
                source: FileSource {
                    path: root.join("graph.bin"),
                    offset: 0,
                    nbytes: graph.len() as u64,
                },
            },
        ];
        RankOutput {
            hardware: Some("gb202".to_string()),
            payload_leases: Default::default(),
            index: wire::ModelIndex {
                version: 4,
                ranks: vec![bundle],
                contract: Some(Default::default()),
                frontend: Some(Default::default()),
                sampling_defaults: Some(Default::default()),
                ..Default::default()
            },
            payload,
            allocations: vec![],
            module_globals: vec![],
            external: vec![],
            staging: None,
            weight_payload: None,
        }
    }

    fn binding(owner: u32, readers: Vec<u32>) -> wire::PeerMemory {
        wire::PeerMemory {
            owner,
            address: 0x1000 + u64::from(owner) * 0x1000,
            nbytes: 4096,
            readers,
        }
    }

    #[test]
    fn peer_bindings_include_imports_consumed_by_the_selected_rank() {
        let peers = vec![
            binding(1, vec![0]),
            binding(2, vec![3]),
            binding(0, vec![1, 2]),
        ];
        let selected = selected_peers(&peers, 0);
        let owners: Vec<u32> = selected.iter().map(|peer| peer.owner).collect();
        assert_eq!(owners, vec![1, 0]);
        let selected = selected_peers(&peers, 3);
        assert_eq!(
            selected.iter().map(|peer| peer.owner).collect::<Vec<_>>(),
            vec![2]
        );
        assert!(selected_peers(&peers, 9).is_empty());
    }

    #[test]
    fn two_rank_peer_records_are_owner_inclusive_and_survive_publication() {
        let base = std::env::temp_dir().join(format!(
            "flare-peer-publication-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let sources = base.join("sources");
        std::fs::create_dir_all(&sources).unwrap();
        let mut first = staged_rank_seeded(&sources, 0x11);
        first.index.ranks[0].allocations[0].address = Some(0x1000);
        first.index.ranks[0].allocations.push(wire::Allocation {
            name: "peer-rank1".into(),
            nbytes: 64,
            alignment: 256,
            address: Some(0x2000),
            memory: wire::Memory::Peer as i32,
            ..Default::default()
        });
        let mut second = staged_rank_seeded(&sources, 0x22);
        second.index.ranks[0].allocations[0].address = Some(0x2000);
        second.index.ranks[0].allocations.push(wire::Allocation {
            name: "peer-rank0".into(),
            nbytes: 64,
            alignment: 256,
            address: Some(0x1000),
            memory: wire::Memory::Peer as i32,
            ..Default::default()
        });
        let peers = vec![
            wire::PeerMemory {
                owner: 0,
                address: 0x1000,
                nbytes: 64,
                readers: vec![0, 1],
            },
            wire::PeerMemory {
                owner: 1,
                address: 0x2000,
                nbytes: 64,
                readers: vec![0, 1],
            },
        ];
        let output = base.join("package");
        publish_payload(
            &output,
            &[first, second],
            peers,
            None,
            &publisher(),
            None,
            false,
            "directory",
        )
        .unwrap();
        let metadata = Metadata::open(&output).unwrap();
        let envelope = helios_core::package::protection::envelope(&metadata.envelope).unwrap();
        let mut rank_bodies = Vec::new();
        for signed in &envelope.ranks {
            rank_bodies.push(
                helios_core::package::schema::wire::RankBody::decode(signed.body.as_slice())
                    .unwrap(),
            );
        }
        rank_bodies.sort_by_key(|body| body.rank);
        assert_eq!(rank_bodies.len(), 2);
        for body in rank_bodies {
            let owners: Vec<u32> = body.peers.iter().map(|peer| peer.owner).collect();
            assert_eq!(owners, vec![0, 1]);
            for peer in &body.peers {
                assert!(peer.readers.contains(&peer.owner));
                assert_eq!(peer.readers, vec![0, 1]);
            }
            let peer_allocations: Vec<_> = body
                .allocations
                .iter()
                .filter(|allocation| allocation.memory == wire::Memory::Peer as i32)
                .map(|allocation| (allocation.address, allocation.nbytes))
                .collect();
            let expected_peer = if body.rank == 0 {
                (Some(0x2000), 64)
            } else {
                (Some(0x1000), 64)
            };
            assert_eq!(peer_allocations, vec![expected_peer]);
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn invalid_payload_paths_are_rejected_before_any_output() {
        use helios_core::package::export as payload_export;
        use helios_core::package::schema::wire as pwire;
        use helios_core::package::PayloadId;

        let base = std::env::temp_dir().join(format!(
            "flare-invalid-payload-path-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let source = base.join("source.bin");
        std::fs::write(&source, b"test").unwrap();
        let invalid = "weights/weights.tree.model.language_model.layers[16].mlp_hyper_connection.hc_norm.weight.safetensors";
        let payload = || payload_export::PublishPayload {
            role: pwire::PayloadRole::Weights,
            relative: invalid.into(),
            source: payload_export::pin_file(&source, 4).unwrap(),
            id: PayloadId::of(b"test"),
            tensor: None,
        };
        let publisher = publisher();

        for (label, rank_payloads, deployment_payloads) in [
            ("rank", vec![payload()], Vec::new()),
            ("deployment", Vec::new(), vec![payload()]),
        ] {
            let output = base.join(label);
            let request = payload_export::PublishRequest {
                package: pwire::PackageBody::default(),
                payloads: Vec::new(),
                deployments: vec![payload_export::PublishDeployment {
                    body: pwire::DeploymentBody {
                        deployment_id: "gb202-tp1".into(),
                        ..Default::default()
                    },
                    ranks: vec![payload_export::PublishRank {
                        body: pwire::RankBody::default(),
                        payloads: rank_payloads,
                        resources: Vec::new(),
                    }],
                    payloads: deployment_payloads,
                }],
                package_id: [1; 32],
                public_key: publisher.public_key().try_into().unwrap(),
                accepted: "example".into(),
                no_weights: false,
            };
            let error = payload_export::publish(
                &request,
                publisher.signing_key(),
                payload_export::Destination::Directory(&output),
            )
            .unwrap_err();
            assert!(error.to_string().contains(invalid), "{label}: {error}");
            assert!(
                error.to_string().contains("payload destination"),
                "{label}: {error}"
            );
            assert!(!output.exists(), "{label} created output");
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn group_coverage_accepts_multiple_pieces_and_rejects_partial_cover() {
        let span = |offset: u64, nbytes: u64| crate::export::WeightPayloadSourceSpan {
            allocation: "device-0".into(),
            destination_offset: offset,
            source_offset: offset,
            nbytes,
        };
        let pieces = vec![span(0, 8), span(8, 8), span(0x100, 8)];
        assert!(group_covers(&pieces, "device-0", 0, 16));
        assert!(group_covers(&pieces, "device-0", 4, 16));
        assert!(!group_covers(&pieces, "device-0", 0, 24));
        assert!(!group_covers(&pieces, "device-0", 0x100, 0x110));
        assert!(!group_covers(&pieces, "device-1", 0, 8));
    }

    fn publisher() -> protection::Publisher {
        protection::Publisher::from_seed(
            &[7u8; 32],
            "example".into(),
            "https://huggingface.co/example/model".into(),
            false,
            None,
        )
        .unwrap()
    }

    fn manifest_ids(root: &Path) -> (Vec<u8>, Vec<String>) {
        let metadata = Metadata::open(root).unwrap();
        let envelope = helios_core::package::protection::envelope(&metadata.envelope).unwrap();
        let ids = metadata
            .locations
            .locations
            .iter()
            .filter(|location| !location.file.is_empty())
            .map(|location| {
                location
                    .payload_id
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect()
            })
            .collect();
        (envelope.package_id, ids)
    }

    #[test]
    fn reuse_weights_publishes_cached_payloads_without_live_weight_sources() -> Result<()> {
        use crate::export::{
            CaptureRequest, WeightPayloadGroup, WeightPayloadSourceSpan, WeightPayloadSources,
        };
        use helios_core::package::{PayloadId, cache::Root};
        let directory = tempfile::tempdir()?;
        let root = Root::open(directory.path().join("cache"))?;
        assert!(super::super::weight_reuse::open_source(root.path(), "model", "[]").is_err());
        let stages = directory.path().join("stages");
        fs::create_dir_all(&stages)?;
        let mut rank = staged_rank(&stages);
        let rebuild = wire::RebuildManifest {
            version: 1,
            recipe: "fixture".into(),
            checkpoint_repository: "fixture/model".into(),
            checkpoint_revision: "revision-a".into(),
            tensor_parallel: 1,
            ..Default::default()
        };
        rank.index.rebuild = Some(rebuild.clone());
        rank.index.ranks[0].rebuild_weights = Some(wire::RebuildWeights {
            version: 1,
            tensors: (0..4)
                .map(|index| wire::RebuildTensor {
                    name: format!("weight-{index}"),
                    dtype: "uint8".into(),
                    element_bytes: 1,
                    shape: vec![16],
                    stride: vec![1],
                    storage: Some(wire::Reference {
                        allocation: "device-0".into(),
                        offset: index * 16,
                    }),
                    ..Default::default()
                })
                .collect(),
            unavailable: vec![],
        });
        let mut groups = Vec::new();
        let mut views = Vec::new();
        for index in 0..2u64 {
            let group_name = ["embedding", "layer-000"][index as usize];
            let bytes = vec![index as u8 + 1; 32];
            let id = PayloadId::of(&bytes);
            root.insert_payload(id, 32, std::io::Cursor::new(bytes))?;
            groups.push(WeightPayloadGroup {
                name: group_name.into(),
                storage: if index == 0 { "stable" } else { "tuned" }.into(),
                nbytes: 32,
                payload_id: id.as_bytes().to_vec(),
                spans: (0..2)
                    .map(|part| WeightPayloadSourceSpan {
                        allocation: "device-0".into(),
                        destination_offset: index * 32 + part * 16,
                        source_offset: part * 16,
                        nbytes: 16,
                    })
                    .collect(),
            });
            for part in 0..2 {
                views.push(serde_json::json!({
                    "name":format!("weight-{}",index * 2 + part),"group":group_name,
                    "storage":if index == 0 {"stable"} else {"tuned"},
                    "element_bytes":1,"shape":[16],"stride":[1],"address":0x1000 + index * 32 + part * 16,
                }));
            }
        }
        rank.weight_payload = Some(WeightPayloadSources {
            cached: Vec::new(),
            engram: None,
            groups: groups.clone(),
            views: views
                .iter()
                .cloned()
                .map(serde_json::from_value)
                .collect::<std::result::Result<_, _>>()?,
            initial_state: WeightPayloadGroup {
                name: "initial-state".into(),
                storage: "state".into(),
                ..Default::default()
            },
        });
        let first = publish_payload(
            Path::new(""),
            &[rank.clone()],
            vec![],
            None,
            &publisher(),
            Some(root.path().to_path_buf()),
            false,
            "directory",
        )?;
        let signer = publisher();
        let profiles = serde_json::json!([{"namespace":signer.namespace,"name":"fixture",
            "public_key":signer.public_key(),"development":false}])
        .to_string();
        let source = super::super::weight_reuse::open_source(root.path(), "model", &profiles)?;
        let mut metadata = serde_json::to_value(&source.source)?;
        metadata["rebuild_manifest"] = serde_json::json!({"version":1,"recipe":"fixture",
            "checkpoint_repository":"fixture/model","checkpoint_revision":"revision-a","tensor_parallel":1});
        // Live sources contain only fresh state; the weight bytes cannot be read here.
        fs::write(stages.join("storage-0.bin"), vec![0x77u8; 32])?;
        let bundle = &mut rank.index.ranks[0];
        bundle.allocations[0].name = "moved".into();
        bundle.allocations[0].nbytes = 512;
        bundle.allocations[0].address = Some(0x2000);
        bundle.allocations[0].data_spans = vec![wire::Span {
            offset: 320,
            nbytes: 32,
            ..Default::default()
        }];
        bundle.resources.get_mut("storage-0.bin").unwrap().nbytes = 32;
        for tensor in &mut bundle.rebuild_weights.as_mut().unwrap().tensors {
            let storage = tensor.storage.as_mut().unwrap();
            storage.allocation = "moved".into();
            storage.offset += 256;
        }
        rank.payload[0].source.nbytes = 32;
        rank.allocations = vec![super::super::Allocation {
            name: "moved".into(),
            address: 0x2000,
            nbytes: 512,
            alignment: 256,
            initialize: "snapshot".into(),
            relocations: vec![],
            preserve_address: true,
            memory: "device".into(),
            snapshot_spans: Some(vec![(256, 96)]),
            sizing: None,
        }];
        rank.staging = Some(stages.clone());
        for (index, view) in views.iter_mut().enumerate() {
            view["address"] = (0x2100 + index * 16).into();
        }
        let mut request: CaptureRequest = serde_json::from_value(serde_json::json!({
            "path":stages.join("rank.json"),"context":0,"graphs":[],"allocations":[],
            "numeric":[[0x2100,0x2140]],
            "weight_payload":{"version":1,"views":views},
            "options":{"cache":root.path(),"no_weights":true,"reuse_weights":metadata},
        }))?;
        let reused =
            super::super::capture::resolve_staged_weight_payload(&request, &mut rank)?.unwrap();
        assert_eq!(reused.groups.len(), 2);
        for (group, previous) in reused.groups.iter().zip(&groups) {
            assert_eq!(group.payload_id, previous.payload_id);
            assert_eq!(
                group.spans[0].destination_offset,
                previous.spans[0].destination_offset + 256
            );
            assert_eq!(group.spans[0].allocation, "moved");
            assert_eq!(group.spans[0].nbytes, 32);
        }
        let declaration: super::super::weight_payload::WeightPayloadDeclaration =
            serde_json::from_value(request.weight_payload.clone().unwrap())?;
        let catalog = super::super::weight_payload::extract(&declaration.into_request(
            super::super::capture::weight_allocations(&rank.allocations),
            Vec::new(),
        ))?;
        for fault in ["gap", "group", "allocation", "storage"] {
            let mut invalid_catalog = catalog.clone();
            match fault {
                "gap" => invalid_catalog.groups[0].spans[0].nbytes -= 1,
                "group" => {
                    let span = invalid_catalog.groups[0].spans.pop().unwrap();
                    invalid_catalog
                        .groups
                        .push(super::super::weight_payload::WeightGroupPayload {
                            name: "other".into(),
                            storage: super::super::weight_payload::StorageClass::Stable,
                            nbytes: span.nbytes,
                            spans: vec![span],
                        });
                }
                "allocation" => invalid_catalog.groups[0].spans[0].allocation = "other".into(),
                "storage" => {
                    invalid_catalog.groups[0].storage =
                        super::super::weight_payload::StorageClass::Tuned
                }
                _ => unreachable!(),
            }
            let error = request
                .options
                .reuse_weights
                .as_ref()
                .unwrap()
                .groups(&request, &rank, &invalid_catalog, &reused)
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("not covered by one declared weight group"),
                "{fault}: {error}"
            );
        }
        assert!(!reused.initial_state.payload_id.is_empty());
        assert_eq!(
            super::super::weight_reuse::numeric_ranges(&request, &rank.allocations)?,
            vec![(0x2100, 0x2140)]
        );
        rank.weight_payload = Some(reused);
        let watched: Vec<_> = groups.iter().map(|group| {
            root.payload(PayloadId::from_bytes(group.payload_id.as_slice().try_into().unwrap()))
                .unwrap().unwrap().path().to_path_buf()
        }).collect();
        let publish = || publish_payload(
            Path::new(""),
            &[rank.clone()],
            vec![],
            None,
            &signer,
            Some(root.path().to_path_buf()),
            true,
            "directory",
        );
        #[cfg(unix)]
        let (second, bytes_read) = helios_core::package::container::count_payload_reads(&watched, publish);
        #[cfg(not(unix))]
        let second = publish();
        let second = second?;
        #[cfg(unix)]
        assert_eq!(bytes_read, 0, "cached weights must not be read during publication");
        let first_entry = root.entry(
            first["model"].as_str().unwrap(),
            first["entry"].as_str().unwrap(),
        )?;
        let second_entry = root.entry(
            second["model"].as_str().unwrap(),
            second["entry"].as_str().unwrap(),
        )?;
        let first_meta = Metadata::open(first_entry.directory())?;
        let second_meta = Metadata::open(second_entry.directory())?;
        for group in &groups {
            let locate = |metadata: &Metadata| {
                metadata
                    .locations
                    .locations
                    .iter()
                    .find(|location| location.payload_id == group.payload_id)
                    .unwrap()
                    .file
                    .clone()
            };
            let left = first_entry.directory().join(locate(&first_meta));
            let right = second_entry.directory().join(locate(&second_meta));
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                assert_eq!(fs::metadata(left)?.ino(), fs::metadata(right)?.ino());
            }
            #[cfg(not(unix))]
            {
                assert_eq!(fs::read(left)?, fs::read(right)?);
            }
        }
        // Reuse rejects a changed checkpoint or a changed tuned layout.
        let original = request.options.reuse_weights.clone();
        let mut changed = metadata.clone();
        changed["rebuild_manifest"]["checkpoint_revision"] = "revision-b".into();
        request.options.reuse_weights = Some(serde_json::from_value(changed)?);
        assert!(super::super::capture::resolve_staged_weight_payload(&request, &mut rank).is_err());
        request.options.reuse_weights = original;
        rank.index.ranks[0]
            .rebuild_weights
            .as_mut()
            .unwrap()
            .tensors[3]
            .stride[0] = 2;
        assert!(super::super::capture::resolve_staged_weight_payload(&request, &mut rank).is_err());
        Ok(())
    }
    #[test]
    fn cache_only_export_commits_an_entry_and_no_weights_reuses_payloads() {
        use helios_core::package::cache::Root;
        let base = std::env::temp_dir().join(format!("flare-adapter-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let cache = base.join("cache");
        let sources = base.join("sources");
        std::fs::create_dir_all(&sources).unwrap();
        let first = publish_payload(
            Path::new(""),
            &[staged_rank(&sources)],
            vec![],
            None,
            &publisher(),
            Some(cache.clone()),
            false,
            "directory",
        )
        .unwrap();
        let model = first["model"].as_str().unwrap().to_string();
        let entry = first["entry"].as_str().unwrap().to_string();
        let root = Root::open(&cache).unwrap();
        let committed = root.entry(&model, &entry).unwrap();
        assert!(committed.directory().join("model.safetensors").exists());
        assert!(entry.starts_with("gb202-tp1-"), "{entry}");
        assert_eq!(entry.len(), "gb202-tp1-".len() + 12, "{entry}");
        assert_eq!(root.resolve(&model).unwrap().unwrap().id(), entry.as_str());

        // Identical weight bytes hit the cache; executables are re-sealed under
        // the fresh package identity.
        let second = publish_payload(
            Path::new(""),
            &[staged_rank(&sources)],
            vec![],
            None,
            &publisher(),
            Some(cache.clone()),
            true,
            "directory",
        )
        .unwrap();
        let second_entry = second["entry"].as_str().unwrap().to_string();
        assert!(
            second_entry.starts_with(&format!("{entry}-2")),
            "{second_entry}"
        );
        assert_eq!(
            root.resolve(&model).unwrap().unwrap().id(),
            second_entry.as_str()
        );

        // Same bytes, same content ID: even an initial-state object must reject
        // a corrupted cached copy rather than silently replacing it.
        for blob in std::fs::read_dir(cache.join("blobs")).unwrap().flatten() {
            let path = blob.path();
            let metadata = std::fs::metadata(&path).unwrap();
            if !metadata.is_file() {
                continue;
            }
            let length = metadata.len();
            std::fs::remove_file(&path).unwrap();
            std::fs::write(&path, vec![0x5au8; length as usize]).unwrap();
        }
        assert!(
            publish_payload(
                Path::new(""),
                &[staged_rank(&sources)],
                vec![],
                None,
                &publisher(),
                Some(cache.clone()),
                true,
                "directory"
            )
            .is_err()
        );
        // A failed export leaves the previously selected default untouched.
        assert_eq!(
            root.resolve(&model).unwrap().unwrap().id(),
            second_entry.as_str()
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn cached_declared_groups_use_object_offsets_and_span_initializers() {
        use crate::export::{WeightPayloadGroup, WeightPayloadSourceSpan, WeightPayloadSources};
        use helios_core::package::PayloadId;
        use helios_core::package::cache::Root;
        let base =
            std::env::temp_dir().join(format!("flare-declared-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let cache = base.join("cache");
        std::fs::create_dir_all(&base).unwrap();
        let root = Root::open(&cache).unwrap();
        // Two nonzero owner blocks with an elided zero block between their destinations.
        let stored = b"firstblksecondblk".to_vec();
        let id = PayloadId::of(&stored);
        root.insert_payload(
            id,
            stored.len() as u64,
            std::io::Cursor::new(stored.clone()),
        )
        .unwrap();
        let group = WeightPayloadGroup {
            name: "layer-0".into(),
            storage: "stable".into(),
            nbytes: stored.len() as u64,
            spans: vec![
                WeightPayloadSourceSpan {
                    allocation: "device-0".into(),
                    destination_offset: 0,
                    source_offset: 0,
                    nbytes: 8,
                },
                WeightPayloadSourceSpan {
                    allocation: "device-0".into(),
                    destination_offset: 16,
                    source_offset: 8,
                    nbytes: 9,
                },
            ],
            payload_id: id.as_bytes().to_vec(),
        };
        let sources = base.join("sources");
        std::fs::create_dir_all(&sources).unwrap();
        let mut rank = staged_rank(&sources);
        rank.index.ranks[0].allocations[0].nbytes = 4096;
        rank.weight_payload = Some(WeightPayloadSources {
            cached: Vec::new(),
            engram: None,
            groups: vec![group],
            initial_state: WeightPayloadGroup {
                name: "initial-state".into(),
                storage: "stable".into(),
                nbytes: 0,
                spans: vec![],
                payload_id: vec![],
            },
            views: vec![],
        });
        let published = publish_payload(
            Path::new(""),
            &[rank.clone()],
            vec![],
            None,
            &publisher(),
            Some(cache.clone()),
            false,
            "directory",
        )
        .unwrap();
        let entry = published["entry"].as_str().unwrap().to_string();
        let model = published["model"].as_str().unwrap().to_string();
        let committed = root.entry(&model, &entry).unwrap();
        let metadata = Metadata::open(committed.directory()).unwrap();
        assert!(
            metadata
                .locations
                .locations
                .iter()
                .any(|location| location.payload_id == id.as_bytes())
        );
        let envelope = helios_core::package::protection::envelope(&metadata.envelope).unwrap();
        let deployment = helios_core::package::schema::wire::DeploymentBody::decode(
            envelope.deployments[0].body.as_slice(),
        )
        .unwrap();
        let rank_body = envelope
            .ranks
            .iter()
            .find(|rank| rank.deployment == 0)
            .unwrap();
        let body = helios_core::package::schema::wire::RankBody::decode(rank_body.body.as_slice())
            .unwrap();
        assert!(
            deployment
                .payloads
                .iter()
                .any(|payload| payload.id == id.as_bytes())
        );
        // Exact initializers: destination offsets keep the gapped layout, and slice offsets are
        // object-relative stored offsets rather than accumulated.
        assert_eq!(body.initializers.len(), 2);
        let slices: Vec<(u64, u64, u64)> = body
            .initializers
            .iter()
            .map(|initializer| {
                let slice = initializer.source.as_ref().unwrap().slice.as_ref().unwrap();
                (initializer.destination_offset, slice.offset, slice.nbytes)
            })
            .collect();
        assert_eq!(slices, vec![(0, 0, 8), (16, 8, 9)]);
        // no_weights reuses the same cached object without any staged weight file.
        let reused = publish_payload(
            Path::new(""),
            &[rank],
            vec![],
            None,
            &publisher(),
            Some(cache.clone()),
            true,
            "directory",
        )
        .unwrap();
        assert!(
            reused["entry"]
                .as_str()
                .unwrap()
                .starts_with(&format!("{entry}-2"))
        );
        // A declared payload absent from the cache is a hard error, never a re-materialization.
        let empty = Root::open(base.join("empty-cache")).unwrap();
        let _ = empty;
        let mut missing = staged_rank(&sources);
        missing.weight_payload = Some(WeightPayloadSources {
            cached: Vec::new(),
            engram: None,
            groups: vec![WeightPayloadGroup {
                name: "layer-0".into(),
                storage: "stable".into(),
                nbytes: stored.len() as u64,
                spans: vec![WeightPayloadSourceSpan {
                    allocation: "device-0".into(),
                    destination_offset: 0,
                    source_offset: 0,
                    nbytes: stored.len() as u64,
                }],
                payload_id: PayloadId::of(b"absent payload").as_bytes().to_vec(),
            }],
            initial_state: WeightPayloadGroup {
                name: "initial-state".into(),
                storage: "stable".into(),
                nbytes: 0,
                spans: vec![],
                payload_id: vec![],
            },
            views: vec![],
        });
        assert!(
            publish_payload(
                Path::new(""),
                &[missing],
                vec![],
                None,
                &publisher(),
                Some(cache),
                false,
                "directory"
            )
            .is_err()
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn failed_standalone_export_preserves_the_default_and_removes_the_new_entry() {
        use helios_core::package::cache::Root;
        let base = std::env::temp_dir().join(format!("flare-default-fail-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let cache = base.join("cache");
        let sources = base.join("sources");
        std::fs::create_dir_all(&sources).unwrap();
        let first = publish_payload(
            Path::new(""),
            &[staged_rank(&sources)],
            vec![],
            None,
            &publisher(),
            Some(cache.clone()),
            false,
            "directory",
        )
        .unwrap();
        let model = first["model"].as_str().unwrap().to_string();
        let first_entry = first["entry"].as_str().unwrap().to_string();
        let root = Root::open(&cache).unwrap();
        assert_eq!(root.resolve(&model).unwrap().unwrap().id(), first_entry);
        // A conflicting standalone destination fails before the default is written.
        let conflict = base.join("occupied");
        std::fs::create_dir_all(&conflict).unwrap();
        let error = publish_payload(
            &conflict,
            &[staged_rank(&sources)],
            vec![],
            None,
            &publisher(),
            Some(cache.clone()),
            false,
            "directory",
        )
        .unwrap_err();
        assert!(error.to_string().contains("exists"), "{error}");
        // Previous default is unchanged and the just-created cache entry was removed.
        assert_eq!(root.resolve(&model).unwrap().unwrap().id(), first_entry);
        assert_eq!(root.versions(&model).unwrap(), vec![first_entry]);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn combined_frontend_asset_uses_combined_payload_coordinates() {

        let base = std::env::temp_dir().join(format!("flare-asset-shift-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("sources")).unwrap();
        let sources = base.join("sources");
        let mut first = staged_rank_seeded(&sources, 0x11);
        let second = staged_rank_seeded(&sources, 0x22);
        let frontend = b"shifted-frontend-asset".to_vec();
        std::fs::write(sources.join("frontend.bin"), &frontend).unwrap();
        let frontend_sha_digest = ring::digest::digest(&ring::digest::SHA256, &frontend);
        let frontend_sha = frontend_sha_digest
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        first.index.assets.push(wire::Asset {
            name: "tokenizer".into(),
            payload: Some(wire::Span {
                offset: 12288,
                nbytes: frontend.len() as u64,
                filename: String::new(),
            }),
            sha256: frontend_sha.clone(),
        });
        first.payload.push(FilePart {
            offset: 12288,
            source: FileSource {
                path: sources.join("frontend.bin"),
                offset: 0,
                nbytes: frontend.len() as u64,
            },
        });
        let output = base.join("package");
        publish_payload(
            &output,
            &[first, second],
            vec![],
            None,
            &publisher(),
            None,
            false,
            "directory",
        )
        .unwrap();
        let metadata = Metadata::open(&output).unwrap();
        let source_publisher = publisher();
        let signer = helios_core::runtime_package::Publisher {
            name: "test".into(),
            namespace: source_publisher.namespace.clone(),
            public_key: source_publisher.public_key().try_into().unwrap(),
            development: false,
        };
        let verified = helios_core::package::protection::verify(
            &metadata.envelope,
            std::slice::from_ref(&signer),
            helios_core::package::protection::Selection::Deployment {
                deployment: 0,
                rank: None,
            },
        )
        .unwrap();
        let asset = verified
            .package()
            .assets
            .iter()
            .find(|asset| asset.name == "tokenizer")
            .unwrap();
        let slice = asset.payload.as_ref().unwrap().slice.as_ref().unwrap();
        let location = metadata
            .locations
            .locations
            .iter()
            .find(|location| location.payload_id == slice.payload_id)
            .unwrap();
        let stored = if location.file.is_empty() {
            metadata.primary().tensor(&location.tensor).unwrap()
        } else {
            let container =
                helios_core::package::container::Container::open(&output.join(&location.file))
                    .unwrap();
            container.tensor(&location.tensor).unwrap()
        };
        let mut bytes = vec![0; stored.len() as usize];
        stored.read_at(0, &mut bytes).unwrap();
        assert_eq!(bytes, frontend);
        assert_eq!(
            slice.payload_id,
            helios_core::package::PayloadId::of(&bytes).as_bytes()
        );
        assert_eq!(asset.sha256, frontend_sha_digest.as_ref());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn published_catalog_accepts_rank3_source_for_flat_owner() {
        let base = std::env::temp_dir().join(format!("flare-rank3-export-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("sources")).unwrap();
        let sources = base.join("sources");
        let mut rank = staged_rank_seeded(&sources, 0x33);
        rank.index.ranks[0].rebuild_weights = Some(wire::RebuildWeights {
            version: 1,
            tensors: vec![wire::RebuildTensor {
                name: "experts.w13".into(),
                dtype: "uint8".into(),
                element_bytes: 1,
                shape: vec![64],
                stride: vec![1],
                storage: Some(wire::Reference {
                    allocation: "device-0".into(),
                    offset: 0,
                }),
                ..Default::default()
            }],
            unavailable: vec![],
        });
        rank.weight_payload = Some(WeightPayloadSources {
            cached: Vec::new(),
            engram: None,
            groups: vec![],
            initial_state: WeightPayloadGroup {
                name: "initial-state".into(),
                storage: "stable".into(),
                nbytes: 0,
                spans: vec![],
                payload_id: vec![],
            },
            views: vec![WeightPayloadView {
                name: "experts.w13".into(),
                group: "experts".into(),
                storage: "stable".into(),
                sources: vec![WeightPayloadSource {
                    tensor: "experts.0.up_proj.weight".into(),
                    ranges: vec![(0, 8), (0, 4), (0, 2)],
                    concatenation_axis: Some(0),
                    pieces: vec![],
                    piece_encoding: String::new(),
                }],
                packing: "iq2_xxs".into(),
                packing_parameters: None,
                statistics_domain: "tensor".into(),
            }],
        });
        let output = base.join("package");
        let published = publish_payload(
            &output,
            &[rank],
            vec![],
            None,
            &publisher(),
            None,
            false,
            "directory",
        )
        .unwrap();
        assert!(published.is_object());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn reuse_engram_hardlinks_shared_and_partitioned_embedding_planes() -> Result<()> {
        use crate::export::{CaptureRequest, WeightPayloadGroup, WeightPayloadSources};
        use helios_core::package::cache::Root;
        for partitioned in [false, true] {
            let directory = tempfile::tempdir()?;
            let stages = directory.path().join("stages");
            fs::create_dir_all(&stages)?;
            let root = Root::open(directory.path().join("cache"))?;
            let mut rank = staged_rank(&stages);
            let world = if partitioned { 2 } else { 1 };
            rank.index.rebuild = Some(wire::RebuildManifest {
                version: 1,
                recipe: "fixture".into(),
                checkpoint_repository: "fixture/model".into(),
                checkpoint_revision: "revision-a".into(),
                tensor_parallel: world,
                ..Default::default()
            });
            let mut table = wire::EngramTable {
                layer: 1,
                rows: world as u64,
                embedding_width: 160,
                scale_block: 16,
                hash: Some(wire::EngramHash {
                    method: wire::engram_hash::Method::EosXorModulo as i32,
                    ..Default::default()
                }),
                weight_buffer: "weights".into(),
                scale_buffer: "scales".into(),
                hash_buffer: "hashes".into(),
                request_buffer: "requests".into(),
                ready_event: "ready".into(),
                encoding: wire::engram_table::Encoding::Nvfp4E4m3Global as i32,
                global_scale: 1.0,
                ..Default::default()
            };
            let mut cursor = 12288;
            for shard in 0..world {
                let mut planes = Vec::new();
                for (plane, length) in [("weights", 80), ("scales", 10)] {
                    let path = stages.join(format!("engram-{shard}-{plane}.bin"));
                    fs::write(&path, vec![shard as u8 + 1; length])?;
                    rank.payload.push(FilePart {
                        offset: cursor,
                        source: FileSource {
                            path,
                            offset: 0,
                            nbytes: length as u64,
                        },
                    });
                    planes.push(wire::Span {
                        offset: cursor,
                        nbytes: length as u64,
                        ..Default::default()
                    });
                    cursor += length as u64;
                }
                if partitioned {
                    table.shards.push(wire::EngramTableShard {
                        rank: shard,
                        first_row: u64::from(shard),
                        rows: 1,
                        weights: Some(planes[0].clone()),
                        scales: Some(planes[1].clone()),
                    });
                } else {
                    table.weights = Some(planes[0].clone());
                    table.scales = Some(planes[1].clone());
                }
            }
            rank.index.engram = Some(wire::Engram {
                token_map: vec![],
                pad_token_id: 1,
                compressed_vocabulary: 100,
                tables: vec![table],
            });
            let mut ranks = vec![rank.clone()];
            if partitioned {
                let peer = stages.join("peer");
                fs::create_dir(&peer)?;
                ranks.push(staged_rank(&peer));
            }
            let signer = publisher();
            let first = publish_payload(
                Path::new(""),
                &ranks,
                vec![],
                None,
                &signer,
                Some(root.path().to_path_buf()),
                false,
                "directory",
            )?;
            let profiles = serde_json::json!([{"namespace":signer.namespace,"name":"fixture",
                "public_key":signer.public_key(),"development":false}])
            .to_string();
            let source = super::super::weight_reuse::open_source(root.path(), "model", &profiles)?;
            let mut source_metadata = serde_json::to_value(&source.source)?;
            source_metadata["rebuild_manifest"] = serde_json::json!({"version":1,"recipe":"fixture",
                "checkpoint_repository":"fixture/model","checkpoint_revision":"revision-a","tensor_parallel":world});
            let request: CaptureRequest = serde_json::from_value(serde_json::json!({
                "path":stages.join("rank.json"),"context":0,"world_size":world,"graphs":[],"allocations":[],
                "options":{"cache":root.path(),"reuse_weights":source_metadata},
            }))?;
            let mut data = serde_json::to_value(rank.index.engram.as_ref().unwrap())?;
            data["tables"][0]["weight_buffer"] = "rebound_weights".into();
            let set_planes = |value: &mut serde_json::Value| {
                for (plane, nbytes) in [("weights", 80), ("scales", 10)] {
                    value[plane] = serde_json::json!([{"path":stages.join("absent-checkpoint"),"offset":0,"nbytes":nbytes}]);
                }
            };
            if partitioned {
                for shard in data["tables"][0]["shards"].as_array_mut().unwrap() {
                    set_planes(shard);
                }
            } else {
                set_planes(&mut data["tables"][0]);
            }
            let reused = request
                .options
                .reuse_weights
                .as_ref()
                .unwrap()
                .engram(&request, &rank, &data)?;
            assert_eq!(reused.tables[0].weight_buffer, "rebound_weights");
            rank.index.engram = None;
            for part in rank.payload.drain(3..) {
                fs::remove_file(part.source.path)?;
            }
            rank.weight_payload = Some(WeightPayloadSources {
                engram: Some(reused.clone()),
                initial_state: WeightPayloadGroup {
                    name: "initial-state".into(),
                    ..Default::default()
                },
                ..Default::default()
            });
            ranks[0] = rank.clone();
            let watched: Vec<_> = super::super::weight_reuse::engram_uses(&reused).iter().map(|source| {
                let id = helios_core::package::PayloadId::from_bytes(source.slice.as_ref().unwrap().payload_id.as_slice().try_into().unwrap());
                root.payload(id).unwrap().unwrap().path().to_path_buf()
            }).collect();
            let publish = || publish_payload(
                Path::new(""),
                &ranks,
                vec![],
                None,
                &signer,
                Some(root.path().to_path_buf()),
                true,
                "directory",
            );
            #[cfg(unix)]
            let (second, bytes_read) = helios_core::package::container::count_payload_reads(&watched, publish);
            #[cfg(not(unix))]
            let second = publish();
            let second = second?;
            #[cfg(unix)]
            assert_eq!(bytes_read, 0, "cached Engram planes must not be read during publication");
            let first_entry = root.entry(
                first["model"].as_str().unwrap(),
                first["entry"].as_str().unwrap(),
            )?;
            let second_entry = root.entry(
                second["model"].as_str().unwrap(),
                second["entry"].as_str().unwrap(),
            )?;
            let first_meta = Metadata::open(first_entry.directory())?;
            let second_meta = Metadata::open(second_entry.directory())?;
            for source in super::super::weight_reuse::engram_uses(&reused) {
                let id = &source.slice.as_ref().unwrap().payload_id;
                let locate = |metadata: &Metadata| {
                    metadata
                        .locations
                        .locations
                        .iter()
                        .find(|location| &location.payload_id == id)
                        .unwrap()
                        .file
                        .clone()
                };
                let left = first_entry.directory().join(locate(&first_meta));
                let right = second_entry.directory().join(locate(&second_meta));
                #[cfg(unix)]
                {
                    use std::os::unix::fs::MetadataExt;
                    assert_eq!(fs::metadata(left)?.ino(), fs::metadata(right)?.ino());
                }
                #[cfg(not(unix))]
                {
                    assert_eq!(fs::read(left)?, fs::read(right)?);
                }
            }
            data["tables"][0]["global_scale"] = 2.0.into();
            assert!(
                request
                    .options
                    .reuse_weights
                    .as_ref()
                    .unwrap()
                    .engram(&request, &rank, &data)
                    .is_err()
            );
        }
        Ok(())
    }
    #[test]
    fn published_engram_uses_weight_roles_for_empty_token_map() {
        let base = std::env::temp_dir().join(format!("flare-engram-roles-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("sources")).unwrap();
        let sources = base.join("sources");
        let mut rank = staged_rank_seeded(&sources, 0x44);
        let weights = vec![0u8; 80];
        let scales = vec![0u8; 10];
        std::fs::write(sources.join("engram-w.bin"), &weights).unwrap();
        std::fs::write(sources.join("engram-s.bin"), &scales).unwrap();
        rank.index.engram = Some(wire::Engram {
            token_map: vec![],
            pad_token_id: 1,
            compressed_vocabulary: 100,
            tables: vec![wire::EngramTable {
                layer: 1, rows: 1, embedding_width: 160, scale_block: 16,
                weights: Some(wire::Span { offset: 12288, nbytes: 80, filename: String::new() }),
                scales: Some(wire::Span { offset: 12368, nbytes: 10, filename: String::new() }),
                hash: Some(wire::EngramHash {
                    method: wire::engram_hash::Method::EosXorModulo as i32,
                    orders: vec![], multipliers: vec![], primes: vec![], offsets: vec![],
                }),
                weight_buffer: "ple_weights".into(), scale_buffer: "ple_scales".into(),
                hash_buffer: "ple_hashes".into(), request_buffer: "ple_requests".into(),
                ready_event: "ple_ready".into(),
                encoding: wire::engram_table::Encoding::Nvfp4E4m3Global as i32,
                global_scale: 1.0,
                ..Default::default()
            }],
        });
        rank.payload.push(FilePart { offset: 12288, source: FileSource {
            path: sources.join("engram-w.bin"), offset: 0, nbytes: 80 } });
        rank.payload.push(FilePart { offset: 12368, source: FileSource {
            path: sources.join("engram-s.bin"), offset: 0, nbytes: 10 } });
        let output = base.join("package");
        publish_payload(&output, &[rank], vec![], None, &publisher(), None, false, "directory").unwrap();
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn payload_source_pins_exact_file_range_without_staging() {
        use helios_core::package::PayloadId;
        let base = tempfile::tempdir().unwrap();
        let path = base.path().join("checkpoint.bin");
        fs::write(&path, b"prefix-weights-suffix").unwrap();
        let parts = [FilePart {
            offset: 4096,
            source: FileSource {
                path: path.clone(),
                offset: 3,
                nbytes: 14,
            },
        }];
        let span = wire::Span {
            offset: 4100,
            nbytes: 7,
            ..Default::default()
        };
        let staging = base.path().join("absent-staging");
        let (id, nbytes, source) = payload_source(&parts, &span, &staging, "unused").unwrap();
        assert_eq!(id, PayloadId::of(b"weights"));
        assert_eq!(nbytes, 7);
        assert_eq!(source.offset(), 7);
        assert!(!staging.exists());
        let mut bytes = [0; 7];
        source.read_at(0, &mut bytes).unwrap();
        assert_eq!(&bytes, b"weights");

        fs::rename(&path, base.path().join("retained.bin")).unwrap();
        fs::write(&path, b"replacement").unwrap();
        assert_eq!(source.digest().unwrap(), id);
        fs::write(base.path().join("retained.bin"), b"prefix-changed-suffix").unwrap();
        assert_ne!(source.digest().unwrap(), id);
    }

    #[test]
    fn payload_source_joins_fragmented_ranges() {
        use helios_core::package::PayloadId;
        let base = tempfile::tempdir().unwrap();
        let first = base.path().join("first.bin");
        let second = base.path().join("second.bin");
        fs::write(&first, b"xxabcdyy").unwrap();
        fs::write(&second, b"zefghzz").unwrap();
        let parts = [
            FilePart {
                offset: 100,
                source: FileSource {
                    path: first,
                    offset: 2,
                    nbytes: 4,
                },
            },
            FilePart {
                offset: 104,
                source: FileSource {
                    path: second,
                    offset: 1,
                    nbytes: 4,
                },
            },
        ];
        let span = wire::Span {
            offset: 102,
            nbytes: 4,
            ..Default::default()
        };
        let (id, nbytes, source) = payload_source(&parts, &span, base.path(), "joined").unwrap();
        assert_eq!(id, PayloadId::of(b"cdef"));
        assert_eq!(nbytes, 4);
        assert_eq!(fs::read(base.path().join("joined")).unwrap(), b"cdef");
        let mut bytes = [0; 4];
        source.read_at(0, &mut bytes).unwrap();
        assert_eq!(&bytes, b"cdef");
    }

    #[test]
    fn plane_sources_keep_declared_order_and_report_failures() {
        use helios_core::package::PayloadId;
        let base = std::env::temp_dir().join(format!("flare-plane-sources-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let staging = base.join("staging");
        fs::create_dir_all(&staging).unwrap();
        let small = vec![0x01; 16];
        let large = vec![0x02; 4096];
        let split_a = vec![0x03; 100];
        let split_b = vec![0x04; 50];
        let mut payload = Vec::new();
        let mut cursor = 0u64;
        for (name, bytes) in [("small", &small), ("large", &large), ("split-a", &split_a), ("split-b", &split_b)] {
            let path = base.join(name);
            fs::write(&path, bytes).unwrap();
            payload.push(FilePart {
                offset: cursor,
                source: FileSource { path, offset: 0, nbytes: bytes.len() as u64 },
            });
            cursor += bytes.len() as u64;
        }
        let span = |offset, nbytes| wire::Span { offset, nbytes, filename: String::new() };
        let small_span = span(0, 16);
        let large_span = span(16, 4096);
        // Crosses two source files, so it is materialized into staging rather than pinned.
        let split_span = span(16 + 4096, 150);
        let planes = vec![
            (&small_span, "small.payload".to_string()),
            (&split_span, "split.payload".to_string()),
            (&large_span, "large.payload".to_string()),
        ];
        // Planes are scheduled largest first but returned in declared order.
        let sources = plane_sources(&payload, &planes, &staging).unwrap();
        let split = [split_a, split_b].concat();
        assert_eq!(sources.len(), 3);
        for ((id, nbytes, _), bytes) in sources.iter().zip([&small, &split, &large]) {
            assert_eq!(*id, PayloadId::of(bytes));
            assert_eq!(*nbytes, bytes.len() as u64);
        }
        let beyond = span(cursor, 8);
        let planes = vec![(&large_span, "a.payload".to_string()), (&beyond, "b.payload".to_string())];
        assert!(plane_sources(&payload, &planes, &staging).is_err());
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn package_engram_preserves_shared_and_partitioned_plane_layouts() {
        use helios_core::package::schema::wire as pwire;
        let base = std::env::temp_dir().join(format!(
            "flare-engram-partitioned-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("sources")).unwrap();
        let sources = base.join("sources");
        let shared_weights = vec![0x11; 80];
        let shared_scales = vec![0x22; 10];
        let shard0_weights = vec![0x33; 256];
        let shard0_scales = vec![0x44; 8];
        let shard1_weights = vec![0x55; 256];
        let shard1_scales = vec![0x66; 8];
        for (name, bytes) in [
            ("shared-w.bin", &shared_weights),
            ("shared-s.bin", &shared_scales),
            ("shard0-w.bin", &shard0_weights),
            ("shard0-s.bin", &shard0_scales),
            ("shard1-w.bin", &shard1_weights),
            ("shard1-s.bin", &shard1_scales),
        ] {
            fs::write(sources.join(name), bytes).unwrap();
        }
        let mut rank = staged_rank_seeded(&sources, 0x77);
        rank.index.engram = Some(wire::Engram {
            token_map: vec![1, 2],
            pad_token_id: 0,
            compressed_vocabulary: 4,
            tables: vec![
                wire::EngramTable {
                    layer: 1,
                    rows: 1,
                    embedding_width: 160,
                    scale_block: 16,
                    weights: Some(wire::Span {
                        offset: 12288,
                        nbytes: shared_weights.len() as u64,
                        filename: String::new(),
                    }),
                    scales: Some(wire::Span {
                        offset: 12368,
                        nbytes: shared_scales.len() as u64,
                        filename: String::new(),
                    }),
                    hash: Some(wire::EngramHash {
                        method: wire::engram_hash::Method::XorModulo as i32,
                        orders: vec![],
                        multipliers: vec![],
                        primes: vec![],
                        offsets: vec![],
                    }),
                    weight_buffer: "shared_weights".into(),
                    scale_buffer: "shared_scales".into(),
                    hash_buffer: "shared_hashes".into(),
                    ready_event: "shared_ready".into(),
                    encoding: wire::engram_table::Encoding::Nvfp4E4m3Global as i32,
                    request_buffer: "shared_requests".into(),
                    global_scale: 1.0,
                    shards: vec![],
                },
                wire::EngramTable {
                    layer: 2,
                    rows: 2,
                    embedding_width: 256,
                    scale_block: 32,
                    weights: None,
                    scales: None,
                    hash: Some(wire::EngramHash {
                        method: wire::engram_hash::Method::XorModulo as i32,
                        orders: vec![],
                        multipliers: vec![],
                        primes: vec![],
                        offsets: vec![],
                    }),
                    weight_buffer: "partitioned_weights".into(),
                    scale_buffer: "partitioned_scales".into(),
                    hash_buffer: "partitioned_hashes".into(),
                    ready_event: "partitioned_ready".into(),
                    encoding: wire::engram_table::Encoding::Fp8E4m3E8m0 as i32,
                    request_buffer: "partitioned_requests".into(),
                    global_scale: 1.0,
                    shards: vec![
                        wire::EngramTableShard {
                            rank: 0,
                            first_row: 0,
                            rows: 1,
                            weights: Some(wire::Span {
                                offset: 12448,
                                nbytes: shard0_weights.len() as u64,
                                filename: String::new(),
                            }),
                            scales: Some(wire::Span {
                                offset: 12704,
                                nbytes: shard0_scales.len() as u64,
                                filename: String::new(),
                            }),
                        },
                        wire::EngramTableShard {
                            rank: 1,
                            first_row: 1,
                            rows: 1,
                            weights: Some(wire::Span {
                                offset: 12712,
                                nbytes: shard1_weights.len() as u64,
                                filename: String::new(),
                            }),
                            scales: Some(wire::Span {
                                offset: 12968,
                                nbytes: shard1_scales.len() as u64,
                                filename: String::new(),
                            }),
                        },
                    ],
                },
            ],
        });
        for (offset, name, nbytes) in [
            (12288, "shared-w.bin", shared_weights.len()),
            (12368, "shared-s.bin", shared_scales.len()),
            (12448, "shard0-w.bin", shard0_weights.len()),
            (12704, "shard0-s.bin", shard0_scales.len()),
            (12712, "shard1-w.bin", shard1_weights.len()),
            (12968, "shard1-s.bin", shard1_scales.len()),
        ] {
            rank.payload.push(FilePart {
                offset,
                source: FileSource {
                    path: sources.join(name),
                    offset: 0,
                    nbytes: nbytes as u64,
                },
            });
        }
        let assembly = base.join("assembly");
        fs::create_dir(&assembly).unwrap();
        let mut installed = Vec::new();
        package_engram(
            rank.index.engram.as_ref().unwrap(),
            &rank.payload,
            &assembly,
            "gb202-tp2",
            &mut installed,
        )
        .unwrap();
        assert_eq!(installed.len(), 6);
        assert_eq!(fs::read_dir(&assembly).unwrap().count(), 0);
        for payload in &installed {
            assert_eq!(payload.source.digest().unwrap(), payload.id);
        }
        let output = base.join("package");
        let mut second = rank.clone();
        second.index.engram = None;
        publish_payload(
            &output,
            &[rank, second],
            vec![],
            None,
            &publisher(),
            None,
            false,
            "directory",
        )
        .unwrap();

        let metadata = Metadata::open(&output).unwrap();
        let envelope = helios_core::package::protection::envelope(&metadata.envelope).unwrap();
        let deployment = pwire::DeploymentBody::decode(envelope.deployments[0].body.as_slice())
            .unwrap();
        let engram = deployment.engram.unwrap();
        assert_eq!(engram.tables.len(), 2);
        let shared = &engram.tables[0];
        assert!(shared.weights.is_some() && shared.scales.is_some());
        assert!(shared.shards.is_empty());
        let partitioned = &engram.tables[1];
        assert!(partitioned.weights.is_none() && partitioned.scales.is_none());
        assert_eq!(partitioned.shards.len(), 2);
        assert_eq!(
            partitioned
                .shards
                .iter()
                .map(|shard| (shard.rank, shard.first_row, shard.rows))
                .collect::<Vec<_>>(),
            vec![(0, 0, 1), (1, 1, 1)]
        );

        let read_use = |use_: &pwire::PayloadUse| -> Vec<u8> {
            assert_eq!(use_.role, pwire::PayloadRole::Weights as i32);
            let slice = use_.slice.as_ref().unwrap();
            let location = metadata
                .locations
                .locations
                .iter()
                .find(|location| location.payload_id == slice.payload_id)
                .unwrap();
            let stored = if location.file.is_empty() {
                metadata.primary().tensor(&location.tensor).unwrap()
            } else {
                helios_core::package::container::Container::open(
                    &output.join(&location.file),
                )
                .unwrap()
                .tensor(&location.tensor)
                .unwrap()
            };
            let mut bytes = vec![0; slice.nbytes as usize];
            stored.read_at(slice.offset, &mut bytes).unwrap();
            bytes
        };
        for (plane, use_) in [("weights", shared.weights.as_ref().unwrap()), ("scales", shared.scales.as_ref().unwrap())] {
            let id = &use_.slice.as_ref().unwrap().payload_id;
            let locations: Vec<_> = metadata.locations.locations.iter()
                .filter(|location| &location.payload_id == id).collect();
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].file, format!("weights/engram-0-{plane}.safetensors"));
        }
        assert_eq!(read_use(shared.weights.as_ref().unwrap()), shared_weights);
        assert_eq!(read_use(shared.scales.as_ref().unwrap()), shared_scales);
        assert_eq!(
            read_use(partitioned.shards[0].weights.as_ref().unwrap()),
            shard0_weights
        );
        assert_eq!(
            read_use(partitioned.shards[0].scales.as_ref().unwrap()),
            shard0_scales
        );
        assert_eq!(
            read_use(partitioned.shards[1].weights.as_ref().unwrap()),
            shard1_weights
        );
        assert_eq!(
            read_use(partitioned.shards[1].scales.as_ref().unwrap()),
            shard1_scales
        );
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn duplicate_engram_shard_rank_is_rejected_without_output_or_source_mutation() {
        let base = std::env::temp_dir().join(format!(
            "flare-engram-duplicate-rank-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("sources")).unwrap();
        let sources = base.join("sources");
        let weights = vec![0x31; 32];
        let scales = vec![0x42; 8];
        fs::write(sources.join("duplicate-w.bin"), &weights).unwrap();
        fs::write(sources.join("duplicate-s.bin"), &scales).unwrap();
        let mut rank = staged_rank_seeded(&sources, 0x77);
        rank.index.engram = Some(wire::Engram {
            token_map: vec![1, 2],
            pad_token_id: 0,
            compressed_vocabulary: 4,
            tables: vec![wire::EngramTable {
                layer: 1,
                rows: 2,
                embedding_width: 256,
                scale_block: 32,
                weights: None,
                scales: None,
                hash: Some(wire::EngramHash {
                    method: wire::engram_hash::Method::XorModulo as i32,
                    orders: vec![],
                    multipliers: vec![],
                    primes: vec![],
                    offsets: vec![],
                }),
                weight_buffer: "duplicate_weights".into(),
                scale_buffer: "duplicate_scales".into(),
                hash_buffer: "duplicate_hashes".into(),
                ready_event: "duplicate_ready".into(),
                encoding: wire::engram_table::Encoding::Fp8E4m3E8m0 as i32,
                request_buffer: "duplicate_requests".into(),
                global_scale: 1.0,
                shards: vec![
                    wire::EngramTableShard {
                        rank: 0,
                        first_row: 0,
                        rows: 1,
                        weights: Some(wire::Span {
                            offset: 12288,
                            nbytes: weights.len() as u64,
                            filename: String::new(),
                        }),
                        scales: Some(wire::Span {
                            offset: 12320,
                            nbytes: scales.len() as u64,
                            filename: String::new(),
                        }),
                    },
                    wire::EngramTableShard {
                        rank: 0,
                        first_row: 1,
                        rows: 1,
                        weights: Some(wire::Span {
                            offset: 12288,
                            nbytes: weights.len() as u64,
                            filename: String::new(),
                        }),
                        scales: Some(wire::Span {
                            offset: 12320,
                            nbytes: scales.len() as u64,
                            filename: String::new(),
                        }),
                    },
                ],
            }],
        });
        rank.payload.push(FilePart {
            offset: 12288,
            source: FileSource {
                path: sources.join("duplicate-w.bin"),
                offset: 0,
                nbytes: weights.len() as u64,
            },
        });
        rank.payload.push(FilePart {
            offset: 12320,
            source: FileSource {
                path: sources.join("duplicate-s.bin"),
                offset: 0,
                nbytes: scales.len() as u64,
            },
        });
        let output = base.join("package");
        let error = publish_payload(
            &output,
            &[rank],
            vec![],
            None,
            &publisher(),
            None,
            false,
            "directory",
        )
        .expect_err("duplicate shard ranks must fail before publication");
        assert!(
            error.to_string().contains("repeats shard rank"),
            "{error}"
        );
        assert!(!output.exists());
        assert_eq!(fs::read(sources.join("duplicate-w.bin")).unwrap(), weights);
        assert_eq!(fs::read(sources.join("duplicate-s.bin")).unwrap(), scales);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn custom_cluster_identity_preserves_captured_hardware_identity() {
        use helios_core::package::schema::wire as pwire;
        let base =
            std::env::temp_dir().join(format!("flare-hardware-identity-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let sources = base.join("sources");
        std::fs::create_dir_all(&sources).unwrap();
        let cluster = wire::Cluster {
            world: 1,
            export_id: "custom-export".into(),
        };
        let output = base.join("package");
        publish_payload(
            &output,
            &[staged_rank(&sources)],
            vec![],
            Some(cluster),
            &publisher(),
            None,
            false,
            "directory",
        )
        .unwrap();
        let metadata = Metadata::open(&output).unwrap();
        let envelope = helios_core::package::protection::envelope(&metadata.envelope).unwrap();
        let body = pwire::PackageBody::decode(envelope.package_body.as_slice()).unwrap();
        assert_eq!(body.hardware_identity, "gb202");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn cached_state_and_staged_globals_use_unique_payload_names() {
        use crate::export::{WeightPayloadGroup, WeightPayloadSourceSpan, WeightPayloadSources};
        use helios_core::package::PayloadId;
        use helios_core::package::cache::Root;
        let base = std::env::temp_dir().join(format!("flare-state-globals-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let cache = base.join("cache");
        let sources = base.join("sources");
        std::fs::create_dir_all(&sources).unwrap();
        let root = Root::open(&cache).unwrap();
        let state_bytes = b"residSt8".to_vec();
        let state_id = PayloadId::of(&state_bytes);
        root.insert_payload(state_id, 8, std::io::Cursor::new(state_bytes))
            .unwrap();
        let weight_bytes = b"weights8".to_vec();
        let weight_id = PayloadId::of(&weight_bytes);
        root.insert_payload(weight_id, 8, std::io::Cursor::new(weight_bytes))
            .unwrap();
        let mut rank = staged_rank(&sources);
        rank.index.ranks[0].allocations[0].nbytes = 4096;
        let globals = b"global16".to_vec();
        std::fs::write(sources.join("globals.bin"), &globals).unwrap();
        rank.index.ranks[0].modules[0].globals.push(wire::Global {
            symbol: "const".into(),
            file: "globals.bin".into(),
            nbytes: globals.len() as u64,
            relocations: vec![],
        });
        rank.index.ranks[0].resources.insert(
            "globals.bin".into(),
            wire::Span {
                offset: 12288,
                nbytes: globals.len() as u64,
                filename: String::new(),
            },
        );
        rank.payload.push(FilePart {
            offset: 12288,
            source: FileSource {
                path: sources.join("globals.bin"),
                offset: 0,
                nbytes: globals.len() as u64,
            },
        });
        rank.weight_payload = Some(WeightPayloadSources {
            cached: Vec::new(),
            engram: None,
            groups: vec![WeightPayloadGroup {
                name: "layer-0".into(),
                storage: "stable".into(),
                nbytes: 8,
                spans: vec![WeightPayloadSourceSpan {
                    allocation: "device-0".into(),
                    destination_offset: 0,
                    source_offset: 0,
                    nbytes: 8,
                }],
                payload_id: weight_id.as_bytes().to_vec(),
            }],
            initial_state: WeightPayloadGroup {
                name: "initial-state".into(),
                storage: "stable".into(),
                nbytes: 8,
                spans: vec![WeightPayloadSourceSpan {
                    allocation: "device-0".into(),
                    destination_offset: 32,
                    source_offset: 0,
                    nbytes: 8,
                }],
                payload_id: state_id.as_bytes().to_vec(),
            },
            views: vec![],
        });
        let published = publish_payload(
            Path::new(""),
            &[rank],
            vec![],
            None,
            &publisher(),
            Some(cache),
            false,
            "directory",
        )
        .unwrap();
        let entry = published["entry"].as_str().unwrap().to_string();
        let model = published["model"].as_str().unwrap().to_string();
        let committed = root.entry(&model, &entry).unwrap();
        let metadata = Metadata::open(committed.directory()).unwrap();
        let mut names: Vec<String> = metadata
            .locations
            .locations
            .iter()
            .map(|location| location.file.clone())
            .collect();
        names.retain(|name| !name.is_empty());
        names.sort();
        names.dedup();
        assert!(
            names
                .iter()
                .any(|name| name.ends_with("initial-state.safetensors")),
            "{names:?}"
        );
        assert!(
            names
                .iter()
                .any(|name| name.ends_with("initial-state-globals.safetensors")),
            "{names:?}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn payload_publication_coalesces_adjacent_initializers_and_preserves_destination_gaps() {
        use crate::export::{WeightPayloadGroup, WeightPayloadSourceSpan, WeightPayloadSources};
        for (case, spans, expected_count) in [
            (
                "adjacent",
                vec![
                    WeightPayloadSourceSpan {
                        allocation: "device-0".into(),
                        destination_offset: 0,
                        source_offset: 0,
                        nbytes: 32,
                    },
                    WeightPayloadSourceSpan {
                        allocation: "device-0".into(),
                        destination_offset: 32,
                        source_offset: 32,
                        nbytes: 32,
                    },
                ],
                1usize,
            ),
            (
                "gap",
                vec![
                    WeightPayloadSourceSpan {
                        allocation: "device-0".into(),
                        destination_offset: 0,
                        source_offset: 0,
                        nbytes: 32,
                    },
                    WeightPayloadSourceSpan {
                        allocation: "device-0".into(),
                        destination_offset: 64,
                        source_offset: 32,
                        nbytes: 32,
                    },
                ],
                2usize,
            ),
        ] {
            let base = std::env::temp_dir().join(format!(
                "flare-initializer-coalesce-{}-{case}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&base);
            let sources = base.join("sources");
            std::fs::create_dir_all(&sources).unwrap();
            let mut rank = staged_rank_seeded(&sources, 0x11);
            rank.index.ranks[0].allocations[0].nbytes = 128;
            rank.weight_payload = Some(WeightPayloadSources {
                cached: Vec::new(),
                engram: None,
                groups: vec![WeightPayloadGroup {
                    name: "layer-0".into(),
                    storage: "stable".into(),
                    nbytes: 64,
                    spans,
                    payload_id: vec![],
                }],
                initial_state: WeightPayloadGroup {
                    name: "state".into(),
                    storage: "state".into(),
                    ..Default::default()
                },
                views: vec![],
            });
            let output = base.join("package");
            publish_payload(
                &output,
                &[rank],
                vec![],
                None,
                &publisher(),
                None,
                false,
                "directory",
            )
            .unwrap();
            let metadata = Metadata::open(&output).unwrap();
            let envelope = helios_core::package::protection::envelope(&metadata.envelope).unwrap();
            let body = helios_core::package::schema::wire::RankBody::decode(
                envelope.ranks[0].body.as_slice(),
            )
            .unwrap();
            assert_eq!(body.initializers.len(), expected_count, "{case}");
            if expected_count == 1 {
                let slice = body.initializers[0]
                    .source
                    .as_ref()
                    .unwrap()
                    .slice
                    .as_ref()
                    .unwrap();
                assert_eq!(body.initializers[0].destination_offset, 0);
                assert_eq!(slice.offset, 0);
                assert_eq!(slice.nbytes, 64);
            } else {
                assert_eq!(body.initializers[0].destination_offset, 0);
                assert_eq!(body.initializers[1].destination_offset, 64);
                assert_eq!(body.initializers[0].source.as_ref().unwrap().slice.as_ref().unwrap().nbytes, 32);
                assert_eq!(body.initializers[1].source.as_ref().unwrap().slice.as_ref().unwrap().nbytes, 32);
            }
            let _ = std::fs::remove_dir_all(&base);
        }
    }

    #[test]
    fn payload_publication_rejects_overlapping_initializer_boundaries() {
        use crate::export::{WeightPayloadGroup, WeightPayloadSourceSpan, WeightPayloadSources};
        let base = std::env::temp_dir().join(format!(
            "flare-initializer-overlap-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let sources = base.join("sources");
        std::fs::create_dir_all(&sources).unwrap();
        let mut rank = staged_rank_seeded(&sources, 0x11);
        rank.index.ranks[0].allocations[0].nbytes = 128;
        rank.weight_payload = Some(WeightPayloadSources {
            cached: Vec::new(),
            engram: None,
            groups: vec![WeightPayloadGroup {
                name: "layer-0".into(),
                storage: "stable".into(),
                nbytes: 64,
                spans: vec![
                    WeightPayloadSourceSpan {
                        allocation: "device-0".into(),
                        destination_offset: 0,
                        source_offset: 0,
                        nbytes: 32,
                    },
                    WeightPayloadSourceSpan {
                        allocation: "device-0".into(),
                        destination_offset: 16,
                        source_offset: 32,
                        nbytes: 32,
                    },
                ],
                payload_id: vec![],
            }],
            initial_state: WeightPayloadGroup {
                name: "state".into(),
                storage: "state".into(),
                ..Default::default()
            },
            views: vec![],
        });
        let output = base.join("package");
        let error = publish_payload(
            &output,
            &[rank],
            vec![],
            None,
            &publisher(),
            None,
            false,
            "directory",
        )
        .expect_err("overlapping initializer boundaries must be rejected");
        assert!(
            error.to_string().contains("overlap"),
            "unexpected overlap failure: {error}"
        );
        assert!(!output.exists());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn tp2_and_tp4_rank0_groups_publish_and_convert_without_name_collision() {
        use crate::export::{WeightPayloadGroup, WeightPayloadSourceSpan, WeightPayloadSources};
        use helios_core::package::{conversion, reader as payload_reader};
        let base = std::env::temp_dir().join(format!("flare-tp-compose-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let publisher = publisher();
        let rng = ring::rand::SystemRandom::new();
        let pkcs8 = ring::signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = ring::signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let publish = |name: &str, ranks: usize, byte: u8| -> std::path::PathBuf {
            let sources = base.join(format!("{name}-sources"));
            std::fs::create_dir_all(&sources).unwrap();
            let mut rank = staged_rank(&sources);
            std::fs::write(sources.join("storage-0.bin"), vec![byte; 64]).unwrap();
            rank.payload[0].source.path = sources.join("storage-0.bin");
            rank.weight_payload = Some(WeightPayloadSources {
                cached: Vec::new(),
                engram: None,
                groups: vec![WeightPayloadGroup {
                    name: "layer-0".into(),
                    storage: "stable".into(),
                    nbytes: 64,
                    spans: vec![WeightPayloadSourceSpan {
                        allocation: "device-0".into(),
                        destination_offset: 0,
                        source_offset: 0,
                        nbytes: 64,
                    }],
                    payload_id: vec![],
                }],
                initial_state: WeightPayloadGroup {
                    name: "initial-state".into(),
                    storage: "stable".into(),
                    nbytes: 0,
                    spans: vec![],
                    payload_id: vec![],
                },
                views: vec![],
            });
            let mut all = Vec::new();
            for _ in 0..ranks {
                all.push(rank.clone());
            }
            let output = base.join(name);
            publish_payload(
                &output,
                &all,
                vec![],
                None,
                &publisher,
                None,
                false,
                "directory",
            )
            .unwrap();
            output
        };
        let tp2 = publish("tp2", 2, 0x11);
        let tp4 = publish("tp4", 4, 0x22);
        for (path, tp) in [(&tp2, 2), (&tp4, 4)] {
            let metadata = payload_reader::Metadata::open(path).unwrap();
            assert!(
                metadata
                    .locations
                    .locations
                    .iter()
                    .any(|location| location.file.contains(&format!("weights/tp{tp}/rank-0/"))),
                "{tp}: {:?}",
                metadata
                    .locations
                    .locations
                    .iter()
                    .map(|l| &l.file)
                    .collect::<Vec<_>>()
            );
        }
        let runtime_publisher = helios_core::runtime_package::Publisher {
            name: "Flare test publisher".into(),
            namespace: publisher.namespace.clone(),
            public_key: publisher.public_key().try_into().unwrap(),
            development: publisher.development,
        };
        let combined = base.join("combined");
        conversion::combine(&conversion::CombineRequest {
            sources: &[tp2.clone(), tp4.clone()],
            output: &combined,
            publishers: std::slice::from_ref(&runtime_publisher),
            accepted: "example",
            target_accepted: "example",
            target_publisher: None,
            key: &key,
        })
        .unwrap();
        let assert_both = |path: &std::path::Path, label: &str| {
            let metadata = payload_reader::Metadata::open(path).unwrap();
            let names: Vec<String> = metadata
                .locations
                .locations
                .iter()
                .map(|l| l.file.clone())
                .collect();
            assert!(
                names.iter().any(|name| name.contains("tp2/rank-0/")),
                "{label}: {names:?}"
            );
            assert!(
                names.iter().any(|name| name.contains("tp4/rank-0/")),
                "{label}: {names:?}"
            );
            let ids: std::collections::BTreeSet<Vec<u8>> = metadata
                .locations
                .locations
                .iter()
                .filter(|l| l.file.contains("rank-0/") && l.file.contains("/tp"))
                .map(|l| l.payload_id.clone())
                .collect();
            assert!(
                ids.len() >= 2,
                "{label}: distinct TP rank0 bytes must survive"
            );
        };
        assert_both(&combined, "combined");
        let packed = base.join("combined.safetensors");
        conversion::pack(&combined, &packed).unwrap();
        let unpacked = base.join("unpacked");
        conversion::unpack(&packed, &unpacked).unwrap();
        assert_both(&unpacked, "unpacked");
        let _ = std::fs::remove_dir_all(&base);
    }

    fn empty_rank(hardware: Option<&str>) -> RankOutput {
        RankOutput {
            hardware: hardware.map(str::to_string),
            payload_leases: Default::default(),
            index: wire::ModelIndex {
                version: 4,
                ranks: vec![Default::default()],
                ..Default::default()
            },
            payload: vec![],
            allocations: vec![],
            module_globals: vec![],
            external: vec![],
            staging: None,
            weight_payload: None,
        }
    }

    #[test]
    fn staged_sources_preserve_hardware_and_reject_inconsistent_ranks() {
        let directory =
            std::env::temp_dir().join(format!("flare-sources-hardware-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("rank-0.json");
        write_sources(&path, &empty_rank(Some("gb202"))).unwrap();
        let read = read_sources(&path).unwrap();
        assert_eq!(read.hardware.as_deref(), Some("gb202"));
        assert!(
            combine(
                &[empty_rank(Some("gb202")), empty_rank(Some("gb10"))],
                vec![],
                None
            )
            .is_err()
        );
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn changed_undeclared_state_publishes_fresh_under_no_weights() {
        let base = std::env::temp_dir().join(format!("flare-adapter-state-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let cache = base.join("cache");
        let sources = base.join("sources");
        std::fs::create_dir_all(&sources).unwrap();
        let first = publish_payload(
            Path::new(""),
            &[staged_rank_seeded(&sources, 0x11)],
            vec![],
            None,
            &publisher(),
            Some(cache.clone()),
            false,
            "directory",
        )
        .unwrap();
        let model = first["model"].as_str().unwrap().to_string();
        // Different state bytes mean a different content ID: fresh state is
        // allowed under --no-weights and the new entry becomes default.
        let second = publish_payload(
            Path::new(""),
            &[staged_rank_seeded(&sources, 0x22)],
            vec![],
            None,
            &publisher(),
            Some(cache.clone()),
            true,
            "directory",
        )
        .unwrap();
        let entry = second["entry"].as_str().unwrap().to_string();
        assert_ne!(first["entry"], second["entry"]);
        let root = helios_core::package::cache::Root::open(&cache).unwrap();
        assert_eq!(root.resolve(&model).unwrap().unwrap().id(), entry);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn sync_parent_normalizes_a_relative_basename() {
        let directory = std::env::temp_dir().join(format!("flare-sync-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let previous = std::env::current_dir().unwrap();
        std::env::set_current_dir(&directory).unwrap();
        let result = sync_parent(Path::new("model.safetensors"));
        std::env::set_current_dir(previous).unwrap();
        let _ = std::fs::remove_dir_all(&directory);
        result.unwrap();
    }

    #[test]
    fn payload_export_writes_runnable_artifact_and_fresh_identity() {
        let base = std::env::temp_dir().join(format!("flare-adapter-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let first = base.join("first");
        let second = base.join("second");
        for destination in [&first, &second] {
            let sources = base.join(format!(
                "sources-{}",
                destination.file_name().unwrap().to_string_lossy()
            ));
            std::fs::create_dir_all(&sources).unwrap();
            let rank = staged_rank(&sources);
            let result = publish_payload(
                destination,
                &[rank],
                vec![],
                None,
                &publisher(),
                None,
                false,
                "directory",
            )
            .unwrap();
            assert_eq!(result["format"], "payload-directory");
            assert!(destination.join("model.safetensors").exists());
            assert!(
                destination
                    .join("deployments/gb202-tp1/rank-0/initial-state.safetensors")
                    .exists()
            );
            assert!(
                destination
                    .join("deployments/gb202-tp1/rank-0/kernels.safetensors")
                    .exists()
            );
            assert!(
                destination
                    .join("deployments/gb202-tp1/rank-0/graphs.safetensors")
                    .exists()
            );
        }
        let (first_id, first_payloads) = manifest_ids(&first);
        let (second_id, second_payloads) = manifest_ids(&second);
        assert_ne!(first_id, second_id);
        assert!(!first_payloads.is_empty());
        // Compressed executable payloads are deterministic; only the fresh
        // package identity differs.
        assert_eq!(first_payloads, second_payloads);
        let _ = std::fs::remove_dir_all(&base);
    }
}

#[cfg(test)]
mod packing_metadata_tests {
    use super::super::{
        WeightPayloadPiece, WeightPayloadSource, WeightPayloadView,
        weight_payload::WeightPackingParameters,
    };

    #[test]
    fn prepared_provenance_carries_declared_byte_pieces_only() {
        let legacy = WeightPayloadSource {
            tensor: "model.embed.weight".into(),
            ranges: vec![(0, 8)],
            concatenation_axis: Some(0),
            pieces: vec![],
            piece_encoding: String::new(),
        };
        let view = WeightPayloadView {
            name: "embed".into(),
            group: "embedding".into(),
            storage: "stable".into(),
            sources: vec![legacy],
            packing: String::new(),
            packing_parameters: None,
            statistics_domain: String::new(),
        };
        let (sources, _, _) = super::prepared_provenance(&view);
        assert!(sources[0].pieces.is_empty());
        assert_eq!(sources[0].ranges[0].begin, 0);
        assert_eq!(sources[0].ranges[0].end, 8);

        let mapped = WeightPayloadView {
            sources: vec![WeightPayloadSource {
                tensor: "model.experts.0.up_proj.weight_scale".into(),
                ranges: vec![(0, 2304), (0, 8), (0, 74)],
                concatenation_axis: Some(0),
                piece_encoding: "w4a16_mma_k16_n64_v1".into(),
                pieces: vec![WeightPayloadPiece {
                    source_offset: 0,
                    destination_offset: 0,
                    contiguous_bytes: 4,
                    extents: vec![9, 4, 32, 112],
                    source_strides: vec![57344, 14336, 448, 4],
                    destination_strides: vec![57344, 4, 16, 512],
                }],
            }],
            ..view
        };
        let (sources, _, _) = super::prepared_provenance(&mapped);
        assert_eq!(sources[0].pieces.len(), 1);
        assert_eq!(sources[0].pieces[0].contiguous_bytes, 4);
        assert_eq!(
            sources[0].pieces[0].destination_strides,
            vec![57344, 4, 16, 512]
        );
        assert_eq!(sources[0].ranges.len(), 3);
        assert_eq!(sources[0].piece_encoding, "w4a16_mma_k16_n64_v1");
    }

    #[test]
    fn signed_packing_parameters_carry_only_declared_geometry() {
        let declared = WeightPayloadView {
            name: "layers.0.experts".into(),
            group: "layer-0".into(),
            storage: "stable".into(),
            sources: vec![],
            packing: "nvfp4".into(),
            packing_parameters: Some(WeightPackingParameters {
                format: "nvfp4".into(),
                group_size: 16,
                ..Default::default()
            }),
            statistics_domain: "tensor".into(),
        };
        let (_, packing, domain) = super::prepared_provenance(&declared);
        let packing = packing.expect("declared parameters must reach the signed view");
        assert_eq!(packing.format, "nvfp4");
        assert_eq!(packing.group_size, 16);
        // Nothing is inferred from the format name.
        assert_eq!(packing.block_size, 0);
        assert_eq!(packing.scale_atom_rows, 0);
        assert_eq!(packing.tile_order, 0);
        assert_eq!(packing.row_rotation, 0);
        assert_eq!(packing.locality_tile, 0);
        assert_eq!(domain, "tensor");

        // A producer that only knows a format stays format-only.
        let plain = WeightPayloadView {
            packing_parameters: None,
            ..declared
        };
        let (_, packing, _) = super::prepared_provenance(&plain);
        let packing = packing.expect("format-only packing is preserved");
        assert_eq!(packing.format, "nvfp4");
        assert_eq!(packing.group_size, 0);

        // No packing at all stays absent.
        let absent = WeightPayloadView {
            packing: String::new(),
            packing_parameters: None,
            ..plain
        };
        let (_, packing, _) = super::prepared_provenance(&absent);
        assert!(packing.is_none());
    }
}
