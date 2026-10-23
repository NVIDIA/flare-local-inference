//! Rank assembly and immutable package publication.
use super::archive::{self, Container, Layout};
use super::blob::BlobStore;
use super::io::{IoBudget, Source};
use super::protection::{self, Publisher};
use super::{
    ExportOptions, ExportResult as Result, FilePart, FileSource, RankOutput, invalid, wire,
};
use prost::Message;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::Write,
    path::Path,
};

pub(crate) const ALIGNMENT: u64 = 4096;
pub(crate) const DEFAULT_SHARD_SIZE: u64 = 64 << 30;
pub(crate) const MAX_PUBLISHED_BLOB_BYTES: u64 = 64_000_000_000;
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
    pub key: SpanKey,
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
                key: SpanKey::Resource(owner, name.into()),
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
    for (i, asset) in index.assets.iter().enumerate() {
        result.push(Role {
            kind: 3,
            owner: None,
            key: SpanKey::Asset(i),
            span: asset
                .payload
                .clone()
                .ok_or_else(|| invalid("asset lacks payload"))?,
        });
    }
    if let Some(engram) = &index.engram {
        for (i, table) in engram.tables.iter().enumerate() {
            if table.shards.is_empty() {
                for scales in [false, true] {
                    result.push(Role {
                        kind: 4,
                        owner: None,
                        key: SpanKey::Table(i, None, scales),
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
                for (j, shard) in table.shards.iter().enumerate() {
                    if shard.rank as usize >= index.ranks.len() {
                        return Err(invalid("embedding partition owner outside package ranks"));
                    }
                    for scales in [false, true] {
                        result.push(Role {
                            kind: 4,
                            owner: Some(shard.rank as usize),
                            key: SpanKey::Table(i, Some(j), scales),
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
}
pub(crate) fn write_sources(path: &Path, rank: &RankOutput) -> Result<()> {
    let length = validate_payload(&rank.payload)?;
    validate_index(&rank.index, length)?;
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer(
        &mut output,
        &Sources {
            index: rank.index.clone(),
            payload: rank.payload.clone(),
        },
    )?;
    output.sync_all()?;
    Ok(())
}
pub(crate) fn read_sources(path: &Path) -> Result<RankOutput> {
    let record: Sources = serde_json::from_reader(File::open(path)?)?;
    validate_index(&record.index, validate_payload(&record.payload)?)?;
    Ok(RankOutput {
        index: record.index,
        payload: record.payload,
        allocations: vec![],
        module_globals: vec![],
        external: vec![],
        staging: None,
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

#[derive(Clone)]
pub(crate) struct ResourceGroup {
    pub roles: Vec<Role>,
    pub filename: String,
    pub offset: u64,
    pub nbytes: u64,
}
pub(crate) fn groups(index: &wire::ModelIndex, shard_size: u64) -> Result<Vec<ResourceGroup>> {
    if !(ALIGNMENT < shard_size && shard_size <= DEFAULT_SHARD_SIZE) {
        return Err(invalid(
            "payload shard limit must exceed 4096 bytes and be at most 64G",
        ));
    }
    let limit = shard_size.min(MAX_PUBLISHED_BLOB_BYTES) - ALIGNMENT;
    let mut roles = roles(index)?;
    roles.sort_by_key(|r| (r.span.filename.clone(), r.span.offset));
    let mut result: Vec<ResourceGroup> = Vec::new();
    for role in roles {
        if role.span.nbytes > limit {
            return Err(invalid("package resource exceeds the payload shard limit"));
        }
        let append = result.last().is_some_and(|g| {
            g.filename == role.span.filename
                && g.roles[0].owner == role.owner
                && (g.roles[0].kind == 4) == (role.kind == 4)
                && role.span.offset + role.span.nbytes - g.offset <= limit
        });
        if append {
            let group = result.last_mut().unwrap();
            group.nbytes = role.span.offset + role.span.nbytes - group.offset;
            group.roles.push(role);
        } else {
            result.push(ResourceGroup {
                filename: role.span.filename.clone(),
                offset: role.span.offset,
                nbytes: role.span.nbytes,
                roles: vec![role],
            });
        }
    }
    let mut segments = result.len();
    let mut ends = BTreeMap::new();
    for group in &result {
        if group.offset > *ends.get(&group.filename).unwrap_or(&0) {
            segments += 1;
        }
        ends.insert(group.filename.clone(), group.offset + group.nbytes);
    }
    if segments > 65536 {
        return Err(invalid("snapshot has too many payload segments"));
    }
    Ok(result)
}
#[derive(Debug, Serialize)]
pub(crate) struct PublishStats {
    pub written: u64,
    pub reused: u64,
}

pub(crate) fn publish(
    path: &Path,
    index: wire::ModelIndex,
    payload: Vec<FilePart>,
    publisher: Option<&Publisher>,
    options: &ExportOptions,
    shard_size: u64,
) -> Result<PublishStats> {
    publish_checked(
        path,
        index,
        payload,
        publisher,
        options,
        shard_size,
        |_, _| Ok(()),
    )
}
pub(crate) fn publish_checked(
    path: &Path,
    index: wire::ModelIndex,
    payload: Vec<FilePart>,
    publisher: Option<&Publisher>,
    options: &ExportOptions,
    shard_size: u64,
    verify: impl Fn(&protection::Protected, &Layout) -> Result<()>,
) -> Result<PublishStats> {
    if path.try_exists()? {
        return Err(invalid(format!("destination exists: {}", path.display())));
    }
    let parent = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let Some(publisher) = publisher else {
        write_staging(path, &index, &payload)?;
        return Ok(PublishStats {
            written: 0,
            reused: 0,
        });
    };
    let budget = IoBudget::for_ranks(1)?;
    let stage = tempfile::Builder::new()
        .prefix(".flare-protection-")
        .tempdir_in(parent)?;
    let progress = helios_core::startup::StartupProgress::stderr_rows();
    let protected_bytes = roles(&index)?
        .iter()
        .filter(|r| r.kind != 4)
        .map(|r| r.span.nbytes)
        .sum::<u64>();
    let protect_phase = progress.phase(
        "Hash and encrypt executable resources",
        Some(protected_bytes),
    );
    let protected = protection::prepare(
        index,
        payload,
        publisher,
        stage.path(),
        shard_size,
        path.file_stem().and_then(|s| s.to_str()).unwrap_or("model"),
        budget.clone(),
        &protect_phase,
    )?;
    protect_phase.finish(format!(
        "{} plaintext bytes read; {} encrypted resources",
        protected_bytes,
        protected
            .code
            .values()
            .map(|c| c.resources.len())
            .sum::<usize>()
    ));
    let blob_bytes = groups(&protected.index, shard_size)?
        .iter()
        .map(|g| g.nbytes)
        .sum();
    let blob_phase = progress.phase("Hash and publish payload blobs", Some(blob_bytes));
    let cache = options
        .cache
        .clone()
        .unwrap_or_else(|| parent.join(".flare-cache"));
    let store = BlobStore::new(&cache, budget.clone())?;
    let (layout, hashes, detail) = publish_groups(
        &store,
        &protected,
        options.no_weights,
        shard_size,
        &blob_phase,
    )?;
    for source in &protected.source_snapshots {
        source.check_unchanged()?;
    }
    for parts in layout.files.values() {
        for (_, name) in parts {
            if !name.is_empty() {
                store.link(name, parent)?;
            }
        }
    }
    verify(&protected, &layout)?;
    blob_phase.finish(detail);
    let sign_phase = progress.phase("Sign and publish package metadata", None);
    let envelope = protected.envelope(publisher, &hashes)?;
    store.sync()?;
    archive::install_snapshot(path, &envelope, &layout)?;
    sign_phase.finish(format!("{} metadata bytes", envelope.len()));
    Ok(PublishStats {
        written: store.written.load(std::sync::atomic::Ordering::Relaxed),
        reused: store.reused.load(std::sync::atomic::Ordering::Relaxed),
    })
}

pub(crate) fn publish_groups(
    store: &BlobStore,
    protected: &protection::Protected,
    no_weights: bool,
    shard_size: u64,
    phase: &helios_core::startup::Phase,
) -> Result<(Layout, BTreeMap<Option<usize>, Vec<u8>>, String)> {
    use ring::digest;
    use std::sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc,
    };
    let read = AtomicU64::new(0);
    let hashed = AtomicU64::new(0);
    let copied = AtomicU64::new(0);
    let describe = || {
        format!(
            "{} source bytes read; {} bytes hashed; {} bytes copied; {} bytes installed; {} bytes reused",
            read.load(Ordering::Relaxed),
            hashed.load(Ordering::Relaxed),
            copied.load(Ordering::Relaxed),
            store.written.load(Ordering::Relaxed),
            store.reused.load(Ordering::Relaxed)
        )
    };
    let planned = groups(&protected.index, shard_size)?;
    let mut source_files = BTreeMap::new();
    for source in &protected.source_snapshots {
        source_files.insert(source.path.clone(), source.clone());
    }
    let mut logical_sources = BTreeMap::new();
    for (name, parts) in &protected.payloads {
        let mut resources = Vec::with_capacity(parts.len());
        for part in parts {
            if !source_files.contains_key(&part.source.path) {
                source_files.insert(
                    part.source.path.clone(),
                    protection::SourceSnapshot::open(&part.source.path)?,
                );
            }
            let snapshot = &source_files[&part.source.path];
            if part
                .source
                .offset
                .checked_add(part.source.nbytes)
                .is_none_or(|end| end > snapshot.identity.size)
            {
                return Err(invalid("export source range exceeds file"));
            }
            resources.push((part.offset, part.source.clone()));
        }
        logical_sources.insert(name.clone(), resources);
    }
    let active = Mutex::new(BTreeMap::<std::path::PathBuf, Source>::new());
    let open_source = |source: &FileSource| -> Result<Source> {
        let mut active = active
            .lock()
            .map_err(|_| invalid("publication source lock poisoned"))?;
        if !active.contains_key(&source.path) {
            let snapshot = &source_files[&source.path];
            let pin = snapshot.source(0, snapshot.identity.size)?;
            if active.len() == 8 {
                active.pop_first();
            }
            active.insert(source.path.clone(), pin);
        }
        active[&source.path].slice(source.offset, source.nbytes)
    };
    let selections: Result<Vec<_>> = planned
        .iter()
        .map(|group| {
            let parts = logical_sources
                .get(&group.filename)
                .ok_or_else(|| invalid("missing publication logical file"))?;
            let first =
                parts.partition_point(|(offset, source)| offset + source.nbytes <= group.offset);
            let last = parts.partition_point(|(offset, _)| *offset < group.offset + group.nbytes);
            Ok(&parts[first..last])
        })
        .collect();
    let selections = selections?;
    let fill = |i: usize, offset: u64, buffer: &mut [u8]| -> Result<()> {
        let start = planned[i].offset + offset;
        let end = start + buffer.len() as u64;
        let mut cursor = start;
        let selected = selections[i];
        let first =
            selected.partition_point(|(position, source)| position + source.nbytes <= start);
        for (position, source) in &selected[first..] {
            if *position >= end {
                break;
            }
            let first = start.max(*position);
            let stop = end.min(*position + source.nbytes);
            if cursor < first {
                buffer[(cursor - start) as usize..(first - start) as usize].fill(0);
            }
            if first < stop {
                let source = open_source(source)?;
                source.check_unchanged()?;
                store.budget.read_at(
                    &source,
                    first - *position,
                    &mut buffer[(first - start) as usize..(stop - start) as usize],
                )?;
                source.check_unchanged()?;
                read.fetch_add(stop - first, Ordering::Relaxed);
                cursor = stop;
            }
        }
        if cursor < end {
            buffer[(cursor - start) as usize..].fill(0);
        }
        Ok(())
    };
    let prepare = |i: usize, mut digest: Option<&mut digest::Context>| {
        let group = &planned[i];
        let selected = selections[i];
        for (_, source) in selected {
            source_files[&source.path].check_unchanged()?;
        }
        let source_blob = if selected.len() == 1
            && selected[0].0 == group.offset
            && selected[0].1.nbytes == group.nbytes
        {
            Some(open_source(&selected[0].1)?)
        } else {
            None
        };
        let prepared = store.prepare(
            group.nbytes,
            no_weights && group.roles[0].kind == 4,
            source_blob.as_ref(),
            |offset, buffer| {
                fill(i, offset, buffer)?;
                if let Some(hash) = digest.as_deref_mut() {
                    hash.update(buffer);
                }
                hashed.fetch_add(buffer.len() as u64, Ordering::Relaxed);
                phase.advance(buffer.len() as u64);
                phase.detail(describe());
                Ok(())
            },
        )?;
        for (_, source) in selected {
            source_files[&source.path].check_unchanged()?;
        }
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(prepared)
    };
    let owners: Vec<_> = protected.ranges.keys().copied().collect();
    let writers = owners.len().min(4).min(store.budget.depth).max(1);
    let (jobs, receiver) = mpsc::sync_channel::<(usize, super::blob::PreparedBlob)>(writers);
    let receiver = Mutex::new(receiver);
    let (hashes, mut names) = std::thread::scope(|scope| -> Result<_> {
        let (outgoing, results) = mpsc::channel();
        let mut workers = Vec::with_capacity(writers);
        for _ in 0..writers {
            let receiver = &receiver;
            let outgoing = outgoing.clone();
            let fill = &fill;
            let selections = &selections;
            let source_files = &source_files;
            let copied = &copied;
            let describe = &describe;
            workers.push(scope.spawn(move || {
                loop {
                    let job = match receiver.lock() {
                        Ok(receiver) => receiver.recv(),
                        Err(_) => break,
                    };
                    let Ok((i, prepared)) = job else { break };
                    let result = (|| {
                        let name = store.commit(prepared, |offset, buffer| {
                            fill(i, offset, buffer)?;
                            copied.fetch_add(buffer.len() as u64, Ordering::Relaxed);
                            phase.detail(describe());
                            Ok(())
                        })?;
                        for (_, source) in selections[i] {
                            source_files[&source.path].check_unchanged()?;
                        }
                        phase.detail(describe());
                        Ok::<_, Box<dyn std::error::Error + Send + Sync>>((i, name))
                    })();
                    let _ = outgoing.send(result);
                }
            }));
        }
        drop(outgoing);
        let submit = |i, prepared| -> Result<()> {
            jobs.send((i, prepared))
                .map_err(|_| invalid("payload writer queue closed"))
        };
        let stored: Result<Vec<_>> = owners
            .par_iter()
            .map(|owner| {
                let mut digest = digest::Context::new(&digest::SHA256);
                let zeros = [0u8; 4096];
                for part in &protected.ranges[owner] {
                    let mut cursor = part.offset;
                    for (i, group) in planned.iter().enumerate().filter(|(_, g)| {
                        g.roles[0].kind == 4
                            && g.roles[0].owner == *owner
                            && g.filename == part.filename
                            && g.offset >= part.offset
                            && g.offset + g.nbytes <= part.offset + part.nbytes
                    }) {
                        let mut gap = group.offset - cursor;
                        while gap > 0 {
                            let count = gap.min(zeros.len() as u64) as usize;
                            digest.update(&zeros[..count]);
                            gap -= count as u64;
                        }
                        let prepared = prepare(i, Some(&mut digest))?;
                        submit(i, prepared)?;
                        cursor = group.offset + group.nbytes;
                    }
                    let mut gap = part.offset + part.nbytes - cursor;
                    while gap > 0 {
                        let count = gap.min(zeros.len() as u64) as usize;
                        digest.update(&zeros[..count]);
                        gap -= count as u64;
                    }
                }
                Ok((*owner, digest.finish().as_ref().to_vec()))
            })
            .collect();
        let execution: Result<()> = if stored.is_ok() {
            planned
                .par_iter()
                .enumerate()
                .filter(|(_, g)| g.roles[0].kind != 4)
                .try_for_each(|(i, _)| submit(i, prepare(i, None)?))
        } else {
            Ok(())
        };
        drop(jobs);
        for worker in workers {
            worker
                .join()
                .map_err(|_| invalid("payload writer panicked"))?;
        }
        let mut names = BTreeMap::new();
        let mut write_error = None;
        for result in results {
            match result {
                Ok((i, name)) => {
                    names.insert(i, name);
                }
                Err(error) => {
                    if write_error.is_none() {
                        write_error = Some(error);
                    }
                }
            }
        }
        let hashes = stored?.into_iter().collect::<BTreeMap<_, _>>();
        execution?;
        if let Some(error) = write_error {
            return Err(error);
        }
        Ok((hashes, names))
    })?;
    let mut files = BTreeMap::from([(String::new(), Vec::new())]);
    for (i, group) in planned.iter().enumerate() {
        let entries = files.entry(group.filename.clone()).or_insert_with(Vec::new);
        let cursor: u64 = entries.iter().map(|(n, _)| *n).sum();
        if cursor < group.offset {
            entries.push((group.offset - cursor, String::new()));
        }
        entries.push((
            group.nbytes,
            names
                .remove(&i)
                .ok_or_else(|| invalid("unpublished resource group"))?,
        ));
    }
    Ok((Layout { version: 1, files }, hashes, describe()))
}

pub(crate) fn write_staging(
    path: &Path,
    index: &wire::ModelIndex,
    payload: &[FilePart],
) -> Result<()> {
    let length = validate_payload(payload)?;
    validate_index(index, length)?;
    let metadata = index.encode_to_vec();
    if metadata.is_empty() || metadata.len() > 128 << 20 {
        return Err(invalid("model index exceeds its metadata limit"));
    }
    let header = archive::container_header(length, metadata.len() as u64, "flare-staging")?;
    let directory = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(directory)?;
    let (mut output, temporary) = archive::temporary(directory, "staging")?;
    let mut installed = false;
    let result = (|| {
        output.write_all(&header)?;
        let budget = IoBudget::for_ranks(1)?;
        for part in payload {
            let source = Source::open(&part.source.path, part.source.offset, part.source.nbytes)?;
            let destination = header.len() as u64 + part.offset;
            if destination % 4096 == 0 {
                budget.copy(&source, &output, destination)?;
            } else {
                let mut cursor = destination;
                budget.scan(&source, |block| {
                    super::io::write_all_at(&output, block, cursor)?;
                    cursor += block.len() as u64;
                    Ok(())
                })?;
                source.check_unchanged()?;
            }
        }
        use std::io::{Seek, SeekFrom};
        output.seek(SeekFrom::Start(header.len() as u64 + length))?;
        output.write_all(&metadata)?;
        output.sync_all()?;
        archive::install_file(&temporary, path)?;
        installed = true;
        archive::sync_directory(directory)
    })();
    let result = result.and(archive::remove_temporary(&temporary));
    if result.is_err() && installed {
        if let Ok(linked) = super::io::open_read(path) {
            if super::io::Identity::of(&output)?.same_file(&super::io::Identity::of(&linked)?) {
                fs::remove_file(path)?;
            }
        }
    }
    result
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
#[pyo3(signature=(path,index,payload,publisher,options,shard_size))]
fn export_publish(
    py: Python<'_>,
    path: String,
    index: Vec<u8>,
    payload: String,
    publisher: Option<String>,
    options: String,
    shard_size: u64,
) -> PyResult<String> {
    py.detach(|| {
        let index = wire::ModelIndex::decode(index.as_slice())?;
        let payload = decode_payload(&payload)?;
        let publisher = publisher
            .as_deref()
            .map(serde_json::from_str::<protection::PublisherRequest>)
            .transpose()?
            .map(|p| p.publisher())
            .transpose()?;
        let options = serde_json::from_str(&options)?;
        let result = publish(
            Path::new(&path),
            index,
            payload,
            publisher.as_ref(),
            &options,
            shard_size,
        )?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(serde_json::to_string(&result)?)
    })
    .map_err(py_error)
}
#[pyfunction]
#[pyo3(signature=(path,ranks,peer_memory,publisher,shard_size,cluster,cache,no_weights=false))]
fn export_combine(
    py: Python<'_>,
    path: String,
    ranks: Vec<String>,
    peer_memory: String,
    publisher: Option<String>,
    shard_size: u64,
    cluster: Option<String>,
    cache: Option<String>,
    no_weights: bool,
) -> PyResult<String> {
    py.detach(|| {
        let ranks: Result<Vec<_>> = ranks.iter().map(|p| read_sources(Path::new(p))).collect();
        let peers = serde_json::from_str(&peer_memory)?;
        let cluster = cluster.as_deref().map(serde_json::from_str).transpose()?;
        let (index, payload) = combine(&ranks?, peers, cluster)?;
        let publisher = publisher
            .as_deref()
            .map(serde_json::from_str::<protection::PublisherRequest>)
            .transpose()?
            .map(|p| p.publisher())
            .transpose()?;
        let options = ExportOptions {
            cache: cache.map(Into::into),
            no_weights,
        };
        let result = publish(
            Path::new(&path),
            index,
            payload,
            publisher.as_ref(),
            &options,
            shard_size,
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
) -> PyResult<()> {
    py.detach(|| {
        let index = wire::ModelIndex::decode(index.as_slice())?;
        let payload = decode_payload(&payload)?;
        write_sources(
            Path::new(&path),
            &RankOutput {
                index,
                payload,
                allocations: vec![],
                module_globals: vec![],
                external: vec![],
                staging: None,
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
            &serde_json::json!({"index":rank.index.encode_to_vec(),"payload":payload}),
        )?)
    })
    .map_err(py_error)
}
pub(crate) fn read_index(path: &Path) -> Result<(wire::ModelIndex, u64)> {
    read_index_container(Container::open(path)?)
}
fn read_index_container(container: Container) -> Result<(wire::ModelIndex, u64)> {
    let index = if container.format == "flare-staging" {
        wire::ModelIndex::decode(container.envelope.as_slice())?
    } else {
        let envelope = wire::ProtectedIndex::decode(container.envelope.as_slice())?;
        if envelope.version != 4 || envelope.groups.len() != 3 || envelope.ranks.is_empty() {
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
    module.add_function(wrap_pyfunction!(export_write_primary, module)?)?;
    module.add_function(wrap_pyfunction!(export_signature_message, module)?)?;
    module.add_function(wrap_pyfunction!(export_validate_publisher, module)?)?;
    module.add_function(wrap_pyfunction!(export_validate_license, module)?)?;
    super::protection::register(module)?;
    module.add_function(wrap_pyfunction!(export_publish, module)?)?;
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
fn export_write_primary(
    py: Python<'_>,
    path: String,
    envelope: Vec<u8>,
    layout: String,
) -> PyResult<()> {
    let layout: Layout = serde_json::from_str(&layout).map_err(py_error)?;
    py.detach(|| archive::install_snapshot(Path::new(&path), &envelope, &layout))
        .map_err(py_error)
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
mod staging_tests {
    use super::*;

    fn index() -> wire::ModelIndex {
        wire::ModelIndex {
            version: 4,
            ranks: vec![wire::BundleIndex::default()],
            ..Default::default()
        }
    }

    #[test]
    fn staging_publishes_complete_metadata_without_replacing_existing_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("nested/model.safetensors");
        write_staging(&path, &index(), &[]).unwrap();
        let container = Container::open(&path).unwrap();
        assert_eq!(container.format, "flare-staging");
        assert_eq!(
            wire::ModelIndex::decode(container.envelope.as_slice()).unwrap(),
            index()
        );
        let original = fs::read(&path).unwrap();
        assert!(write_staging(&path, &index(), &[]).is_err());
        assert_eq!(fs::read(&path).unwrap(), original);
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
    }

    #[test]
    fn staging_source_failure_leaves_no_primary_or_temporary() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        let payload = [FilePart {
            offset: 0,
            source: FileSource {
                path: directory.path().join("missing"),
                offset: 0,
                nbytes: 3,
            },
        }];
        assert!(write_staging(&path, &index(), &payload).is_err());
        assert!(!path.exists());
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn staging_header_failure_leaves_no_primary_or_temporary() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "export::package::staging_tests::staging_header_failure_child",
                "--ignored",
            ])
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
    }

    #[cfg(unix)]
    #[test]
    #[ignore]
    fn staging_header_failure_child() {
        unsafe {
            libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
        }
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        assert_eq!(
            unsafe { libc::getrlimit(libc::RLIMIT_FSIZE, &mut limit) },
            0
        );
        limit.rlim_cur = limit.rlim_cur.min(128);
        assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_FSIZE, &limit) }, 0);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        assert!(write_staging(&path, &index(), &[]).is_err());
        assert!(!path.exists());
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
    }
}
