//! Select serving graph variants without changing captured storage layouts.

use std::collections::{BTreeSet, HashMap, HashSet};

use serde_json::Value;

use super::format::{Package, invalid};
use crate::{Result, execution::BATCH_CAPACITIES};

#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    pub kv_cache_limit: super::KvCacheLimit,
    pub memory_limit: Option<super::MemoryLimit>,
    /// Auxiliary embedding-table storage; defaults to disk when absent.
    pub embedding_table_storage: Option<crate::engram::Storage>,
    /// Per-host RAM bytes for disk-backed embedding rows; zero disables, absent selects by memory topology.
    pub embedding_table_cache_bytes: Option<u64>,
    /// Keep original scale-plane bytes in host RAM while embedding weights remain on disk.
    pub embedding_table_resident_scales: bool,
    /// Disk workers per table: four on Windows, one elsewhere. Share 128 slots and 8 MiB staging.
    pub embedding_table_workers: Option<usize>,
    pub max_concurrency: Option<usize>,
    pub mtp_draft_tokens: Option<usize>,
    /// Select one exact packed verification capacity per batch before proposal sampling.
    pub mtp_verification_token_budget: Option<usize>,
    /// Retain every captured packed verification capacity at or below this total-row ceiling.
    pub mtp_max_verification_token_budget: Option<usize>,
    pub adaptive_verification: Option<bool>,
    /// User-visible ordinals when the launcher renumbers CUDA devices.
    pub device_labels: Option<Vec<i32>>,
    pub prefix_cache: crate::offload::PrefixCacheOffload,
    /// For a package whose ranks run on separate hosts: the one rank this host loads.
    pub cluster_rank: Option<usize>,
}

impl LoadOptions {
    /// Resolve the disk-reader default without treating it as an explicit storage override.
    pub fn embedding_workers(&self) -> usize {
        self.embedding_table_workers
            .unwrap_or(if cfg!(windows) { 4 } else { 1 })
    }

    pub(super) fn embedding_cache_bytes(&self, has_tables: bool, integrated: bool) -> u64 {
        self.embedding_table_cache_bytes.unwrap_or_else(|| {
            if has_tables
                && self.embedding_table_storage.unwrap_or_default() == crate::engram::Storage::Disk
            {
                if integrated { 1 << 30 } else { 4 << 30 }
            } else {
                0
            }
        })
    }

    pub(super) fn select(&self, packages: &mut [Package], config: &mut Value) -> Result<usize> {
        if self.mtp_verification_token_budget.is_some() && self.mtp_max_verification_token_budget.is_some() {
            return Err(invalid("exact and maximum verification token budgets are mutually exclusive"));
        }
        if let Some(limit) = self.memory_limit {
            limit.validate()?;
        }
        if let Some(workers) = self.embedding_table_workers {
            if !matches!(workers, 1 | 2 | 4 | 8) {
                return Err(invalid("embedding table workers must be 1, 2, 4, or 8"));
            }
            if self.embedding_table_storage == Some(crate::engram::Storage::Ram) {
                return Err(invalid("embedding table workers require disk storage"));
            }
            #[cfg(unix)]
            if workers != 1 {
                return Err(invalid("parallel embedding disk workers require Windows"));
            }
        }
        if self.embedding_table_cache_bytes.is_some_and(|bytes| bytes != 0)
            && self.embedding_table_storage == Some(crate::engram::Storage::Ram)
        {
            return Err(invalid("embedding row cache requires disk table storage"));
        }
        if self.embedding_table_resident_scales
            && self.embedding_table_storage == Some(crate::engram::Storage::Ram)
        {
            return Err(invalid("resident embedding scales require disk table storage; RAM storage already retains all planes"));
        }
        let execution = packages
            .first()
            .and_then(|p| p.execution.as_ref())
            .ok_or_else(|| invalid("model package has no serving contract"))?;
        let prepared = usize::try_from(execution.max_active)
            .map_err(|_| invalid("prepared concurrency exceeds usize"))?;
        let active = self.max_concurrency.unwrap_or(prepared);
        if !(1..=64).contains(&prepared) || active == 0 || active > prepared {
            return Err(invalid(format!(
                "runtime concurrency must be 1 through the captured maximum {prepared}"
            )));
        }
        let widths = config
            .get("speculative_widths")
            .map(|value| {
                value
                    .as_array()
                    .ok_or_else(|| invalid("speculative_widths must be an array"))?
                    .iter()
                    .map(|value| {
                        value
                            .as_u64()
                            .and_then(|n| usize::try_from(n).ok())
                            .ok_or_else(|| invalid("invalid speculative width"))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        crate::scheduler::speculation::validate_widths(&widths)?;
        let token_capacity = config.get("speculative_token_capacity")
            .and_then(Value::as_u64).unwrap_or(0);
        if widths.last().is_some_and(|w| token_capacity < *w as u64) {
            return Err(invalid("speculative token capacity must cover the maximum verification width"));
        }
        let packed_round = execution.graphs.iter().any(|g| g.kind == "packed_speculative");
        let maximum = execution
            .graphs
            .iter()
            .filter_map(|g| g.draft_tokens)
            .max()
            .unwrap_or(0) as usize;
        let drafts = self.mtp_draft_tokens.unwrap_or(maximum);
        if drafts > maximum {
            return Err(invalid(format!(
                "runtime MTP draft tokens must be 0 through the captured maximum {maximum}"
            )));
        }
        let fixed = config
            .get("speculative_fixed_width")
            .filter(|v| !v.is_null())
            .map(|v| {
                v.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| invalid("speculative_fixed_width must be an integer or null"))
            })
            .transpose()?;
        if fixed.is_some_and(|width| !widths.contains(&width)) {
            return Err(invalid(
                "fixed verification width is outside the captured widths",
            ));
        }
        if widths.is_empty() && self.adaptive_verification.is_some() {
            return Err(invalid(
                "verification policy requires a package captured with speculation",
            ));
        }
        let adaptive = self.adaptive_verification.unwrap_or(fixed.is_none());
        let fixed = if widths.is_empty() || adaptive {
            None
        } else {
            Some(if self.mtp_draft_tokens.is_some() {
                widths
                    .iter()
                    .copied()
                    .filter(|w| *w <= drafts + 1)
                    .max()
                    .unwrap_or(1)
            } else {
                fixed.unwrap_or_else(|| widths.last().copied().unwrap_or(1).min(drafts + 1))
            })
        };
        let enabled = drafts > 0 && fixed != Some(1) && !widths.is_empty();
        let fixed = enabled.then_some(fixed).flatten();
        let widths: Vec<_> = widths
            .into_iter()
            .filter(|width| enabled && (packed_round || *width <= drafts + 1))
            .collect();
        let source = match config
            .get("speculative_proposal_source")
            .and_then(Value::as_str)
            .unwrap_or("mtp")
        {
            "mtp" => 1,
            "copy" => 2,
            _ => return Err(invalid("unknown speculative proposal source")),
        };
        let split = execution.graphs.iter().any(|g| g.kind == "proposal");
        if enabled && execution.graphs.iter().any(|g| g.packed_token_capacity != 0)
            && (fixed.is_some() || (!packed_round && drafts != maximum))
        {
            return Err(invalid("packed verification requires adaptive execution at the captured draft depth"));
        }
        let packed_width = execution.graphs.iter()
            .filter(|g| matches!(g.kind.as_str(), "verification" | "packed_speculative") && g.packed_token_capacity != 0)
            .map(|g| g.token_capacity).max();
        let verification_token_limit = self.mtp_verification_token_budget
            .or(self.mtp_max_verification_token_budget);
        if let Some(budget) = verification_token_limit {
            if !enabled || !adaptive {
                return Err(invalid("verification token budget requires enabled adaptive speculation"));
            }
            if !(split || packed_round) || packed_width.is_none() {
                return Err(invalid("verification token budget requires captured packed verification graphs"));
            }
            if budget < active.next_power_of_two() && !(packed_round && !split) {
                return Err(invalid(format!(
                    "verification token budget must cover the selected batch capacity {}",
                    active.next_power_of_two()
                )));
            }
            if self.mtp_max_verification_token_budget.is_some()
                && budget < *widths.last().unwrap() {
                return Err(invalid("maximum verification token budget must cover the captured verification width"));
            }
        }
        let selected_token_capacity = verification_token_limit
            .map_or(token_capacity, |budget| token_capacity.min(budget as u64));
        let mut rank_catalog = None;
        for (rank, package) in packages.iter_mut().enumerate() {
            let execution = package
                .execution
                .as_mut()
                .ok_or_else(|| invalid("rank lacks serving contract"))?;
            if execution.max_active != prepared as u64 {
                return Err(invalid("rank concurrency capacities differ"));
            }
            let mut names = HashSet::new();
            let mut variants = HashSet::new();
            for graph in &execution.graphs {
                let kind = crate::execution::GraphKind::try_from(graph.kind.as_str())?;
                crate::execution::SamplingMode::try_from(graph.sampling_mode.as_str())?;
                let speculative = kind == crate::execution::GraphKind::Speculative
                    || (kind == crate::execution::GraphKind::Mixed
                        && graph.speculation_enabled == Some(true));
                let valid = match kind {
                    crate::execution::GraphKind::Mixed => {
                        BATCH_CAPACITIES.contains(&(graph.batch_capacity as usize))
                            && graph.batch_capacity >= 2
                            && graph.batch_capacity <= prepared.next_power_of_two() as u64
                            && graph.prefill_capacity >= 64
                            && graph.prefill_capacity.is_multiple_of(64)
                            && graph.packed_tokens() <= i32::MAX as u64
                            && graph.prefill_capacity
                                <= config["prefill_chunk_size"].as_u64().unwrap_or(0)
                            && graph.speculation_enabled.is_some()
                            && if speculative {
                                (1..=8).contains(&graph.token_capacity)
                            } else {
                                graph.token_capacity == 1
                            }
                    }
                    crate::execution::GraphKind::Prefill | crate::execution::GraphKind::ScoredPrefill | crate::execution::GraphKind::Vision => {
                        graph.batch_capacity == 1 && graph.token_capacity > 0
                    }
                    _ => {
                        BATCH_CAPACITIES.contains(&(graph.batch_capacity as usize))
                            && graph.batch_capacity <= prepared.next_power_of_two() as u64
                            && if matches!(kind, crate::execution::GraphKind::Speculative | crate::execution::GraphKind::Verification | crate::execution::GraphKind::PackedSpeculative) {
                                (1..=8).contains(&graph.token_capacity)
                            } else if kind == crate::execution::GraphKind::Proposal {
                                (2..=8).contains(&graph.token_capacity)
                            } else {
                                graph.token_capacity == 1
                            }
                    }
                };
                if !valid
                    || (graph.packed_token_capacity != 0
                        && (!matches!(kind, crate::execution::GraphKind::Verification | crate::execution::GraphKind::PackedSpeculative)
                            || graph.packed_token_capacity < graph.batch_capacity
                            || graph.packed_token_capacity > graph.batch_capacity.saturating_mul(graph.token_capacity)))
                    || (kind != crate::execution::GraphKind::Mixed && graph.prefill_capacity != 0)
                    || (kind == crate::execution::GraphKind::Speculative
                        && graph.speculation_enabled == Some(false))
                    || (kind == crate::execution::GraphKind::Decode
                        && graph.speculation_enabled == Some(true))
                    || if kind == crate::execution::GraphKind::PackedSpeculative {
                        graph.verify_tokens != Some(graph.token_capacity as u32)
                            || graph.draft_tokens.is_none_or(|n| !(1..=7).contains(&n) || n + 1 > graph.token_capacity as u32)
                            || graph.draft_source != 1 || graph.packed_token_capacity == 0
                            || graph.speculation_enabled == Some(false)
                    } else if speculative {
                        graph.verify_tokens != Some(graph.token_capacity as u32)
                            || graph
                                .draft_tokens
                                .is_none_or(|n| n > 7 || n + 1 < graph.token_capacity as u32)
                            || ![1, 2].contains(&graph.draft_source)
                    } else if kind == crate::execution::GraphKind::Proposal {
                        graph.draft_tokens != Some(graph.token_capacity as u32 - 1)
                            || graph.verify_tokens.is_some() || graph.draft_source != 1
                    } else if kind == crate::execution::GraphKind::Verification {
                        graph.draft_tokens.is_some()
                            || graph.verify_tokens != Some(graph.token_capacity as u32)
                            || graph.draft_source != 0
                    } else {
                        graph.draft_tokens.is_some()
                            || graph.verify_tokens.is_some()
                            || graph.draft_source != 0
                    }
                    || ((speculative || matches!(kind, crate::execution::GraphKind::Verification | crate::execution::GraphKind::PackedSpeculative))
                        && graph.token_capacity > 1
                        && (if graph.packed_token_capacity == 0 {
                            graph.batch_capacity.saturating_mul(graph.token_capacity)
                        } else {
                            graph.packed_token_capacity
                        }) > token_capacity)
                    || !variants.insert((
                        graph.kind.clone(),
                        graph.sampling_mode.clone(),
                        graph.batch_capacity,
                        graph.token_capacity,
                        graph.prefill_capacity,
                        graph.packed_token_capacity,
                        graph.draft_tokens,
                        graph.draft_source,
                        graph.speculation_enabled,
                    ))
                {
                    return Err(invalid("invalid or duplicate serving graph variant"));
                }
                if !names.insert(graph.name.clone()) {
                    return Err(invalid("duplicate execution graph name"));
                }
            }
            let resources: HashSet<_> = package
                .graphs
                .iter()
                .map(|graph| graph.name.clone())
                .collect();
            if names != resources {
                return Err(invalid(
                    "execution catalog must describe every captured graph exactly once",
                ));
            }
            execution.graphs.retain(|graph| {
                graph.batch_capacity <= active.next_power_of_two() as u64
                    && graph
                        .speculation_enabled
                        .is_none_or(|value| value == enabled)
                    && (graph.kind != "decode" || !enabled)
                    && (graph.kind != "speculative" || graph.token_capacity == 1
                        || graph.batch_capacity * graph.token_capacity <= selected_token_capacity)
                    && (graph.kind != "proposal" || (enabled
                        && graph.draft_tokens == Some(drafts as u32)))
                    && (graph.kind != "packed_speculative" || (enabled
                        && graph.draft_tokens == Some(drafts as u32)
                        && widths.contains(&(graph.token_capacity as usize))
                        && self.mtp_verification_token_budget.is_none_or(|budget| graph.packed_token_capacity == (budget as u64).min(graph.batch_capacity * graph.token_capacity))
                        && self.mtp_max_verification_token_budget.is_none_or(|budget| graph.packed_token_capacity <= budget as u64)))
                    && (graph.kind != "verification" || (enabled
                        && widths.contains(&(graph.token_capacity as usize))
                        && self.mtp_verification_token_budget.is_none_or(|budget|
                            graph.token_capacity == packed_width.unwrap()
                                && graph.packed_token_capacity == (budget as u64)
                                    .min(graph.batch_capacity * graph.token_capacity))
                        && self.mtp_max_verification_token_budget.is_none_or(|budget|
                            graph.packed_token_capacity != 0
                                && graph.packed_token_capacity <= budget as u64)
                        && (graph.token_capacity == 1 || fixed.is_none_or(|w| graph.token_capacity == w as u64))))
                    && (!(graph.kind == "speculative" || (graph.kind == "mixed" && enabled))
                        || (graph.draft_source == source
                            && graph.draft_tokens.is_some_and(|n| n as usize <= drafts)
                            && widths.contains(&(graph.verify_tokens.unwrap() as usize))
                            && (graph.token_capacity == 1 || fixed.is_none_or(|width| graph.token_capacity == width as u64))))
            });
            let mut draft_counts = HashMap::new();
            for graph in &execution.graphs {
                if let Some(drafts) = graph.draft_tokens.filter(|_| graph.kind != "packed_speculative") {
                    let key = (
                        graph.kind.clone(),
                        graph.batch_capacity,
                        graph.token_capacity,
                        graph.prefill_capacity,
                        graph.sampling_mode.clone(),
                    );
                    let count = draft_counts.entry(key).or_insert(drafts);
                    *count = (*count).max(drafts);
                }
            }
            execution.graphs.retain(|g| {
                g.kind == "packed_speculative" || g.draft_tokens.is_none_or(|drafts| {
                    draft_counts[&(
                        g.kind.clone(),
                        g.batch_capacity,
                        g.token_capacity,
                        g.prefill_capacity,
                        g.sampling_mode.clone(),
                    )] == drafts
                })
            });
            if enabled && packed_round && !execution.graphs.iter().any(|g| g.kind == "packed_speculative") {
                return Err(invalid(format!("rank {rank} has no packed combined graphs within the selected proposal and row limits")));
            }
            let mut selected_variants = HashSet::new();
            for graph in &execution.graphs {
                if !selected_variants.insert((
                    graph.kind.as_str(),
                    graph.sampling_mode.as_str(),
                    graph.batch_capacity,
                    graph.token_capacity,
                    graph.prefill_capacity,
                    graph.packed_token_capacity,
                    (graph.kind == "packed_speculative").then_some(graph.draft_tokens),
                )) {
                    return Err(invalid(
                        "ambiguous graph variants for the selected speculation mode",
                    ));
                }
            }
            for batch in BATCH_CAPACITIES
                .into_iter()
                .filter(|batch| *batch <= active.next_power_of_two())
            {
                for kind in [
                    "init",
                    "restore",
                    "snapshot",
                    "resume",
                    if widths.is_empty() {
                        "decode"
                    } else {
                        "speculative"
                    },
                ] {
                    if !execution
                        .graphs
                        .iter()
                        .any(|g| g.kind == kind && g.batch_capacity == batch as u64
                            && (kind != "speculative" || !packed_round || g.token_capacity == 1))
                    {
                        return Err(invalid(format!(
                            "rank {rank} has no captured {kind} graph for batch {batch} within the requested verification limits; export the required capacity range"
                        )));
                    }
                }
                if enabled && packed_round {
                    if batch as u64 * 2 <= selected_token_capacity
                        && !execution.graphs.iter().any(|g| g.kind == "packed_speculative"
                        && g.batch_capacity == batch as u64) {
                        return Err(invalid(format!("rank {rank} lacks packed combined batch {batch} within the selected proposal and row limits")));
                    }
                    continue;
                }
                if enabled {
                    if let Some(budget) = self.mtp_verification_token_budget {
                        let width = packed_width.unwrap();
                        let capacity = (budget as u64).min(batch as u64 * width);
                        if !execution.graphs.iter().any(|g| g.kind == "verification"
                            && g.batch_capacity == batch as u64 && g.token_capacity == width
                            && g.packed_token_capacity == capacity) {
                            return Err(invalid(format!(
                                "rank {rank} lacks exact packed verification graph for batch {batch}, width {width}, token capacity {capacity}"
                            )));
                        }
                    }
                    let mut required = Vec::new();
                    if split {
                        required.push(("proposal", drafts + 1));
                    }
                    for &width in &widths {
                        if width == 1 || (batch * width <= selected_token_capacity as usize
                            && fixed.is_none_or(|w| w == width)) {
                            required.push((if split { "verification" } else { "speculative" }, width));
                        }
                    }
                    for (kind, width) in required {
                        if !execution.graphs.iter().any(|g| g.kind == kind
                            && g.batch_capacity == batch as u64
                            && (g.token_capacity == width as u64 && g.packed_token_capacity == 0
                                || kind == "verification" && g.packed_token_capacity >= (batch * width) as u64
                                    && g.token_capacity >= width as u64)) {
                            return Err(invalid(format!("rank {rank} lacks required {kind} graph for batch {batch}, width {width}")));
                        }
                    }
                }
            }
            let catalog: BTreeSet<_> = execution
                .graphs
                .iter()
                .map(|g| {
                    (
                        g.name.clone(),
                        g.kind.clone(),
                        g.sampling_mode.clone(),
                        g.batch_capacity,
                        g.token_capacity,
                        g.prefill_capacity,
                        g.packed_token_capacity,
                        g.draft_tokens,
                        g.verify_tokens,
                        g.draft_source,
                    )
                })
                .collect();
            if rank_catalog
                .as_ref()
                .is_some_and(|expected| *expected != catalog)
            {
                return Err(invalid("selected graph catalogs differ between ranks"));
            }
            rank_catalog = Some(catalog);
            let selected: HashSet<_> = execution.graphs.iter().map(|g| g.name.as_str()).collect();
            package
                .graphs
                .retain(|graph| selected.contains(graph.name.as_str()));
            prune_kernels(package)?;
        }
        let object = config
            .as_object_mut()
            .ok_or_else(|| invalid("server configuration must be an object"))?;
        object.insert("max_active".into(), active.into());
        object.insert("speculative_widths".into(), serde_json::json!(widths));
        object.insert("speculative_fixed_width".into(), serde_json::json!(fixed));
        if verification_token_limit.is_some() {
            object.insert("speculative_token_capacity".into(), selected_token_capacity.into());
        }
        if !enabled {
            object.insert("history_drafts".into(), false.into());
        }
        Ok(active)
    }
}

fn prune_kernels(package: &mut Package) -> Result<()> {
    let used: HashSet<_> = package
        .graphs
        .iter()
        .flat_map(|g| g.functions.iter().copied())
        .collect();
    package.functions.retain(|f| used.contains(&f.id));
    let modules: HashSet<_> = package
        .functions
        .iter()
        .map(|f| f.module.as_str())
        .collect();
    // Module globals can own persistent data referenced through opaque kernel arguments.
    package
        .modules
        .retain(|module| modules.contains(module.name.as_str()) || !module.globals.is_empty());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_package::{format::*, protobuf::wire};

    fn package() -> Package {
        let mut execution = wire::Execution {
            max_active: 4,
            input_frame_bytes: 8192,
            ..Default::default()
        };
        for batch in [1, 2, 4] {
            for enabled in [false, true] {
                for kind in ["init", "restore", "snapshot", "resume", "decode"] {
                    if enabled && kind == "decode" {
                        continue;
                    }
                    execution.graphs.push(wire::ExecutionGraph {
                        name: format!("{kind}{batch}-{}", if enabled { "mtp" } else { "no-mtp" }),
                        kind: kind.try_into().unwrap(),
                        sampling_mode: "general".into(),
                        estimated_duration_ns: 1000,
                        batch_capacity: batch,
                        token_capacity: 1,
                        speculation_enabled: Some(enabled),
                        ..Default::default()
                    });
                }
            }
            for width in 1..=6 {
                execution.graphs.push(wire::ExecutionGraph {
                    name: format!("speculative{batch}x{width}"),
                    kind: "speculative".into(),
                    sampling_mode: "general".into(),
                    estimated_duration_ns: 1000,
                    batch_capacity: batch,
                    token_capacity: width,
                    draft_tokens: Some(width as u32 - 1),
                    verify_tokens: Some(width as u32),
                    draft_source: wire::DraftSource::Mtp as i32,
                    ..Default::default()
                });
            }
        }
        let graphs = execution
            .graphs
            .iter()
            .enumerate()
            .map(|(function, g)| Graph {
                name: g.name.clone(),
                flags: 0,
                resource: g.name.clone(),
                node_count: 1,
                edge_count: 0,
                functions: vec![function],
            })
            .collect();
        let functions = execution
            .graphs
            .iter()
            .enumerate()
            .map(|(id, g)| Function {
                id,
                module: g.name.clone(),
                symbol: "kernel".into(),
                parameter_sizes: vec![],
                attributes: vec![],
            })
            .collect();
        let modules = execution
            .graphs
            .iter()
            .map(|g| Module {
                name: g.name.clone(),
                file: format!("{}.cubin", g.name),
                globals: vec![],
            })
            .collect();
        Package {
            resources: std::collections::HashMap::new(),
            events: vec![],
            architecture: [12, 0],
            graphs,
            functions,
            modules,
            execution: Some(execution),
            context: Context::default(),
            reservations: vec![],
            allocations: ["weights", "mtp-weights", "scratch"]
                .map(|name| Allocation {
                    name: name.into(),
                    nbytes: 8192,
                    alignment: 256,
                    file: None,
                    data_spans: vec![],
                    relocations: vec![],
                    address: None,
                    memory: Memory::Device,
                    sizing: None,
                })
                .into(),
        }
    }

    fn config() -> Value {
        serde_json::json!({"speculative_token_capacity": 24, "speculative_widths": [1,2,3,4,5,6], "speculative_fixed_width": null})
    }

    #[test]
    fn embedding_worker_options_reject_invalid_counts_and_resident_storage() {
        assert_eq!(LoadOptions::default().embedding_workers(), if cfg!(windows) { 4 } else { 1 });
        let resident = LoadOptions { embedding_table_storage: Some(crate::engram::Storage::Ram), ..Default::default() };
        resident.select(&mut [package()], &mut config()).unwrap();
        let single = LoadOptions { embedding_table_workers: Some(1), ..Default::default() };
        assert_eq!(single.embedding_workers(), 1);
        single.select(&mut [package()], &mut config()).unwrap();
        for workers in [0, 3, 5] {
            let options = LoadOptions { embedding_table_workers: Some(workers), ..Default::default() };
            assert!(options.select(&mut [package()], &mut config()).unwrap_err().to_string().contains("workers must be"));
        }
        let options = LoadOptions { embedding_table_workers: Some(2), embedding_table_storage: Some(crate::engram::Storage::Ram), ..Default::default() };
        assert!(options.select(&mut [package()], &mut config()).unwrap_err().to_string().contains("require disk storage"));
    }

    #[test]
    fn indeterminate_graph_durations_are_loadable() {
        let mut package = package();
        package.execution.as_mut().unwrap().graphs[0].estimated_duration_ns = 0;
        LoadOptions::default()
            .select(&mut [package], &mut config())
            .unwrap();
    }

    #[test]
    fn embedding_cache_defaults_follow_storage_and_memory_topology() {
        use crate::engram::Storage;
        for storage in [None, Some(Storage::Disk)] {
            let mut options = LoadOptions { embedding_table_storage: storage, ..Default::default() };
            options.select(&mut [package()], &mut config()).unwrap();
            assert_eq!(options.embedding_cache_bytes(true, false), 4 << 30);
            assert_eq!(options.embedding_cache_bytes(true, true), 1 << 30);
            assert_eq!(options.embedding_cache_bytes(false, false), 0);
            assert_eq!(options.embedding_cache_bytes(false, true), 0);
            for bytes in [0, 2 << 30, 8 << 30] {
                options.embedding_table_cache_bytes = Some(bytes);
                options.select(&mut [package()], &mut config()).unwrap();
                assert_eq!(options.embedding_cache_bytes(true, false), bytes);
                assert_eq!(options.embedding_cache_bytes(true, true), bytes);
            }
        }
        for bytes in [None, Some(0)] {
            let options = LoadOptions {
                embedding_table_storage: Some(Storage::Ram),
                embedding_table_cache_bytes: bytes,
                ..Default::default()
            };
            options.select(&mut [package()], &mut config()).unwrap();
            assert_eq!(options.embedding_cache_bytes(true, false), 0);
            assert_eq!(options.embedding_cache_bytes(true, true), 0);
        }
    }

    #[test]
    fn cached_disk_rows_reject_resident_table_storage_before_cuda() {
        let options = LoadOptions {
            embedding_table_storage: Some(crate::engram::Storage::Ram),
            embedding_table_cache_bytes: Some(1 << 30),
            ..Default::default()
        };
        assert!(
            options
                .select(&mut [package()], &mut config())
                .unwrap_err()
                .to_string()
                .contains("embedding row cache requires disk table storage")
        );
    }

    #[test]
    fn resident_scales_allow_cached_disk_rows_and_reject_whole_table_ram() {
        let mut options = LoadOptions {
            embedding_table_resident_scales: true,
            embedding_table_cache_bytes: Some(1 << 30),
            ..Default::default()
        };
        options.select(&mut [package()], &mut config()).unwrap();
        options.embedding_table_cache_bytes = Some(0);
        options.embedding_table_storage = Some(crate::engram::Storage::Ram);
        assert!(options.select(&mut [package()], &mut config()).unwrap_err().to_string()
            .contains("resident embedding scales require disk table storage"));
    }

    #[test]
    fn offload_identity_separates_selected_state_without_fragmenting_compatible_limits() {
        let mut identities = HashMap::new();
        for source in ["mtp", "copy"] {
            for concurrency in 1..=4 {
                for drafts in 0..=5 {
                    for adaptive in [false, true] {
                        let mut package = package();
                        for graph in &mut package.execution.as_mut().unwrap().graphs {
                            if graph.kind == "speculative" {
                                graph.draft_source = if source == "mtp" { 1 } else { 2 };
                            }
                        }
                        let mut config = config();
                        config["identity"] = "01".repeat(32).into();
                        config["speculative_proposal_source"] = source.into();
                        LoadOptions {
                            max_concurrency: Some(concurrency),
                            mtp_draft_tokens: Some(drafts),
                            adaptive_verification: Some(adaptive),
                            ..Default::default()
                        }
                        .select(std::slice::from_mut(&mut package), &mut config)
                        .unwrap();
                        let identity = super::super::model::offload_identity(&config).unwrap();
                        assert_ne!(identity, [1; 32]);
                        let state = if drafts == 0 { "ordinary" } else { source };
                        assert_eq!(*identities.entry(state).or_insert(identity), identity);
                    }
                }
            }
        }
        assert_eq!(identities.values().collect::<HashSet<_>>().len(), 3);

        let mut config = config();
        config["identity"] = "01".repeat(32).into();
        config["speculative_fixed_width"] = 1.into();
        LoadOptions::default()
            .select(&mut [package()], &mut config)
            .unwrap();
        assert_eq!(
            super::super::model::offload_identity(&config).unwrap(),
            identities["ordinary"]
        );
        config["identity"] = "02".repeat(32).into();
        assert_ne!(
            super::super::model::offload_identity(&config).unwrap(),
            identities["ordinary"]
        );
    }

    #[test]
    fn offload_identity_respects_default_proposals_and_overridden_fixed_policy() {
        let mut adaptive_config = config();
        adaptive_config["identity"] = "ab".repeat(32).into();
        LoadOptions::default()
            .select(&mut [package()], &mut adaptive_config)
            .unwrap();
        let speculative = super::super::model::offload_identity(&adaptive_config).unwrap();
        adaptive_config["speculative_proposal_source"] = "mtp".into();
        adaptive_config["identity"] = "AB".repeat(32).into();
        assert_eq!(
            super::super::model::offload_identity(&adaptive_config).unwrap(),
            speculative
        );
        for adaptive in [None, Some(false), Some(true)] {
            let mut config = config();
            config["identity"] = "ab".repeat(32).into();
            config["speculative_fixed_width"] = 1.into();
            LoadOptions {
                adaptive_verification: adaptive,
                ..Default::default()
            }
            .select(&mut [package()], &mut config)
            .unwrap();
            assert_eq!(
                super::super::model::offload_identity(&config).unwrap() == speculative,
                adaptive == Some(true)
            );
        }
    }

    #[test]
    fn draft_count_and_verification_width_are_independent_selection_axes() {
        for limit in [3, 5] {
            let mut p = package();
            for drafts in [3, 5] {
                let name = format!("speculative2-mtp{drafts}-verify2");
                p.execution
                    .as_mut()
                    .unwrap()
                    .graphs
                    .push(wire::ExecutionGraph {
                        name: name.clone(),
                        kind: "speculative".into(),
                        sampling_mode: "general".into(),
                        estimated_duration_ns: 1000,
                        batch_capacity: 2,
                        token_capacity: 2,
                        draft_tokens: Some(drafts),
                        verify_tokens: Some(2),
                        draft_source: 1,
                        ..Default::default()
                    });
                p.graphs.push(Graph {
                    name,
                    resource: "graph".into(),
                    node_count: 1,
                    edge_count: 0,
                    flags: 0,
                    functions: vec![0],
                });
            }
            LoadOptions {
                max_concurrency: Some(2),
                mtp_draft_tokens: Some(limit),
                adaptive_verification: Some(true),
                ..Default::default()
            }
            .select(std::slice::from_mut(&mut p), &mut config())
            .unwrap();
            let variants: Vec<_> = p
                .execution
                .unwrap()
                .graphs
                .into_iter()
                .filter(|g| {
                    g.kind == "speculative" && g.batch_capacity == 2 && g.verify_tokens == Some(2)
                })
                .collect();
            assert_eq!(variants.len(), 1);
            assert_eq!(variants[0].draft_tokens, Some(limit as u32));
        }
    }

    #[test]
    fn draft_limit_comes_from_draft_declarations_not_verification_capacity() {
        let mut p = package();
        p.execution
            .as_mut()
            .unwrap()
            .graphs
            .retain(|g| g.token_capacity <= 2);
        for graph in &mut p.execution.as_mut().unwrap().graphs {
            if graph.kind == "speculative" && graph.token_capacity == 2 {
                graph.draft_tokens = Some(5);
            }
        }
        let names: HashSet<_> = p
            .execution
            .as_ref()
            .unwrap()
            .graphs
            .iter()
            .map(|g| g.name.clone())
            .collect();
        p.graphs.retain(|g| names.contains(&g.name));
        let mut config = serde_json::json!({"speculative_token_capacity": 24, "speculative_widths": [1,2]});
        LoadOptions {
            mtp_draft_tokens: Some(5),
            adaptive_verification: Some(false),
            ..Default::default()
        }
        .select(std::slice::from_mut(&mut p), &mut config)
        .unwrap();
        assert_eq!(config["speculative_fixed_width"], 2);
        assert!(
            p.execution
                .unwrap()
                .graphs
                .iter()
                .filter(|g| g.kind == "speculative")
                .all(|g| (g.draft_tokens == Some(0) && g.verify_tokens == Some(1))
                    || (g.draft_tokens == Some(5) && g.verify_tokens == Some(2)))
        );
    }

    #[test]
    fn every_concurrency_and_width_reduction_selects_only_required_graphs() {
        for concurrency in 1usize..=4 {
            for drafts in 0..=5 {
                for adaptive in [false, true] {
                    let mut packages = vec![package(), package()];
                    let mut config = config();
                    let options = LoadOptions {
                        max_concurrency: Some(concurrency),
                        mtp_draft_tokens: Some(drafts),
                        adaptive_verification: Some(adaptive),
                        ..Default::default()
                    };
                    options.select(&mut packages, &mut config).unwrap();
                    let batches = concurrency.next_power_of_two().ilog2() as usize + 1;
                    let expected = batches * (4 + if adaptive { drafts + 1 } else if drafts == 0 { 1 } else { 2 });
                    for package in &packages {
                        let execution = package.execution.as_ref().unwrap();
                        assert_eq!(execution.max_active, 4);
                        assert_eq!(execution.input_frame_bytes, 8192);
                        assert_eq!(package.graphs.len(), expected);
                        assert_eq!(package.functions.len(), expected);
                        assert_eq!(package.modules.len(), expected);
                        assert_eq!(package.allocations.len(), 3);
                        assert_eq!(package.allocations[1].name, "mtp-weights");
                        for graph in &package.graphs {
                            assert_eq!(
                                package
                                    .functions
                                    .iter()
                                    .find(|f| f.id == graph.functions[0])
                                    .unwrap()
                                    .module,
                                graph.name
                            );
                        }
                    }
                    assert_eq!(config["max_active"], concurrency);
                    assert_eq!(
                        config["speculative_widths"].as_array().unwrap().len(),
                        if drafts == 0 { 0 } else { drafts + 1 }
                    );
                }
            }
        }
    }

    #[test]
    fn invalid_limits_and_missing_graphs_fail_before_resource_loading() {
        for options in [
            LoadOptions {
                max_concurrency: Some(0),
                ..Default::default()
            },
            LoadOptions {
                max_concurrency: Some(5),
                ..Default::default()
            },
            LoadOptions {
                mtp_draft_tokens: Some(6),
                ..Default::default()
            },
        ] {
            assert!(options.select(&mut [package()], &mut config()).is_err());
        }
        let mut p = package();
        p.graphs.retain(|g| g.name != "speculative2x3");
        p.execution
            .as_mut()
            .unwrap()
            .graphs
            .retain(|g| g.name != "speculative2x3");
        let options = LoadOptions {
            mtp_draft_tokens: Some(2),
            adaptive_verification: Some(false),
            ..Default::default()
        };
        assert!(
            options
                .select(&mut [p], &mut config())
                .unwrap_err()
                .to_string()
                .contains("batch 2")
        );
    }

    #[test]
    fn tensor_parallel_catalogs_must_match() {
        let mut packages = vec![package(), package()];
        packages[1].graphs.retain(|g| g.name != "speculative2x3");
        packages[1]
            .execution
            .as_mut()
            .unwrap()
            .graphs
            .retain(|g| g.name != "speculative2x3");
        assert!(
            LoadOptions::default()
                .select(&mut packages, &mut config())
                .unwrap_err()
                .to_string()
                .contains("lacks required speculative graph")
        );
    }

    #[test]
    fn policy_defaults_and_overrides_select_the_recorded_catalog() {
        let mut fixed_config = config();
        fixed_config["speculative_fixed_width"] = 6.into();
        let mut packages = [package()];
        LoadOptions::default()
            .select(&mut packages, &mut fixed_config)
            .unwrap();
        assert_eq!(packages[0].graphs.len(), 18);
        let mut packages = [package()];
        LoadOptions {
            mtp_draft_tokens: Some(2),
            adaptive_verification: Some(true),
            ..Default::default()
        }
        .select(&mut packages, &mut fixed_config)
        .unwrap();
        assert!(fixed_config["speculative_fixed_width"].is_null());
        assert_eq!(packages[0].graphs.len(), 21);
    }

    #[test]
    fn globals_remain_even_when_their_module_has_no_selected_kernel() {
        let mut package = package();
        package.modules.push(Module {
            name: "data".into(),
            file: "data.cubin".into(),
            globals: vec![Global {
                symbol: "weights".into(),
                file: "weights.bin".into(),
                nbytes: 8,
                relocations: vec![],
            }],
        });
        let mut packages = [package];
        LoadOptions {
            max_concurrency: Some(1),
            mtp_draft_tokens: Some(0),
            ..Default::default()
        }
        .select(&mut packages, &mut config())
        .unwrap();
        assert_eq!(packages[0].graphs.len(), 5);
        assert_eq!(packages[0].modules.len(), 6);
        assert_eq!(packages[0].modules.last().unwrap().name, "data");
    }

    #[test]
    fn pinned_zero_selects_ordinary_operations_and_disables_draft_scheduling() {
        for adaptive in [None, Some(false), Some(true)] {
            let mut p = package();
            let mut config = config();
            config["history_drafts"] = true.into();
            LoadOptions {
                mtp_draft_tokens: Some(0),
                adaptive_verification: adaptive,
                ..Default::default()
            }
            .select(std::slice::from_mut(&mut p), &mut config)
            .unwrap();
            assert_eq!(config["speculative_widths"], serde_json::json!([]));
            assert!(config["speculative_fixed_width"].is_null());
            assert_eq!(config["history_drafts"], false);
            assert!(
                p.execution
                    .as_ref()
                    .unwrap()
                    .graphs
                    .iter()
                    .all(|g| g.speculation_enabled == Some(false))
            );
            assert_eq!(
                p.execution
                    .as_ref()
                    .unwrap()
                    .graphs
                    .iter()
                    .filter(|g| g.kind == "decode")
                    .count(),
                3
            );
        }
        let mut config = config();
        config["speculative_fixed_width"] = 1.into();
        LoadOptions::default()
            .select(&mut [package()], &mut config)
            .unwrap();
        assert_eq!(config["speculative_widths"], serde_json::json!([]));
    }

    #[test]
    fn pinned_zero_requires_ordinary_decode_instead_of_using_speculative_width_one() {
        let mut p = package();
        let names: HashSet<_> = p
            .execution
            .as_ref()
            .unwrap()
            .graphs
            .iter()
            .filter(|g| g.kind == "decode")
            .map(|g| g.name.clone())
            .collect();
        p.execution
            .as_mut()
            .unwrap()
            .graphs
            .retain(|g| !names.contains(&g.name));
        p.graphs.retain(|g| !names.contains(&g.name));
        let error = LoadOptions {
            mtp_draft_tokens: Some(0),
            ..Default::default()
        }
        .select(&mut [p], &mut config())
        .unwrap_err();
        assert!(error.to_string().contains("no captured decode graph"));
    }

    #[test]
    fn prefill_selection_matches_the_decode_mode() {
        for drafts in [0, 5] {
            let mut p = package();
            for enabled in [false, true] {
                let name = format!("prefill64-{enabled}");
                p.execution
                    .as_mut()
                    .unwrap()
                    .graphs
                    .push(wire::ExecutionGraph {
                        name: name.clone(),
                        kind: "prefill".into(),
                        sampling_mode: "general".into(),
                        estimated_duration_ns: 1000,
                        batch_capacity: 1,
                        token_capacity: 64,
                        speculation_enabled: Some(enabled),
                        ..Default::default()
                    });
                p.graphs.push(Graph {
                    name,
                    resource: "prefill".into(),
                    node_count: 1,
                    edge_count: 0,
                    flags: 0,
                    functions: vec![0],
                });
            }
            LoadOptions {
                mtp_draft_tokens: Some(drafts),
                ..Default::default()
            }
            .select(std::slice::from_mut(&mut p), &mut config())
            .unwrap();
            let selected: Vec<_> = p
                .execution
                .as_ref()
                .unwrap()
                .graphs
                .iter()
                .filter(|g| g.kind == "prefill")
                .collect();
            assert_eq!(selected.len(), 1);
            assert_eq!(selected[0].speculation_enabled, Some(drafts > 0));
        }
    }
    fn mixed_package() -> Package {
        let mut p = package();
        for enabled in [false, true] {
            for prefill in [64, 512] {
                for width in if enabled { vec![1, 3, 6] } else { vec![1] } {
                    let name = format!("mixed-4-{prefill}-{width}-{enabled}");
                    p.execution
                        .as_mut()
                        .unwrap()
                        .graphs
                        .push(wire::ExecutionGraph {
                            name: name.clone(),
                            kind: "mixed".into(),
                            sampling_mode: "general".into(),
                            batch_capacity: 4,
                            token_capacity: width,
                            prefill_capacity: prefill,
                            speculation_enabled: Some(enabled),
                            draft_tokens: enabled.then_some(width as u32 - 1),
                            verify_tokens: enabled.then_some(width as u32),
                            draft_source: if enabled { 1 } else { 0 },
                            ..Default::default()
                        });
                    p.graphs.push(Graph {
                        name,
                        resource: "mixed".into(),
                        node_count: 1,
                        edge_count: 0,
                        flags: 0,
                        functions: vec![0],
                    });
                }
            }
        }
        p
    }

    #[test]
    fn mixed_selection_preserves_prompt_variants_and_applies_decode_limits() {
        for (drafts, adaptive, widths) in [
            (0, true, vec![1]),
            (2, true, vec![1, 3]),
            (2, false, vec![1, 3]),
        ] {
            let mut p = mixed_package();
            let mut c = config();
            c["prefill_chunk_size"] = 512.into();
            LoadOptions {
                mtp_draft_tokens: Some(drafts),
                adaptive_verification: Some(adaptive),
                ..Default::default()
            }
            .select(std::slice::from_mut(&mut p), &mut c)
            .unwrap();
            let mixed: Vec<_> = p
                .execution
                .as_ref()
                .unwrap()
                .graphs
                .iter()
                .filter(|g| g.kind == "mixed")
                .collect();
            assert_eq!(mixed.len(), 2 * widths.len());
            for prefill in [64, 512] {
                for &width in &widths {
                    let graph = mixed
                        .iter()
                        .find(|g| g.prefill_capacity == prefill && g.token_capacity == width)
                        .unwrap();
                    assert_eq!(graph.speculation_enabled, Some(drafts > 0));
                    assert_eq!(graph.packed_tokens(), prefill + 3 * width);
                }
            }
        }
    }

    #[test]
    fn mixed_selection_rejects_rank_shape_mismatch_and_invalid_mode() {
        let mut ranks = [mixed_package(), mixed_package()];
        let mut c = config();
        c["prefill_chunk_size"] = 512.into();
        ranks[1]
            .execution
            .as_mut()
            .unwrap()
            .graphs
            .iter_mut()
            .find(|g| g.kind == "mixed" && g.speculation_enabled == Some(true))
            .unwrap()
            .prefill_capacity = 128;
        assert!(
            LoadOptions::default()
                .select(&mut ranks, &mut c)
                .unwrap_err()
                .to_string()
                .contains("catalogs differ")
        );
        for mutation in 0..4 {
            let mut p = mixed_package();
            let g = p
                .execution
                .as_mut()
                .unwrap()
                .graphs
                .iter_mut()
                .find(|g| g.kind == "mixed")
                .unwrap();
            match mutation {
                0 => g.speculation_enabled = None,
                1 => g.token_capacity = 3,
                2 => g.prefill_capacity = 65,
                _ => g.prefill_capacity = 1024,
            }
            let mut c = config();
            c["prefill_chunk_size"] = 512.into();
            assert!(LoadOptions::default().select(&mut [p], &mut c).is_err());
        }
    }

    fn split_package(active: usize, token_budget: usize) -> Package {
        let mut p = package();
        let e = p.execution.as_mut().unwrap();
        e.max_active = active as u64;
        e.graphs.clear();
        for batch in BATCH_CAPACITIES.into_iter().filter(|b| *b <= active.next_power_of_two()) {
            let mut add = |kind: &str, width: usize, enabled: bool| {
                e.graphs.push(wire::ExecutionGraph {
                    name: format!("{kind}-{batch}-{width}-{enabled}"), kind: kind.into(),
                    sampling_mode: "general".into(), batch_capacity: batch as u64,
                    token_capacity: width as u64, speculation_enabled: Some(enabled),
                    draft_tokens: matches!(kind, "proposal" | "speculative").then_some(width as u32 - 1),
                    verify_tokens: matches!(kind, "verification" | "speculative").then_some(width as u32),
                    draft_source: i32::from(matches!(kind, "proposal" | "speculative")),
                    ..Default::default()
                });
            };
            for enabled in [false, true] {
                for kind in ["init", "restore", "snapshot", "resume"] { add(kind, 1, enabled); }
            }
            add("decode", 1, false);
            add("speculative", 1, true);
            for depth in 1..=7 { add("proposal", depth + 1, true); }
            for width in 1..=8 {
                if width == 1 || batch * width <= token_budget { add("verification", width, true); }
            }
        }
        p.graphs = e.graphs.iter().map(|g| Graph { name: g.name.clone(), flags: 0,
            resource: g.name.clone(), node_count: 1, edge_count: 0, functions: vec![0] }).collect();
        p
    }

    #[test]
    fn split_catalog_selects_one_configured_producer_and_ragged_verification() {
        for prepared in [1usize, 3, 4, 9, 33, 64] {
            let budget = prepared.next_power_of_two().max(8);
            for active in [1, prepared] {
                for depth in 0..=7 {
                    for adaptive in [false, true] {
                        let mut p = split_package(prepared, budget);
                        let mut c = serde_json::json!({"speculative_widths": [1,2,3,4,5,6,7,8],
                            "speculative_token_capacity": budget});
                        LoadOptions { max_concurrency: Some(active), mtp_draft_tokens: Some(depth),
                            adaptive_verification: Some(adaptive), ..Default::default() }
                            .select(std::slice::from_mut(&mut p), &mut c).unwrap();
                        let graphs = &p.execution.as_ref().unwrap().graphs;
                        let proposals: Vec<_> = graphs.iter().filter(|g| g.kind == "proposal").collect();
                        assert_eq!(proposals.len(), if depth == 0 { 0 } else { active.next_power_of_two().ilog2() as usize + 1 });
                        assert!(proposals.iter().all(|g| g.draft_tokens == Some(depth as u32) && g.verify_tokens.is_none()));
                        for b in BATCH_CAPACITIES.into_iter().filter(|b| *b <= active.next_power_of_two()) {
                            for w in 1..=8 {
                                let present = graphs.iter().any(|g| g.kind == "verification" && g.batch_capacity == b as u64 && g.token_capacity == w as u64);
                                let expected = depth > 0 && w <= depth + 1 && (w == 1 || (b * w <= budget && (adaptive || w == depth + 1)));
                                assert_eq!(present, expected, "prepared {prepared}, active {active}, depth {depth}, adaptive {adaptive}, {b}x{w}");
                            }
                        }
                        assert_eq!(c["speculative_token_capacity"], budget);
                    }
                }
            }
        }
    }

    #[test]
    fn split_catalog_rejects_missing_depth_and_verification_metadata_before_loading() {
        for omitted in ["proposal-4-4-true", "verification-2-3-true", "speculative-4-1-true"] {
            let mut p = split_package(4, 8);
            p.execution.as_mut().unwrap().graphs.retain(|g| g.name != omitted);
            p.graphs.retain(|g| g.name != omitted);
            let mut c = serde_json::json!({"speculative_widths": [1,2,3,4,5,6,7,8], "speculative_token_capacity": 8});
            assert!(LoadOptions { mtp_draft_tokens: Some(3), ..Default::default() }.select(&mut [p], &mut c).is_err());
        }
        let mut p = split_package(4, 8);
        p.execution.as_mut().unwrap().graphs.iter_mut().find(|g| g.kind == "verification").unwrap().draft_tokens = Some(7);
        let mut c = serde_json::json!({"speculative_widths": [1,2,3,4,5,6,7,8], "speculative_token_capacity": 8});
        assert!(LoadOptions::default().select(&mut [p], &mut c).is_err());
    }

    #[test]
    fn packages_with_only_the_maximum_proposal_depth_remain_loadable() {
        let mut p = split_package(4, 8);
        p.execution.as_mut().unwrap().graphs.retain(|g| g.kind != "proposal" || g.token_capacity == 8);
        let names: HashSet<_> = p.execution.as_ref().unwrap().graphs.iter().map(|g| g.name.clone()).collect();
        p.graphs.retain(|g| names.contains(&g.name));
        let mut c = serde_json::json!({"speculative_widths": [1,2,3,4,5,6,7,8], "speculative_token_capacity": 8});
        LoadOptions::default().select(std::slice::from_mut(&mut p), &mut c).unwrap();
        let proposals: Vec<_> = p.execution.as_ref().unwrap().graphs.iter().filter(|g| g.kind == "proposal").collect();
        assert_eq!(proposals.len(), 3);
        assert!(proposals.iter().all(|g| g.token_capacity == 8));
    }

    fn packed_package(active: usize, budget: usize) -> Package {
        let mut p = split_package(active, budget);
        for graph in &mut p.execution.as_mut().unwrap().graphs {
            if graph.kind == "verification" {
                graph.packed_token_capacity = graph.batch_capacity * graph.token_capacity;
                graph.token_capacity = 8;
                graph.verify_tokens = Some(8);
            }
        }
        p
    }

    #[test]
    fn packed_packages_select_physical_bounds_and_preserve_ordinary_decode() {
        for prepared in [8usize, 32] {
            for budget in [32usize, 64] {
                for active in [1, prepared] {
                    for depth in [0, 7] {
                        let mut ranks = [packed_package(prepared, budget), packed_package(prepared, budget)];
                        let mut c = serde_json::json!({"speculative_widths": [1,2,3,4,5,6,7,8],
                            "speculative_token_capacity": budget});
                        LoadOptions { max_concurrency: Some(active), mtp_draft_tokens: Some(depth),
                            ..Default::default() }.select(&mut ranks, &mut c).unwrap();
                        let graphs = &ranks[0].execution.as_ref().unwrap().graphs;
                        assert!(graphs.iter().all(|g| g.batch_capacity <= active as u64));
                        let verification: Vec<_> = graphs.iter().filter(|g| g.kind == "verification").collect();
                        if depth == 0 {
                            assert!(verification.is_empty());
                            assert!(graphs.iter().any(|g| g.kind == "decode"));
                        } else {
                            assert!(!verification.is_empty());
                            assert!(verification.iter().all(|g| g.token_capacity == 8
                                && g.packed_tokens() <= budget as u64));
                            assert!(verification.iter().any(|g| g.packed_tokens() < g.batch_capacity * 8));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn packed_packages_reject_invalid_bounds_and_inconsistent_ranks_before_loading() {
        for capacity in [1, 65] {
            let mut p = packed_package(8, 64);
            p.execution.as_mut().unwrap().graphs.iter_mut().find(|g|
                g.kind == "verification" && g.batch_capacity == 8).unwrap().packed_token_capacity = capacity;
            let mut c = serde_json::json!({"speculative_widths": [1,2,3,4,5,6,7,8], "speculative_token_capacity": 64});
            assert!(LoadOptions::default().select(&mut [p], &mut c).is_err());
        }
        let mut ranks = [packed_package(8, 32), packed_package(8, 32)];
        ranks[1].execution.as_mut().unwrap().graphs.iter_mut().find(|g|
            g.kind == "verification" && g.batch_capacity == 8 && g.packed_token_capacity == 24)
            .unwrap().packed_token_capacity = 23;
        let mut c = serde_json::json!({"speculative_widths": [1,2,3,4,5,6,7,8], "speculative_token_capacity": 32});
        assert!(LoadOptions::default().select(&mut ranks, &mut c).unwrap_err().to_string().contains("catalogs differ"));
    }

    #[test]
    fn packed_packages_reject_unprepared_execution_modes_before_loading() {
        for options in [LoadOptions { adaptive_verification: Some(false), ..Default::default() },
            LoadOptions { mtp_draft_tokens: Some(3), ..Default::default() }] {
            let mut c = serde_json::json!({"speculative_widths": [1,2,3,4,5,6,7,8], "speculative_token_capacity": 32});
            assert!(options.select(&mut [packed_package(8, 32)], &mut c)
                .unwrap_err().to_string().contains("captured draft depth"));
        }
    }

    fn packed_draft_five_package() -> Package {
        let mut p = packed_package(8, 48);
        let e = p.execution.as_mut().unwrap();
        e.graphs.retain(|g| g.draft_tokens.is_none_or(|n| n <= 5)
            && (g.kind != "verification" || g.packed_token_capacity <= g.batch_capacity * 6));
        for g in &mut e.graphs {
            if g.kind == "verification" {
                g.token_capacity = 6;
                g.verify_tokens = Some(6);
            }
        }
        let names: HashSet<_> = e.graphs.iter().map(|g| g.name.clone()).collect();
        p.graphs.retain(|g| names.contains(&g.name));
        p
    }

    fn packed_draft_five_config() -> Value {
        serde_json::json!({"speculative_widths": [1,2,3,4,5,6],
            "speculative_token_capacity": 48})
    }

    #[test]
    fn verification_token_budget_selects_exact_capacity_for_every_loaded_batch() {
        for budget in [24usize, 32] {
            for active in [1usize, 3, 4, 5, 8] {
                let mut ranks = [packed_draft_five_package(), packed_draft_five_package()];
                let mut c = packed_draft_five_config();
                let mut baseline = [packed_draft_five_package()];
                LoadOptions { max_concurrency: Some(active), ..Default::default() }
                    .select(&mut baseline, &mut packed_draft_five_config()).unwrap();
                LoadOptions { max_concurrency: Some(active),
                    mtp_verification_token_budget: Some(budget), ..Default::default() }
                    .select(&mut ranks, &mut c).unwrap();
                for p in &ranks {
                    let graphs = &p.execution.as_ref().unwrap().graphs;
                    for batch in BATCH_CAPACITIES.into_iter().filter(|b| *b <= active.next_power_of_two()) {
                        let verification: Vec<_> = graphs.iter().filter(|g|
                            g.kind == "verification" && g.batch_capacity == batch as u64).collect();
                        assert_eq!(verification.len(), 1);
                        assert_eq!(verification[0].token_capacity, 6);
                        assert_eq!(verification[0].packed_token_capacity, budget.min(batch * 6) as u64);
                        assert!(graphs.iter().any(|g| g.kind == "proposal"
                            && g.batch_capacity == batch as u64 && g.draft_tokens == Some(5)));
                        assert!(graphs.iter().any(|g| g.kind == "speculative"
                            && g.batch_capacity == batch as u64 && g.token_capacity == 1));
                    }
                    let unchanged = |p: &Package| -> BTreeSet<String> {
                        p.execution.as_ref().unwrap().graphs.iter()
                            .filter(|g| g.kind != "verification").map(|g| g.name.clone()).collect()
                    };
                    assert_eq!(unchanged(p), unchanged(&baseline[0]));
                }
                assert_eq!(c["speculative_token_capacity"], budget);
                assert_eq!(c["speculative_widths"], serde_json::json!([1,2,3,4,5,6]));
            }
        }
    }

    #[test]
    fn verification_token_budget_rejects_missing_exact_capacity_on_every_batch_and_rank() {
        for budget in [24usize, 32] {
            for batch in [1usize, 2, 4, 8] {
                for rank in [0usize, 1] {
                    let mut ranks = [packed_draft_five_package(), packed_draft_five_package()];
                    let p = &mut ranks[rank];
                    p.execution.as_mut().unwrap().graphs.retain(|g| !(g.kind == "verification"
                        && g.batch_capacity == batch as u64
                        && g.packed_token_capacity == budget.min(batch * 6) as u64));
                    let names: HashSet<_> = p.execution.as_ref().unwrap().graphs.iter()
                        .map(|g| g.name.clone()).collect();
                    p.graphs.retain(|g| names.contains(&g.name));
                    let error = LoadOptions { mtp_verification_token_budget: Some(budget),
                        ..Default::default() }.select(&mut ranks, &mut packed_draft_five_config()).unwrap_err();
                    assert!(error.to_string().contains(&format!("rank {rank} lacks exact packed verification graph for batch {batch}")), "{error}");
                }
            }
        }
    }

    #[test]
    fn verification_token_budget_rejects_different_selected_rank_catalogs() {
        let mut ranks = [packed_draft_five_package(), packed_draft_five_package()];
        let g = ranks[1].execution.as_mut().unwrap().graphs.iter_mut().find(|g|
            g.kind == "verification" && g.batch_capacity == 8 && g.packed_token_capacity == 24).unwrap();
        let name = g.name.clone();
        g.name.push_str("-rank");
        ranks[1].graphs.iter_mut().find(|g| g.name == name).unwrap().name.push_str("-rank");
        let error = LoadOptions { mtp_verification_token_budget: Some(24), ..Default::default() }
            .select(&mut ranks, &mut packed_draft_five_config()).unwrap_err();
        assert!(error.to_string().contains("catalogs differ"));
    }

    #[test]
    fn verification_token_budget_requires_packed_adaptive_speculation_and_pending_rows() {
        for options in [
            LoadOptions { mtp_verification_token_budget: Some(0), ..Default::default() },
            LoadOptions { mtp_verification_token_budget: Some(7), ..Default::default() },
            LoadOptions { mtp_verification_token_budget: Some(24), mtp_draft_tokens: Some(0), ..Default::default() },
            LoadOptions { mtp_verification_token_budget: Some(24), adaptive_verification: Some(false), ..Default::default() },
        ] {
            assert!(options.select(&mut [packed_draft_five_package()], &mut packed_draft_five_config()).is_err());
        }
        let error = LoadOptions { mtp_verification_token_budget: Some(24), ..Default::default() }
            .select(&mut [split_package(8, 48)], &mut serde_json::json!({
                "speculative_widths": [1,2,3,4,5,6,7,8], "speculative_token_capacity": 48})).unwrap_err();
        assert!(error.to_string().contains("requires captured packed verification graphs"));
    }

    #[test]
    fn verification_token_budget_respects_prepared_total_capacity() {
        for budget in [None, Some(24usize), Some(32)] {
            let mut p = packed_draft_five_package();
            p.execution.as_mut().unwrap().graphs.retain(|g|
                g.kind != "verification" || g.packed_token_capacity <= 32);
            let names: HashSet<_> = p.execution.as_ref().unwrap().graphs.iter()
                .map(|g| g.name.clone()).collect();
            p.graphs.retain(|g| names.contains(&g.name));
            let mut c = packed_draft_five_config();
            c["speculative_token_capacity"] = 32.into();
            LoadOptions { mtp_verification_token_budget: budget, ..Default::default() }
                .select(std::slice::from_mut(&mut p), &mut c).unwrap();
            assert_eq!(c["speculative_token_capacity"], budget.unwrap_or(32));
            let graphs = &p.execution.as_ref().unwrap().graphs;
            if budget.is_none() {
                let capacities: BTreeSet<_> = graphs.iter().filter(|g|
                    g.kind == "verification" && g.batch_capacity == 8)
                    .map(|g| g.packed_token_capacity).collect();
                assert_eq!(capacities, BTreeSet::from([8, 16, 24, 32]));
            }
        }
    }

    #[test]
    fn maximum_verification_budget_retains_every_captured_lower_capacity() {
        for prepared_capacity in [32usize, 48] {
            for ceiling in [8usize, 24, 25, 32, 64] {
                for active in [1usize, 3, 4, 5, 8] {
                    let package = || {
                        let mut p = packed_draft_five_package();
                        p.execution.as_mut().unwrap().graphs.retain(|g|
                            g.kind != "verification" || g.packed_token_capacity <= prepared_capacity as u64);
                        let names: HashSet<_> = p.execution.as_ref().unwrap().graphs.iter()
                            .map(|g| g.name.clone()).collect();
                        p.graphs.retain(|g| names.contains(&g.name));
                        p
                    };
                    let mut baseline = package();
                    let mut ranks = [package(), package()];
                    let mut c = packed_draft_five_config();
                    c["speculative_token_capacity"] = prepared_capacity.into();
                    let mut baseline_config = c.clone();
                    LoadOptions { max_concurrency: Some(active), ..Default::default() }
                        .select(std::slice::from_mut(&mut baseline), &mut baseline_config).unwrap();
                    LoadOptions { max_concurrency: Some(active),
                        mtp_max_verification_token_budget: Some(ceiling), ..Default::default() }
                        .select(&mut ranks, &mut c).unwrap();
                    let expected: BTreeSet<_> = baseline.execution.as_ref().unwrap().graphs.iter()
                        .filter(|g| g.kind != "verification" || g.packed_token_capacity <= ceiling as u64)
                        .map(|g| g.name.clone()).collect();
                    for p in &ranks {
                        let graphs = &p.execution.as_ref().unwrap().graphs;
                        assert_eq!(graphs.iter().map(|g| g.name.clone()).collect::<BTreeSet<_>>(), expected);
                        for batch in BATCH_CAPACITIES.into_iter().filter(|b| *b <= active.next_power_of_two()) {
                            let capacities: BTreeSet<_> = graphs.iter().filter(|g|
                                g.kind == "verification" && g.batch_capacity == batch as u64)
                                .map(|g| { assert_eq!(g.token_capacity, 6); g.packed_token_capacity }).collect();
                            let limit = prepared_capacity.min(ceiling).min(batch * 6);
                            assert_eq!(capacities, (batch..=limit).step_by(batch).map(|n| n as u64).collect());
                        }
                    }
                    assert_eq!(c["speculative_token_capacity"], prepared_capacity.min(ceiling));
                    assert_eq!(c["speculative_widths"], serde_json::json!([1,2,3,4,5,6]));
                }
            }
        }
    }

    #[test]
    fn maximum_verification_budget_rejects_missing_required_capacity_on_each_batch_and_rank() {
        for batch in [1usize, 2, 4, 8] {
            for rank in [0usize, 1] {
                let mut ranks = [packed_draft_five_package(), packed_draft_five_package()];
                let p = &mut ranks[rank];
                p.execution.as_mut().unwrap().graphs.retain(|g| !(g.kind == "verification"
                    && g.batch_capacity == batch as u64 && g.packed_token_capacity == 24.min(batch * 6) as u64));
                let names: HashSet<_> = p.execution.as_ref().unwrap().graphs.iter().map(|g| g.name.clone()).collect();
                p.graphs.retain(|g| names.contains(&g.name));
                let error = LoadOptions { mtp_max_verification_token_budget: Some(24), ..Default::default() }
                    .select(&mut ranks, &mut packed_draft_five_config()).unwrap_err();
                assert!(error.to_string().contains(&format!("rank {rank} lacks required verification graph for batch {batch}")), "{error}");
            }
        }
    }

    #[test]
    fn maximum_verification_budget_requires_matching_lower_capacity_catalogs() {
        let mut ranks = [packed_draft_five_package(), packed_draft_five_package()];
        let p = &mut ranks[1];
        p.execution.as_mut().unwrap().graphs.retain(|g| !(g.kind == "verification"
            && g.batch_capacity == 4 && g.packed_token_capacity == 4));
        let names: HashSet<_> = p.execution.as_ref().unwrap().graphs.iter().map(|g| g.name.clone()).collect();
        p.graphs.retain(|g| names.contains(&g.name));
        let error = LoadOptions { mtp_max_verification_token_budget: Some(24), ..Default::default() }
            .select(&mut ranks, &mut packed_draft_five_config()).unwrap_err();
        assert!(error.to_string().contains("catalogs differ"));
    }

    #[test]
    fn maximum_verification_budget_rejects_incompatible_modes_and_insufficient_rows() {
        for options in [
            LoadOptions { mtp_max_verification_token_budget: Some(0), ..Default::default() },
            LoadOptions { mtp_max_verification_token_budget: Some(7), ..Default::default() },
            LoadOptions { mtp_max_verification_token_budget: Some(5), max_concurrency: Some(1), ..Default::default() },
            LoadOptions { mtp_max_verification_token_budget: Some(3), max_concurrency: Some(3), ..Default::default() },
            LoadOptions { mtp_max_verification_token_budget: Some(24), mtp_draft_tokens: Some(0), ..Default::default() },
            LoadOptions { mtp_max_verification_token_budget: Some(24), mtp_draft_tokens: Some(3), ..Default::default() },
            LoadOptions { mtp_max_verification_token_budget: Some(24), adaptive_verification: Some(false), ..Default::default() },
            LoadOptions { mtp_max_verification_token_budget: Some(24), mtp_verification_token_budget: Some(24), ..Default::default() },
        ] {
            assert!(options.select(&mut [packed_draft_five_package()], &mut packed_draft_five_config()).is_err());
        }
        let error = LoadOptions { mtp_max_verification_token_budget: Some(24), ..Default::default() }
            .select(&mut [split_package(8, 48)], &mut serde_json::json!({
                "speculative_widths": [1,2,3,4,5,6,7,8], "speculative_token_capacity": 48})).unwrap_err();
        assert!(error.to_string().contains("requires captured packed verification graphs"));
    }

    fn combined_package() -> Package {
        let mut p = packed_draft_five_package();
        let e = p.execution.as_mut().unwrap();
        let proposals: Vec<_> = e.graphs.iter().filter(|g| g.kind == "proposal").cloned().collect();
        let targets: Vec<_> = e.graphs.iter().filter(|g| g.kind == "verification").cloned().collect();
        e.graphs.retain(|g| !matches!(g.kind.as_str(), "proposal" | "verification"));
        for proposal in proposals {
            for target in targets.iter().filter(|g| g.batch_capacity == proposal.batch_capacity) {
                let mut graph = target.clone();
                graph.kind = "packed_speculative".into();
                graph.name = format!("round-{}-{}-{}", graph.batch_capacity,
                    proposal.draft_tokens.unwrap(), graph.packed_token_capacity);
                graph.draft_tokens = proposal.draft_tokens;
                graph.draft_source = 1;
                e.graphs.push(graph);
            }
        }
        p.graphs = e.graphs.iter().map(|g| Graph { name: g.name.clone(), flags: 0,
            resource: g.name.clone(), node_count: 1, edge_count: 0, functions: vec![0] }).collect();
        p
    }

    fn combined_only_package(active: usize, budget: usize) -> Package {
        let mut p = packed_package(active, budget);
        let execution = p.execution.as_mut().unwrap();
        execution.graphs.retain(|g| match g.kind.as_str() {
            "proposal" => false,
            "speculative" => g.token_capacity == 1,
            "verification" => g.packed_token_capacity > g.batch_capacity
                && g.packed_token_capacity <= g.batch_capacity * 6,
            _ => true,
        });
        for graph in &mut execution.graphs {
            if graph.kind == "verification" {
                graph.kind = "packed_speculative".into();
                graph.token_capacity = 6;
                graph.verify_tokens = Some(6);
                graph.draft_tokens = Some(5);
                graph.draft_source = 1;
            }
        }
        let names: HashSet<_> = execution.graphs.iter().map(|g| g.name.clone()).collect();
        p.graphs.retain(|g| names.contains(&g.name));
        p
    }

    #[test]
    fn combined_only_low_budgets_preserve_width_one_at_full_concurrency() {
        for prepared_budget in [8usize, 128] {
            for options in [LoadOptions::default(),
                LoadOptions { mtp_verification_token_budget: Some(8), ..Default::default() },
                LoadOptions { mtp_max_verification_token_budget: Some(8), ..Default::default() }] {
                let mut ranks = [combined_only_package(32, prepared_budget), combined_only_package(32, prepared_budget)];
                let mut config = packed_draft_five_config();
                config["speculative_token_capacity"] = prepared_budget.into();
                options.select(&mut ranks, &mut config).unwrap();
                let selected_budget = if options.mtp_verification_token_budget.is_some()
                    || options.mtp_max_verification_token_budget.is_some() { 8 } else { prepared_budget };
                for package in &ranks {
                    let graphs = &package.execution.as_ref().unwrap().graphs;
                    let rounds: Vec<_> = graphs.iter().filter(|g| g.kind == "packed_speculative").collect();
                    assert_eq!(rounds.len(), if options.mtp_verification_token_budget.is_some() { 3 }
                        else if selected_budget == 8 { 9 } else { 28 });
                    assert!(rounds.iter().all(|g| g.draft_tokens == Some(5) && g.token_capacity == 6
                        && g.packed_token_capacity > g.batch_capacity && g.packed_token_capacity <= selected_budget as u64));
                    assert!(!graphs.iter().any(|g| matches!(g.kind.as_str(), "proposal" | "verification")));
                    for batch in [1u64, 2, 4, 8, 16, 32] {
                        assert!(graphs.iter().any(|g| g.kind == "speculative" && g.batch_capacity == batch && g.token_capacity == 1));
                        assert_eq!(rounds.iter().any(|g| g.batch_capacity == batch), batch * 2 <= selected_budget as u64);
                    }
                }
                assert_eq!(config["speculative_token_capacity"], selected_budget);
            }
        }
    }

    #[test]
    fn combined_only_budgets_reject_missing_useful_rounds_and_width_one() {
        for rank in [0usize, 1] {
            for (kind, batch) in [("packed_speculative", 4u64), ("speculative", 32)] {
                let mut ranks = [combined_only_package(32, 8), combined_only_package(32, 8)];
                let package = &mut ranks[rank];
                let omitted = package.execution.as_ref().unwrap().graphs.iter()
                    .find(|g| g.kind == kind && g.batch_capacity == batch).unwrap().name.clone();
                package.execution.as_mut().unwrap().graphs.retain(|g| g.name != omitted);
                package.graphs.retain(|g| g.name != omitted);
                let mut config = packed_draft_five_config();
                config["speculative_token_capacity"] = 8.into();
                let error = LoadOptions::default().select(&mut ranks, &mut config).unwrap_err().to_string();
                assert!(error.contains(&format!("rank {rank}")) && error.contains(&format!("batch {batch}")), "{error}");
            }
        }
        for budget in [1usize, 7] {
            let mut config = packed_draft_five_config();
            config["speculative_token_capacity"] = 128.into();
            assert!(LoadOptions { mtp_verification_token_budget: Some(budget), ..Default::default() }
                .select(&mut [combined_only_package(32, 128)], &mut config).is_err());
        }
    }

    #[test]
    fn combined_only_partial_batch_capacity_remains_loadable() {
        let mut package = combined_only_package(32, 9);
        let mut graph = package.execution.as_ref().unwrap().graphs.iter()
            .find(|g| g.kind == "packed_speculative" && g.batch_capacity == 4).unwrap().clone();
        graph.name = "packed-partial-eight".into();
        graph.batch_capacity = 8;
        graph.packed_token_capacity = 9;
        package.graphs.push(Graph { name: graph.name.clone(), flags: 0,
            resource: graph.name.clone(), node_count: 1, edge_count: 0, functions: vec![0] });
        package.execution.as_mut().unwrap().graphs.push(graph);
        let mut config = packed_draft_five_config();
        config["speculative_token_capacity"] = 9.into();
        LoadOptions::default().select(std::slice::from_mut(&mut package), &mut config).unwrap();
        assert!(package.execution.as_ref().unwrap().graphs.iter().any(|g|
            g.kind == "packed_speculative" && g.batch_capacity == 8 && g.packed_token_capacity == 9));
    }

    #[test]
    fn exact_combined_budget_equal_to_batch_preserves_pending_only_captures() {
        for pending_only in [false, true] {
            let mut package = if pending_only { combined_package() } else { combined_only_package(8, 48) };
            LoadOptions { mtp_verification_token_budget: Some(8), ..Default::default() }
                .select(std::slice::from_mut(&mut package), &mut packed_draft_five_config()).unwrap();
            let graphs = &package.execution.as_ref().unwrap().graphs;
            assert!(graphs.iter().any(|g| g.kind == "speculative" && g.batch_capacity == 8 && g.token_capacity == 1));
            assert_eq!(graphs.iter().any(|g| g.kind == "packed_speculative"
                && g.batch_capacity == 8 && g.packed_token_capacity == 8), pending_only);
        }
    }

    #[test]
    fn packed_combined_selection_keeps_one_producer_and_exact_row_limits() {
        for budget in [24usize, 32] {
            let mut ranks = [combined_package(), combined_package()];
            let mut config = packed_draft_five_config();
            LoadOptions { mtp_verification_token_budget: Some(budget), ..Default::default() }
                .select(&mut ranks, &mut config).unwrap();
            for package in ranks {
                let graphs = &package.execution.unwrap().graphs;
                assert!(!graphs.iter().any(|g| matches!(g.kind.as_str(), "proposal" | "verification")));
                for batch in [1u64, 2, 4, 8] {
                    let rounds: Vec<_> = graphs.iter().filter(|g| g.kind == "packed_speculative"
                        && g.batch_capacity == batch).collect();
                    assert_eq!(rounds.len(), 1);
                    assert_eq!(rounds[0].draft_tokens, Some(5));
                    assert_eq!(rounds[0].token_capacity, 6);
                    assert_eq!(rounds[0].verify_tokens, Some(6));
                    assert_eq!(rounds[0].packed_token_capacity, (budget as u64).min(batch * 6));
                }
            }
        }
    }

    #[test]
    fn packed_combined_selection_rejects_unknown_missing_or_inconsistent_geometry() {
        for mutation in 0..6 {
            let mut p = combined_package();
            let graph = p.execution.as_mut().unwrap().graphs.iter_mut()
                .find(|g| g.kind == "packed_speculative").unwrap();
            match mutation {
                0 => graph.draft_tokens = None,
                1 => graph.draft_tokens = Some(0),
                2 => graph.draft_tokens = Some(7),
                3 => graph.verify_tokens = Some(2),
                4 => graph.packed_token_capacity = 0,
                _ => graph.kind = "packed_speculative_unknown".into(),
            }
            assert!(LoadOptions::default().select(&mut [p], &mut packed_draft_five_config()).is_err());
        }
        let mut ranks = [combined_package(), combined_package()];
        let e = ranks[1].execution.as_mut().unwrap();
        let name = e.graphs.iter().find(|g| g.kind == "packed_speculative" && g.draft_tokens == Some(5)).unwrap().name.clone();
        e.graphs.retain(|g| g.name != name);
        ranks[1].graphs.retain(|g| g.name != name);
        assert!(LoadOptions::default().select(&mut ranks, &mut packed_draft_five_config()).is_err());
    }

    #[test]
    fn packed_combined_short_depth_keeps_its_declared_wide_target_catalog() {
        let mut p = combined_package();
        let e = p.execution.as_mut().unwrap();
        e.graphs.retain(|g| g.kind != "packed_speculative" || g.draft_tokens == Some(1));
        let names: HashSet<_> = e.graphs.iter().map(|g| g.name.clone()).collect();
        p.graphs.retain(|g| names.contains(&g.name));
        let mut ranks = [p];
        let mut c = packed_draft_five_config();
        LoadOptions::default().select(&mut ranks, &mut c).unwrap();
        assert_eq!(c["speculative_widths"], serde_json::json!([1, 2, 3, 4, 5, 6]));
        let graphs = &ranks[0].execution.as_ref().unwrap().graphs;
        assert!(graphs.iter().any(|g| g.kind == "packed_speculative"));
        assert!(graphs.iter().filter(|g| g.kind == "packed_speculative")
            .all(|g| g.draft_tokens == Some(1) && g.token_capacity == 6 && g.verify_tokens == Some(6)));
    }

    #[test]
    fn packed_catalog_selection_retains_joined_and_split_operations() {
        for drafts in [None, Some(2)] {
            let mut package = combined_package();
            let split = packed_draft_five_package();
            let execution = package.execution.as_mut().unwrap();
            for graph in split.execution.unwrap().graphs.into_iter().filter(|g|
                matches!(g.kind.as_str(), "proposal" | "verification")) {
                execution.graphs.push(graph);
            }
            package.graphs = execution.graphs.iter().map(|g| Graph { name: g.name.clone(), flags: 0,
                resource: g.name.clone(), node_count: 1, edge_count: 0, functions: vec![0] }).collect();
            let mut ranks = [package];
            LoadOptions { mtp_draft_tokens: drafts, mtp_max_verification_token_budget: Some(24), ..Default::default() }
                .select(&mut ranks, &mut packed_draft_five_config()).unwrap();
            let graphs = &ranks[0].execution.as_ref().unwrap().graphs;
            assert!(graphs.iter().filter(|g| g.kind == "speculative")
                .all(|g| g.token_capacity == 1 || g.packed_tokens() <= 24));
            for kind in ["speculative", "proposal", "verification", "packed_speculative"] {
                assert!(graphs.iter().any(|g| g.kind == kind), "missing {kind}");
            }
            assert!(graphs.iter().filter(|g| g.kind == "packed_speculative")
                .all(|g| g.draft_tokens.unwrap() as usize <= drafts.unwrap_or(5)));
        }
    }

}
