//! A verified package's signed metadata as one JSON document: the facts a reader of the model
//! card, an operator sizing a host, or a client tuning requests may want, read from the index
//! alone. Nothing here authorizes the package, touches a GPU, or reads stored weights.

use serde_json::{Value, json};

use super::{
    VerifiedPackage,
    format::invalid,
    protobuf::wire::{self, Memory},
};
use crate::Result;

/// Detail sections included in package inspection output.
#[derive(Clone, Copy, Debug, Default)]
pub struct InspectionOptions {
    pub memory_map: bool,
    pub graphs: bool,
    pub frontend: bool,
    pub all: bool,
}

impl VerifiedPackage {
    /// Describe the package from its verified index. Enumerations appear by name; byte counts
    /// are exact; every rank bundle the copy holds is listed in index order. The document does
    /// not name the file, which the caller knows by the path it opened.
    pub fn describe(&self, options: InspectionOptions) -> Result<Value> {
        let index = &self.index;
        let publisher = self.publisher();
        let primary_bytes = self
            .archive
            .file
            .metadata()
            .map(|m| m.len())
            .map_err(|e| invalid(format!("cannot stat the package: {e}")))?;
        let companions: Vec<Value> = self
            .archive
            .files
            .iter()
            .filter(|(name, _)| !name.is_empty())
            .map(|(name, file)| json!({"name": name, "bytes": file.length}))
            .collect();
        let mut value = describe_index(index, options, self.verified_rank())?;
        value["package"] = json!({
            "signatures_verified": true,
            "format_version": index.version,
            "bytes": primary_bytes,
            "companions": companions,
            "ranks": index.ranks.len(),
            "cluster": index.cluster.as_ref().map(|c| json!({
                "world": c.world, "export_id": c.export_id,
            })),
            "publisher": {
                "name": publisher.name,
                "namespace": publisher.namespace,
                "fingerprint": publisher.fingerprint(),
                "development": publisher.development,
            },
            "official_source": self.official_source(),
            "license": self.license().map(|l| json!({"name": l.name, "link": l.link})),
        });
        Ok(value)
    }
}

pub(super) fn describe_index(
    index: &wire::ModelIndex,
    options: InspectionOptions,
    rank: Option<usize>,
) -> Result<Value> {
    let contract = index
        .contract
        .as_ref()
        .ok_or_else(|| invalid("model package lacks a typed serving contract"))?;
    let model = contract
        .model_contract
        .as_ref()
        .ok_or_else(|| invalid("model package lacks a model contract"))?;
    let frontend = index
        .frontend
        .as_ref()
        .ok_or_else(|| invalid("model package lacks frontend settings"))?;
    let sampling = index
        .sampling_defaults
        .as_ref()
        .ok_or_else(|| invalid("model package lacks sampling defaults"))?;
    let mut value = json!({
        // Preparation metadata is optional and never supplies serving policy. Packages
        // without it, including version 4 exports, retain their complete serving contract.
        "rebuild": index.rebuild,
        "model": {
            "model_id": contract.model_id,
            "identity": contract.identity,
            "served_names": contract.served_names,
            "vocabulary": model.vocabulary,
            "input_vocabulary": model.input_vocabulary,
            "max_sequence_tokens": contract.max_sequence_tokens,
            "prefill_chunk_size": contract.prefill_chunk_size,
            "max_active": contract.max_active.or_else(|| {
                index.ranks.first().and_then(|r| r.execution.as_ref()).map(|e| e.max_active)
            }),
            "max_new_tokens": contract.max_new_tokens,
            "decode_weight_bytes": contract.decode_weight_bytes,
            "reserve_bytes": contract.reserve_bytes,
            "queue_capacity": contract.queue_capacity,
            "frontend_workers": contract.frontend_workers,
            "max_body_bytes": contract.max_body_bytes,
            "context_hint": contract.context_hint,
            "capacity_hint": contract.capacity_hint,
        },
        "cache": {
            "page_tokens": contract.page_tokens,
            "pages": contract.pages,
            "state_rows": contract.state_rows,
            "snapshots": contract.snapshots,
            "regions": model.cache.as_ref().map(|cache| cache.regions.iter().map(|r| json!({
                "kind": wire::HistoryKind::try_from(r.kind).map_or("UNKNOWN", |k| k.as_str_name()),
                "tokens_per_record": r.tokens_per_record,
                "window_tokens": r.window_tokens,
                "stride_bytes": r.storage.as_ref().map_or(0, |s| s.stride),
                "nbytes": r.storage.as_ref().map_or(0, |s| s.nbytes),
            })).collect::<Vec<_>>()).unwrap_or_default(),
        },
        "speculation": {
            "proposal_source": contract.speculative_proposal_source,
            "widths": contract.speculative_widths,
            "fixed_width": contract.speculative_fixed_width,
            "token_capacity": contract.speculative_token_capacity,
            "history_drafts": contract.history_drafts,
        },
        "sampling_defaults": {
            "seed": sampling.seed,
            "temperature": sampling.temperature,
            "top_k": sampling.top_k,
            "top_p": sampling.top_p,
            "min_p": sampling.min_p,
            "repetition_penalty": sampling.repetition_penalty,
            "presence_penalty": sampling.presence_penalty,
            "frequency_penalty": sampling.frequency_penalty,
            "ranges": sampling.ranges,
        },
        "frontend": frontend_json(frontend, model.frontend.as_ref()),
        "vision": vision_json(
            model.vision.as_ref(), frontend,
            (contract.max_sequence_tokens > 0).then_some(contract.max_sequence_tokens),
        ),
        "assets": index.assets.iter().map(|a| json!({
            "name": a.name,
            "bytes": a.payload.as_ref().map_or(0, |p| p.nbytes),
            "sha256": a.sha256,
        })).collect::<Vec<_>>(),
        "engram": index.engram.as_ref().map(engram_json),
        "ranks": index.ranks.iter().enumerate().map(|(i, bundle)| rank_json(rank.unwrap_or(i), bundle, options)).collect::<Result<Vec<_>>>()?,
        "graph_contracts": model.graphs.iter().map(|g| json!({
            "graph": g.graph,
            "kind": wire::GraphOperation::try_from(g.kind).map_or("UNKNOWN", |k| k.as_str_name()),
            "state_rows_per_request": g.state_rows_per_request,
            "embedding_tables": g.embedding_tables,
        })).collect::<Vec<_>>(),
    });
    let mut kv_page_bytes = 0u64;
    let mut state_row_bytes = 0u64;
    let mut max_pages = None::<u64>;
    for bundle in &index.ranks {
        for allocation in &bundle.allocations {
            if let Some(s) = &allocation.sizing {
                match wire::Capacity::try_from(s.capacity) {
                    Ok(wire::Capacity::KvPages) => {
                        kv_page_bytes = kv_page_bytes
                            .checked_add(s.unit_bytes)
                            .ok_or_else(|| invalid("KV page size overflow"))?;
                        max_pages = Some(max_pages.map_or(s.maximum, |n| n.min(s.maximum)));
                    }
                    Ok(wire::Capacity::StateRows) => {
                        state_row_bytes = state_row_bytes
                            .checked_add(s.unit_bytes)
                            .ok_or_else(|| invalid("state row size overflow"))?;
                    }
                    _ => {}
                }
            }
        }
    }
    value["cache"]["kv_page_bytes"] = kv_page_bytes.into();
    value["cache"]["state_row_bytes"] = state_row_bytes.into();
    value["cache"]["max_pages"] = max_pages.unwrap_or(contract.pages).into();
    value["cache"]["history_kinds"] = json!(
        model
            .cache
            .as_ref()
            .map(|cache| cache
                .regions
                .iter()
                .map(|r| wire::HistoryKind::try_from(r.kind).map_or("UNKNOWN", |k| k.as_str_name()))
                .collect::<std::collections::BTreeSet<_>>())
            .unwrap_or_default()
    );
    if !options.frontend && !options.all {
        for key in ["frontend", "vision", "engram", "assets"] {
            value.as_object_mut().unwrap().remove(key);
        }
    }
    if options.all {
        value["contract"] = json!(contract);
    } else {
        value.as_object_mut().unwrap().remove("graph_contracts");
        value["cache"].as_object_mut().unwrap().remove("regions");
    }
    Ok(value)
}

fn frontend_json(
    settings: &wire::FrontendSettings,
    contract: Option<&wire::FrontendContract>,
) -> Value {
    let tokenizer = settings.tokenizer.as_ref();
    let special = |token: Option<&String>| token.cloned();
    json!({
        "prompt_encoder": wire::PromptEncoder::try_from(settings.prompt_encoder)
            .map_or("UNKNOWN", |k| k.as_str_name()),
        "template_encoding": wire::TemplateEncoding::try_from(settings.template_encoding)
            .map_or("UNKNOWN", |k| k.as_str_name()),
        "eos_token_ids": settings.eos_token_ids,
        "image_placeholder_token_id": settings.image_placeholder_token_id,
        "tokenizer": {
            "bos_token": special(tokenizer.and_then(|t| t.bos_token.as_ref())),
            "eos_token": special(tokenizer.and_then(|t| t.eos_token.as_ref())),
            "pad_token": special(tokenizer.and_then(|t| t.pad_token.as_ref())),
            "unk_token": special(tokenizer.and_then(|t| t.unk_token.as_ref())),
            "additional_special_tokens": tokenizer
                .and_then(|t| t.additional_special_tokens.as_ref())
                .map_or(0, |list| list.values.len()),
        },
        "tools": contract.and_then(|c| c.tools.as_ref()).map(|t| json!({
            "kind": wire::ToolParserKind::try_from(t.kind).map_or("UNKNOWN", |k| k.as_str_name()),
            "wrapping_newlines": t.wrapping_newlines,
            "tag_separator": t.tag_separator,
        })),
        "reasoning": contract.and_then(|c| c.reasoning.as_ref()).map(|r| json!({
            "kind": wire::ReasoningParserKind::try_from(r.kind).map_or("UNKNOWN", |k| k.as_str_name()),
            "open": r.open,
            "close": r.close,
            "wrapping_newlines": r.wrapping_newlines,
            "terminal_markers": r.terminal_markers,
        })),
        "assistant_marker": contract.map(|c| c.assistant_marker.clone()),
        "message_end_markers": contract.map(|c| c.message_end_markers.clone()),
        "rejected_tokens": contract.map_or(0, |c| c.rejected_tokens.len()),
        "position_dimensions": contract.map_or(0, |c| c.position_dimensions),
        "prefix_boundary": contract.and_then(|c| c.prefix_boundary.as_ref()).map(|b| json!({
            "roles": b.roles,
            "append_role": b.append_role,
            "continue_final_message": b.continue_final_message,
            "minimum_messages": b.minimum_messages,
        })),
    })
}

fn vision_json(
    vision: Option<&wire::VisionCapacity>,
    settings: &wire::FrontendSettings,
    context_tokens: Option<u64>,
) -> Value {
    let Some(vision) = vision.filter(|v| v.max_images > 0) else {
        return Value::Null;
    };
    let profile = settings.image.as_ref().and_then(|image| image.profile.as_ref()).map(|profile| {
        use wire::image_settings::Profile;
        match profile {
            Profile::Qwen(c) => json!({"profile": "qwen", "min_pixels": c.min_pixels, "max_pixels": c.max_pixels,
                "position_side": c.position_side, "image_token_id": c.image_token_id, "video_token_id": c.video_token_id}),
            Profile::Mimo(c) => json!({"profile": "mimo", "min_pixels": c.min_pixels, "max_pixels": c.max_pixels}),
            Profile::DeepseekRowMajor(c) => json!({"profile": "deepseek_row_major", "min_pixels": c.min_pixels,
                "max_tokens": c.max_tokens, "max_width_height_ratio": c.max_width_height_ratio}),
            Profile::Deepseek4(_) => json!({"profile": "deepseek4"}),
            Profile::Moonvit(_) => json!({"profile": "moonvit"}),
            Profile::Step(_) => json!({"profile": "step"}),
            Profile::Radio(_) => json!({"profile": "radio"}),
            Profile::PatchMerge(c) => json!({"profile": "patch_merge", "patch_size": c.patch_size,
                "min_pixels": c.min_pixels, "max_pixels": c.max_pixels}),
            Profile::Glm(_) => json!({"profile": "glm"}),
        }
    });
    json!({
        "feature_slots": vision.feature_slots,
        "max_image_tokens_per_prompt": context_tokens,
        "max_images": context_tokens,
        "image_feature_window_tokens": vision.features_per_request,
        "patches_per_feature": vision.patches_per_feature,
        "patch_capacities": vision.patch_capacities,
        "image": profile,
    })
}

fn engram_json(engram: &wire::Engram) -> Value {
    json!({
        "compressed_vocabulary": engram.compressed_vocabulary,
        "pad_token_id": engram.pad_token_id,
        "token_map_entries": engram.token_map.len(),
        "tables": engram.tables.iter().map(|t| json!({
            "layer": t.layer,
            "rows": t.rows,
            "embedding_width": t.embedding_width,
            "encoding": wire::engram_table::Encoding::try_from(t.encoding)
                .map_or("UNKNOWN", |e| e.as_str_name()),
            "weight_bytes": t.weights.as_ref().map_or(0, |s| s.nbytes)
                + t.shards.iter().filter_map(|s| s.weights.as_ref()).map(|s| s.nbytes).sum::<u64>(),
            "scale_bytes": t.scales.as_ref().map_or(0, |s| s.nbytes)
                + t.shards.iter().filter_map(|s| s.scales.as_ref()).map(|s| s.nbytes).sum::<u64>(),
            "shards": t.shards.len(),
        })).collect::<Vec<_>>(),
    })
}

fn rank_json(rank: usize, bundle: &wire::BundleIndex, options: InspectionOptions) -> Result<Value> {
    let bytes = |memory: Memory| -> u64 {
        bundle
            .allocations
            .iter()
            .filter(|a| Memory::try_from(a.memory) == Ok(memory))
            .map(|a| a.nbytes)
            .sum()
    };
    let sized: Vec<Value> = bundle
        .allocations
        .iter()
        .filter_map(|a| {
            let sizing = a.sizing.as_ref()?;
            Some(json!({
                "name": a.name,
                "capacity": wire::Capacity::try_from(sizing.capacity).map_or("UNKNOWN", |c| c.as_str_name()),
                "unit_bytes": sizing.unit_bytes,
                "maximum": sizing.maximum,
                "memory": Memory::try_from(a.memory).map_or("UNKNOWN", |m| m.as_str_name()),
            }))
        })
        .collect();
    let execution = bundle.execution.as_ref();
    let mut value = json!({
        "rank": rank,
        "architecture": bundle.architecture,
        "device_bytes": bytes(Memory::Device),
        "mapped_host_bytes": bytes(Memory::MappedHost),
        "peer_bytes": bytes(Memory::Peer),
        "allocations": bundle.allocations.len(),
        "capacity_allocations": sized,
        "modules": bundle.modules.len(),
        "functions": bundle.functions.len(),
        "graphs": bundle.graphs.len(),
        "events": bundle.events.len(),
        "abi_version": execution.map_or(0, |e| e.abi_version),
        "max_active": execution.map_or(0, |e| e.max_active),
        "execution_graphs": execution.map(|e| e.graphs.iter().map(|g| json!({
            "name": g.name,
            "kind": g.kind,
            "sampling_mode": g.sampling_mode,
            "batch_capacity": g.batch_capacity,
            "token_capacity": g.token_capacity,
            "packed_tokens": g.packed_tokens(),
            "prefill_capacity": g.prefill_capacity,
            "draft_tokens": g.draft_tokens,
            "verify_tokens": g.verify_tokens,
            "draft_source": wire::DraftSource::try_from(g.draft_source).map_or("UNKNOWN", |d| d.as_str_name()),
            "speculation_enabled": g.speculation_enabled,
            "estimated_duration_ns": g.estimated_duration_ns,
            "node_count": bundle.graphs.iter().find(|graph| graph.name == g.name).map(|graph| graph.node_count),
            "edge_count": bundle.graphs.iter().find(|graph| graph.name == g.name).map(|graph| graph.edge_count),
        })).collect::<Vec<_>>()).unwrap_or_default(),
        "roce": bundle.roce.as_ref().map(|r| json!({
            "world": r.world, "rank": r.rank, "proxy_abi": r.proxy_abi,
            "region_bytes": r.region_bytes, "varlen_bytes": r.varlen_bytes,
        })),
    });
    if !options.graphs && !options.all {
        value.as_object_mut().unwrap().remove("execution_graphs");
    }
    value["rebuild_weights"] = match &bundle.rebuild_weights {
        None => Value::Null,
        Some(catalog) if options.all => json!(catalog),
        Some(catalog) => json!({"version": catalog.version, "tensors": catalog.tensors.len(),
            "unavailable": catalog.unavailable.len()}),
    };
    if options.memory_map || options.all {
        value["memory_map"] = json!({
            "allocations": bundle.allocations.iter().map(|a| allocation_json(a, bundle)).collect::<Result<Vec<_>>>()?,
            "reservations": bundle.reservations,
            "buffers": execution.map(|e| &e.buffers),
        });
    }
    if !options.all {
        value
            .as_object_mut()
            .unwrap()
            .remove("capacity_allocations");
    }
    Ok(value)
}

fn allocation_json(a: &wire::Allocation, bundle: &wire::BundleIndex) -> Result<Value> {
    let memory =
        wire::Memory::try_from(a.memory).map_err(|_| invalid("unknown allocation memory kind"))?;
    let virtual_bytes = match &a.sizing {
        Some(s) => {
            if !a.alignment.is_power_of_two() {
                return Err(invalid("invalid allocation alignment"));
            }
            s.unit_bytes
                .checked_mul(s.maximum)
                .and_then(|n| n.checked_add(a.alignment - 1))
                .map(|n| n & !(a.alignment - 1))
                .ok_or_else(|| invalid("allocation extent overflow"))?
        }
        None => a.nbytes,
    };
    let stored_bytes = a.data_spans.iter().try_fold(0u64, |n, s| {
        n.checked_add(s.nbytes)
            .ok_or_else(|| invalid("stored allocation size overflow"))
    })?;
    let sizing = a
        .sizing
        .as_ref()
        .map(|s| -> Result<Value> {
            let capacity = wire::Capacity::try_from(s.capacity)
                .map_err(|_| invalid("unknown allocation capacity"))?;
            Ok(
                json!({"capacity": capacity.as_str_name(), "unit_bytes": s.unit_bytes,
            "maximum": s.maximum}),
            )
        })
        .transpose()?;
    Ok(json!({
        "name": a.name, "memory": memory.as_str_name(), "address": a.address,
        "nbytes": a.nbytes, "virtual_bytes": virtual_bytes, "alignment": a.alignment,
        "stored_bytes": stored_bytes, "sizing": sizing, "data_spans": a.data_spans,
        "payload": a.file.as_ref().and_then(|name| bundle.resources.get(name)),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn serialized_packages_without_rebuild_metadata_keep_serving_descriptions() {
        // Version 4, one empty rank, empty sampling/frontend messages, and a typed
        // serving contract with an empty model contract; no rebuild fields on wire.
        let bytes = [8, 4, 18, 0, 58, 0, 66, 0, 82, 2, 26, 0];
        let mut index = wire::ModelIndex::decode(bytes.as_slice()).unwrap();
        let before = describe_index(&index, InspectionOptions::default(), None).unwrap();
        assert!(before["rebuild"].is_null());
        assert!(before["ranks"][0]["rebuild_weights"].is_null());
        // Serving metadata remains usable even if preparation requires an importer
        // for a catalog version this runtime does not interpret.
        index.rebuild = Some(wire::RebuildManifest {
            version: 99,
            ..Default::default()
        });
        index.ranks[0].rebuild_weights = Some(wire::RebuildWeights {
            version: 99,
            ..Default::default()
        });
        let after = describe_index(&index, InspectionOptions::default(), None).unwrap();
        assert_eq!(before["model"], after["model"]);
        assert_eq!(before["cache"], after["cache"]);
        assert_eq!(after["rebuild"]["version"], 99);
        assert_eq!(after["ranks"][0]["rebuild_weights"]["version"], 99);
    }

    #[test]
    fn rebuild_metadata_without_a_chain_field_describes_its_recorded_shape() {
        // Field 22 (`roce_chain`) is absent from rebuild metadata serialized without it; such a
        // ring package keeps its ring shape and reads as not a chain.
        let ring = wire::RebuildManifest {
            version: 1,
            roce_ring: true,
            ..Default::default()
        }
        .encode_to_vec();
        let decoded = wire::RebuildManifest::decode(ring.as_slice()).unwrap();
        assert!(decoded.roce_ring && !decoded.roce_chain);
        // The same minimal serving index as the test above, with the ring's rebuild metadata.
        let bytes = [8, 4, 18, 0, 58, 0, 66, 0, 82, 2, 26, 0];
        let mut index = wire::ModelIndex::decode(bytes.as_slice()).unwrap();
        index.rebuild = Some(decoded);
        let described = describe_index(&index, InspectionOptions::default(), None).unwrap();
        assert_eq!(described["rebuild"]["roce_ring"], true);
        assert_eq!(described["rebuild"]["roce_chain"], false);
    }

    #[test]
    fn summary_and_detail_sections_preserve_optional_values_and_selected_ranks() {
        let index = wire::ModelIndex {
            contract: Some(wire::ServingContract {
                model_contract: Some(wire::ModelContract::default()),
                max_new_tokens: Some(0),
                ..Default::default()
            }),
            sampling_defaults: Some(wire::DefaultSamplingParameters {
                seed: Some(u64::MAX),
                temperature: Some(1.),
                ranges: Some(wire::SamplingRanges {
                    presence_penalty: Some(wire::SamplingFloatRange {
                        min: Some(0.),
                        max: Some(2.),
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            frontend: Some(wire::FrontendSettings::default()),
            ranks: vec![wire::BundleIndex {
                execution: Some(wire::Execution {
                    max_active: 4,
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let summary = describe_index(&index, InspectionOptions::default(), Some(3)).unwrap();
        assert_eq!(summary["model"]["max_active"], 4);
        assert_eq!(summary["model"]["max_new_tokens"], 0);
        assert_eq!(summary["sampling_defaults"]["seed"], u64::MAX);
        assert_eq!(
            summary["sampling_defaults"]["ranges"]["presence_penalty"]["min"],
            0.
        );
        assert_eq!(summary["ranks"][0]["rank"], 3);
        assert!(summary["sampling_defaults"]["top_p"].is_null());
        for key in [
            "contract",
            "frontend",
            "assets",
            "engram",
            "graph_contracts",
        ] {
            assert!(summary.get(key).is_none(), "{key}");
        }
        for key in ["execution_graphs", "memory_map", "capacity_allocations"] {
            assert!(summary["ranks"][0].get(key).is_none(), "{key}");
        }
        for options in [
            InspectionOptions {
                graphs: true,
                ..Default::default()
            },
            InspectionOptions {
                memory_map: true,
                ..Default::default()
            },
            InspectionOptions {
                frontend: true,
                ..Default::default()
            },
            InspectionOptions {
                all: true,
                ..Default::default()
            },
        ] {
            let detail = describe_index(&index, options, Some(3)).unwrap();
            assert_eq!(
                detail.get("frontend").is_some(),
                options.frontend || options.all
            );
            assert_eq!(detail.get("contract").is_some(), options.all);
            assert_eq!(
                detail["ranks"][0].get("execution_graphs").is_some(),
                options.graphs || options.all
            );
            assert_eq!(
                detail["ranks"][0].get("memory_map").is_some(),
                options.memory_map || options.all
            );
        }
    }

    #[test]
    fn rank_bytes_split_by_memory_kind_and_graphs_report_packed_tokens() {
        let allocation = |name: &str, nbytes: u64, memory: Memory, sizing: Option<wire::Sizing>| {
            wire::Allocation {
                name: name.into(),
                nbytes,
                alignment: 256,
                file: None,
                data_spans: vec![],
                relocations: vec![],
                address: None,
                memory: memory as i32,
                sizing,
            }
        };
        let bundle = wire::BundleIndex {
            architecture: vec![10, 3],
            allocations: vec![
                allocation("weights", 1 << 30, Memory::Device, None),
                allocation(
                    "kv",
                    1 << 20,
                    Memory::Device,
                    Some(wire::Sizing {
                        capacity: wire::Capacity::KvPages as i32,
                        unit_bytes: 4096,
                        maximum: 1000,
                    }),
                ),
                allocation("host", 1 << 10, Memory::MappedHost, None),
                allocation("peer", 1 << 8, Memory::Peer, None),
            ],
            modules: vec![],
            functions: vec![],
            graphs: vec![],
            reservations: vec![],
            execution: Some(wire::Execution {
                abi_version: 9,
                max_active: 32,
                input_frame_bytes: 0,
                buffers: vec![],
                graphs: vec![wire::ExecutionGraph {
                    name: "decode_b32".into(),
                    kind: "decode".into(),
                    sampling_mode: "sample".into(),
                    batch_capacity: 32,
                    token_capacity: 1,
                    input_bindings: vec![],
                    draft_tokens: None,
                    verify_tokens: None,
                    draft_source: wire::DraftSource::NoDraft as i32,
                    speculation_enabled: Some(false),
                    estimated_duration_ns: 0,
                    prefill_capacity: 0,
                    packed_token_capacity: 0,
                }],
            }),
            context: None,
            resources: Default::default(),
            events: vec![],
            roce: None,
            rebuild_weights: None,
        };
        let value = rank_json(
            2,
            &bundle,
            InspectionOptions {
                all: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(value["rank"], 2);
        assert_eq!(value["architecture"], json!([10, 3]));
        assert_eq!(value["device_bytes"], (1u64 << 30) + (1 << 20));
        assert_eq!(value["mapped_host_bytes"], 1 << 10);
        assert_eq!(value["peer_bytes"], 1 << 8);
        assert_eq!(value["capacity_allocations"][0]["capacity"], "KV_PAGES");
        assert_eq!(value["capacity_allocations"][0]["unit_bytes"], 4096);
        assert_eq!(value["execution_graphs"][0]["packed_tokens"], 32);
        assert_eq!(value["execution_graphs"][0]["draft_source"], "NO_DRAFT");
        assert_eq!(value["abi_version"], 9);
    }

    #[test]
    fn vision_is_null_without_images_and_names_the_profile_with_them() {
        let settings = wire::FrontendSettings {
            tokenizer: None,
            image: Some(wire::ImageSettings {
                profile: Some(wire::image_settings::Profile::Deepseek4(wire::Empty {})),
            }),
            eos_token_ids: vec![1],
            template_encoding: 0,
            prompt_encoder: wire::PromptEncoder::DeepseekV41 as i32,
            image_placeholder_token_id: Some(7),
        };
        let disabled = wire::VisionCapacity {
            feature_slots: 0,
            features_per_request: 0,
            max_images: 0,
            patches_per_feature: 0,
            patch_capacities: vec![],
        };
        assert_eq!(vision_json(Some(&disabled), &settings, Some(131072)), Value::Null);
        let enabled = wire::VisionCapacity {
            feature_slots: 4,
            features_per_request: 4096,
            max_images: 4096,
            patches_per_feature: 9,
            patch_capacities: vec![3456],
        };
        let value = vision_json(Some(&enabled), &settings, Some(131072));
        assert_eq!(value["max_image_tokens_per_prompt"], 131072);
        assert_eq!(value["max_images"], 131072);
        assert_eq!(value["image_feature_window_tokens"], 4096);
        let absent_context = vision_json(Some(&enabled), &settings, None);
        assert!(absent_context["max_image_tokens_per_prompt"].is_null());
        assert!(absent_context["max_images"].is_null());
        assert_eq!(absent_context["image_feature_window_tokens"], 4096);
        let index = wire::ModelIndex {
            contract: Some(wire::ServingContract {
                model_contract: Some(wire::ModelContract { vision: Some(enabled), ..Default::default() }),
                max_sequence_tokens: 131072, ..Default::default()
            }),
            frontend: Some(settings.clone()), sampling_defaults: Some(Default::default()),
            ..Default::default()
        };
        let inspected = describe_index(&index, InspectionOptions { frontend: true,
            ..Default::default() }, None).unwrap();
        assert_eq!(inspected["vision"]["max_image_tokens_per_prompt"], 131072);
        assert_eq!(inspected["vision"]["image_feature_window_tokens"], 4096);
        assert_eq!(value["image"]["profile"], "deepseek4");
        let frontend = frontend_json(&settings, None);
        assert_eq!(frontend["prompt_encoder"], "DEEPSEEK_V41");
        assert_eq!(frontend["image_placeholder_token_id"], 7);
        assert!(frontend["tools"].is_null());
    }
}
