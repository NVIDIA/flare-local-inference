//! CPython handoff for retained preparation resources.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use super::{CaptureRequest, recorder::NativeModuleRecorder};
use crate::lifetime::OwnerLease;

#[pyfunction]
#[pyo3(signature=(request, owners, recorder=None))]
fn export_capture(
    py: Python<'_>,
    request: &str,
    owners: Py<PyAny>,
    recorder: Option<PyRef<'_, NativeModuleRecorder>>,
) -> PyResult<NativeRankOutput> {
    if owners.bind(py).is_none() {
        return Err(PyValueError::new_err(
            "export owners must retain graph templates and allocation storage",
        ));
    }
    let request: CaptureRequest = serde_json::from_str(request)
        .map_err(|e| PyValueError::new_err(format!("invalid export metadata: {e}")))?;
    py.detach(|| validate_metadata(&request))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut lease = OwnerLease(Some(owners));
    let recorder_lock = recorder.as_ref().map(|r| &r.recorder);
    let outcome = py.detach(|| {
        let result = match recorder_lock {
            Some(recorder) => {
                let recorder = recorder
                    .lock()
                    .map_err(|_| super::invalid("module recorder lock poisoned"))?;
                super::capture::capture_rank(&request, Some(&*recorder))
            }
            None => super::capture::capture_rank(&request, None),
        };
        let result = result.and_then(|mut output| {
            let publication = (|| -> super::Result<()> {
                if request.sources {
                    if request.publisher.is_some() {
                        return Err(super::invalid(
                            "a signed package cannot reference staged sources",
                        ));
                    }
                    super::package::write_sources(&request.path, &output)?;
                } else {
                    let publisher = request
                        .publisher
                        .clone()
                        .map(serde_json::from_value::<super::protection::PublisherRequest>)
                        .transpose()?
                        .map(|p| p.publisher())
                        .transpose()?;
                    super::package::publish(
                        &request.path,
                        output.index.clone(),
                        output.payload.clone(),
                        publisher.as_ref(),
                        &request.options,
                        request.shard_size,
                    )?;
                }
                Ok(())
            })();
            if !request.sources || publication.is_err() {
                if let Some(staging) = output.staging.take() {
                    let _ = std::fs::remove_dir_all(staging);
                }
            }
            publication?;
            Ok(output)
        });
        let drain = super::capture::drain_context(request.context);
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>((result, drain))
    });
    let (result, drain) = outcome.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if drain.is_ok() {
        drop(lease.take_after_drain());
    }
    if let Err(error) = drain {
        let primary = result
            .as_ref()
            .err()
            .map(|e| format!("{e}; "))
            .unwrap_or_default();
        return Err(PyRuntimeError::new_err(format!(
            "{primary}export CUDA drain failed; preparation owners retained: {error}"
        )));
    }
    let result = result.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(NativeRankOutput { output: result })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(export_capture, module)?)?;
    module.add_function(wrap_pyfunction!(export_synchronize, module)?)?;
    module.add_function(wrap_pyfunction!(export_reset_peer_storage, module)?)?;
    module.add_function(wrap_pyfunction!(export_serving, module)?)?;
    module.add_class::<NativeExportStage>()?;
    module.add_class::<NativeRankOutput>()?;
    super::recorder::register(module)?;
    super::package::register(module)?;
    super::conversion::register(module)?;
    #[cfg(unix)]
    super::transfer::register(module)?;
    module.add_function(wrap_pyfunction!(export_source_identity, module)?)?;
    Ok(())
}

pub(crate) fn message<T: prost::Message + Default>(
    name: &str,
    value: &serde_json::Value,
) -> super::Result<T> {
    let pool =
        prost_reflect::DescriptorPool::decode(helios_core::runtime_package::WIRE_DESCRIPTOR)?;
    let descriptor = pool
        .get_message_by_name(&format!("flare.runtime.{name}"))
        .ok_or_else(|| super::invalid(format!("unknown package message {name}")))?;
    let dynamic = prost_reflect::DynamicMessage::deserialize(descriptor, value.clone())?;
    Ok(dynamic.transcode_to()?)
}

pub(crate) fn execution(
    value: &serde_json::Value,
    allocations: &[super::Allocation],
) -> super::Result<super::wire::Execution> {
    let mut value = value.clone();
    let object = value
        .as_object_mut()
        .ok_or_else(|| super::invalid("execution metadata must be an object"))?;
    object.remove("context");
    object.remove("device");
    let buffers = object
        .get_mut("buffers")
        .and_then(|v| v.as_array_mut())
        .ok_or_else(|| super::invalid("execution buffers must be an array"))?;
    for buffer in buffers {
        let buffer = buffer
            .as_object_mut()
            .ok_or_else(|| super::invalid("invalid execution buffer"))?;
        if let Some(address) = buffer.remove("address") {
            let address = address
                .as_u64()
                .ok_or_else(|| super::invalid("invalid buffer address"))?;
            let nbytes = buffer
                .get("nbytes")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| super::invalid("invalid buffer extent"))?;
            let allocation = allocations
                .iter()
                .find(|a| {
                    address >= a.address
                        && address
                            .checked_add(nbytes)
                            .is_some_and(|end| end <= a.address + a.nbytes)
                })
                .ok_or_else(|| super::invalid("execution buffer is outside registered storage"))?;
            buffer.insert("reference".into(),serde_json::json!({"allocation":allocation.name,"offset":address-allocation.address}));
        }
    }
    let graphs = object
        .get_mut("graphs")
        .and_then(|v| v.as_array_mut())
        .ok_or_else(|| super::invalid("execution graphs must be an array"))?;
    for graph in graphs {
        let graph = graph
            .as_object_mut()
            .ok_or_else(|| super::invalid("invalid execution graph"))?;
        graph.remove("handle");
        let kind = graph
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_owned();
        let width = graph
            .get("token_capacity")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let speculative = graph
            .get("speculation_enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if kind == "packed_speculative" {
            graph.insert("verify_tokens".into(), width.into());
            graph.insert("draft_source".into(), "MTP".into());
        } else if kind == "speculative" || kind == "mixed" && speculative {
            graph.insert("draft_tokens".into(), width.saturating_sub(1).into());
            graph.insert("verify_tokens".into(), width.into());
            graph.insert("draft_source".into(), "MTP".into());
        } else if kind == "proposal" {
            graph.insert("draft_tokens".into(), width.saturating_sub(1).into());
            graph.insert("draft_source".into(), "MTP".into());
        } else if kind == "verification" {
            graph.insert("verify_tokens".into(), width.into());
        }
    }
    message("Execution", &value)
}

pub(crate) fn finish_rank(
    request: &CaptureRequest,
    output: &mut super::RankOutput,
) -> super::Result<()> {
    use super::{FilePart, FileSource, wire};
    let mut cursor = output
        .payload
        .iter()
        .map(|p| p.offset + p.source.nbytes)
        .max()
        .unwrap_or(0);
    if let Some(server) = request.server.as_ref() {
        let assets = serving_metadata(&mut output.index, server)?;
        let progress = if request.rank == 0 {
            helios_core::startup::StartupProgress::stderr_rows()
        } else {
            helios_core::startup::StartupProgress::silent()
        };
        let phase = progress.phase("Export frontend assets", None);
        for (name, asset) in assets {
            let stage = output
                .staging
                .as_ref()
                .ok_or_else(|| super::invalid("asset export requires rank staging storage"))?;
            let (path, nbytes, digest) = write_asset(stage, &name, &asset, request.world_size)?;
            phase.advance_bytes(nbytes);
            cursor = align(cursor)?;
            output.index.assets.push(wire::Asset {
                name: name.clone(),
                sha256: digest.into(),
                payload: Some(wire::Span {
                    offset: cursor,
                    nbytes,
                    ..Default::default()
                }),
            });
            output.payload.push(FilePart {
                offset: cursor,
                source: FileSource {
                    path,
                    offset: 0,
                    nbytes,
                },
            });
            cursor = cursor
                .checked_add(nbytes)
                .ok_or_else(|| super::invalid("payload extent overflow"))?;
        }
        phase.finish("");
    }
    if request.proposal_source == "copy" {
        if let Some(execution) = output.index.ranks[0].execution.as_mut() {
            for graph in &mut execution.graphs {
                if graph.kind == "speculative"
                    || graph.kind == "mixed" && graph.speculation_enabled == Some(true)
                {
                    graph.draft_source = wire::DraftSource::CopyDraft as i32;
                }
            }
        }
    }
    if let Some(data) = request.package_data.as_ref() {
        let mut data = data.clone();
        if let Some(tables) = data.get_mut("tables").and_then(|v| v.as_array_mut()) {
            for table in tables {
                append_planes(table, &mut cursor, &mut output.payload)?;
                if let Some(shards) = table.get_mut("shards").and_then(|v| v.as_array_mut()) {
                    for shard in shards {
                        append_planes(shard, &mut cursor, &mut output.payload)?;
                    }
                }
            }
        }
        output.index.engram = Some(message("Engram", &data)?);
    }
    if let Some(weights) = request.rebuild_weights.as_ref() {
        let mut weights = weights.clone();
        let mut unavailable = weights
            .get_mut("unavailable")
            .and_then(|v| v.as_array_mut())
            .ok_or_else(|| super::invalid("invalid unavailable weight list"))?
            .clone();
        let tensors = weights
            .get_mut("tensors")
            .and_then(|v| v.as_array_mut())
            .ok_or_else(|| super::invalid("invalid weight view list"))?;
        let mut retained = Vec::new();
        for tensor in tensors.drain(..) {
            let mut tensor = tensor
                .as_object()
                .ok_or_else(|| super::invalid("invalid weight view"))?
                .clone();
            let address = tensor
                .remove("address")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| super::invalid("weight address is required"))?;
            let extent = tensor
                .remove("extent")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| super::invalid("weight extent is required"))?;
            let end = address
                .checked_add(extent)
                .ok_or_else(|| super::invalid("weight extent overflow"))?;
            let allocation = output.allocations.iter().find(|a| {
                a.memory != "peer"
                    && a.sizing.is_none()
                    && a.address <= address
                    && end <= a.address + a.nbytes
            });
            if let Some(allocation) = allocation.filter(|_| {
                !request
                    .zero_ranges
                    .iter()
                    .any(|&(start, n)| address < start + n && start < end)
            }) {
                tensor.insert("storage".into(),serde_json::json!({"allocation":allocation.name,"offset":address-allocation.address}));
                retained.push(tensor.into());
            } else {
                unavailable.push(
                    tensor
                        .get("name")
                        .cloned()
                        .ok_or_else(|| super::invalid("weight name is required"))?,
                );
            }
        }
        *tensors = retained;
        weights["unavailable"] = unavailable.into();
        let weights = message("RebuildWeights", &weights)?;
        validate_rebuild_weights(&weights, &output.index.ranks[0].allocations)?;
        output.index.ranks[0].rebuild_weights = Some(weights);
    }
    for transport in &request.transports {
        if output.index.ranks[0].roce.is_some() {
            return Err(super::invalid("a rank exports at most one RoCE transport"));
        }
        let transport = serde_json::json!({
            "proxy_abi":transport["proxy_abi"],"world":transport["world_size"],"rank":transport["rank"],
            "region_address":transport["region"]["address"],"region_bytes":transport["region"]["nbytes"],
            "counters_address":transport["counters"]["address"],"counters_bytes":transport["counters"]["nbytes"],
            "slot_bytes":transport["slot_bytes"],"pipe_bytes":transport["pipe_bytes"],
            "ctrl_offset":transport["ctrl_offset"],"varlen_bytes":transport.get("varlen_bytes").cloned().unwrap_or(0.into())
        });
        output.index.ranks[0].roce = Some(message("RoceTransport", &transport)?);
    }
    Ok(())
}

fn validate_rebuild_weights(
    catalog: &super::wire::RebuildWeights,
    allocations: &[super::wire::Allocation],
) -> super::Result<()> {
    use std::collections::{HashMap, HashSet};
    if catalog.version != 1 {
        return Err(super::invalid(format!(
            "unsupported prepared-weight catalog version {}", catalog.version
        )));
    }
    let allocations: HashMap<_, _> = allocations.iter().map(|a| (a.name.as_str(), a)).collect();
    let mut names = HashSet::new();
    for tensor in &catalog.tensors {
        let width = match tensor.dtype.as_str() {
            "bool" | "uint8" | "int8" | "float8_e4m3fn" | "float8_e5m2"
            | "float8_e4m3fnuz" | "float8_e5m2fnuz" | "float8_e8m0fnu" => Some(1),
            "int16" | "uint16" | "float16" | "bfloat16" => Some(2),
            "int32" | "uint32" | "float32" | "complex32" => Some(4),
            "int64" | "uint64" | "float64" | "complex64" => Some(8),
            "complex128" => Some(16),
            _ => None,
        };
        if tensor.name.is_empty() || !names.insert(tensor.name.as_str())
            || tensor.storage.is_none() || width != Some(tensor.element_bytes)
            || tensor.shape.len() != tensor.stride.len() || tensor.shape.contains(&0)
        {
            return Err(super::invalid("invalid prepared-weight tensor description"));
        }
        let extent = tensor.shape.iter().zip(&tensor.stride).try_fold(1u64, |extent, (&size, &stride)| {
            (size - 1).checked_mul(stride).and_then(|span| extent.checked_add(span))
        }).and_then(|extent| extent.checked_mul(u64::from(tensor.element_bytes)));
        let storage = tensor.storage.as_ref().unwrap();
        let allocation = allocations.get(storage.allocation.as_str());
        let valid = allocation.is_some_and(|a| a.sizing.is_none()
            && a.memory != super::wire::Memory::Peer as i32
            && extent.and_then(|n| storage.offset.checked_add(n)).is_some_and(|end| end <= a.nbytes));
        if !valid {
            return Err(super::invalid(format!(
                "prepared-weight tensor {:?} exceeds fixed stored allocation", tensor.name
            )));
        }
    }
    for name in &catalog.unavailable {
        if name.is_empty() || !names.insert(name.as_str()) {
            return Err(super::invalid("duplicate or empty unavailable prepared-weight name"));
        }
    }
    Ok(())
}

fn align(value: u64) -> super::Result<u64> {
    value
        .checked_add(4095)
        .map(|v| v & !4095)
        .ok_or_else(|| super::invalid("payload extent overflow"))
}

fn append_planes(
    value: &mut serde_json::Value,
    cursor: &mut u64,
    payload: &mut Vec<super::FilePart>,
) -> super::Result<()> {
    for name in ["weights", "scales"] {
        if let Some(parts) = value.get_mut(name) {
            let sources: Vec<super::FileSource> = serde_json::from_value(parts.clone())?;
            *cursor = align(*cursor)?;
            let start = *cursor;
            for source in sources {
                if source.nbytes == 0
                    || source.offset.checked_add(source.nbytes).is_none_or(|end| {
                        std::fs::metadata(&source.path)
                            .map(|s| end > s.len())
                            .unwrap_or(true)
                    })
                {
                    return Err(super::invalid(
                        "embedding plane source exceeds checkpoint file",
                    ));
                }
                payload.push(super::FilePart {
                    offset: *cursor,
                    source: source.clone(),
                });
                *cursor = cursor
                    .checked_add(source.nbytes)
                    .ok_or_else(|| super::invalid("embedding plane extent overflow"))?;
            }
            *parts = serde_json::json!({"offset":start,"nbytes":*cursor-start});
        }
    }
    Ok(())
}

#[pyclass(module = "flare_heliopause")]
struct NativeExportStage {
    directory: std::sync::Mutex<Option<tempfile::TempDir>>,
    #[pyo3(get)]
    path: String,
}

#[pymethods]
impl NativeExportStage {
    #[new]
    #[pyo3(signature=(destination, staging=None))]
    fn new(
        py: Python<'_>,
        destination: std::path::PathBuf,
        staging: Option<std::path::PathBuf>,
    ) -> PyResult<Self> {
        let directory = py
            .detach(|| -> super::Result<_> {
                if destination.try_exists()? {
                    return Err(super::invalid("export destination already exists"));
                }
                let parent = destination.parent().unwrap_or(std::path::Path::new("."));
                std::fs::create_dir_all(parent)?;
                let staging = staging.as_deref().unwrap_or(parent);
                std::fs::create_dir_all(staging)?;
                Ok(tempfile::Builder::new()
                    .prefix(".flare-export-")
                    .tempdir_in(staging)?)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            path: directory.path().to_string_lossy().into_owned(),
            directory: std::sync::Mutex::new(Some(directory)),
        })
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let directory = self
            .directory
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(directory) = directory {
            py.detach(|| directory.close())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(())
    }
}

#[pyfunction]
fn export_source_identity(py: Python<'_>, roots: Vec<std::path::PathBuf>) -> PyResult<String> {
    py.detach(|| super::source_identity::source_identity(&roots))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

fn validate_sampling(value: &serde_json::Value) -> super::Result<()> {
    let values = value
        .as_object()
        .ok_or_else(|| super::invalid("sampling defaults must be an object"))?;
    let names = [
        "temperature",
        "top_p",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    ];
    let valid = |name: &str, value: &serde_json::Value| -> bool {
        if name == "top_k" {
            return value
                .as_i64()
                .is_some_and(|n| (-1..=i32::MAX as i64).contains(&n));
        }
        if name == "seed" {
            return value.as_u64().is_some();
        }
        let Some(v) = value.as_f64() else {
            return false;
        };
        v.is_finite()
            && v.abs() <= f32::MAX as f64
            && (!matches!(name, "temperature" | "top_p" | "repetition_penalty")
                || v >= f32::MIN_POSITIVE as f64)
            && (name != "top_p" || v <= 1.)
            && (name != "min_p" || (0. ..=1.).contains(&v))
    };
    for name in names.into_iter().chain(["top_k", "seed"]) {
        if !values.get(name).is_some_and(|v| valid(name, v)) {
            return Err(super::invalid("invalid recipe sampling defaults"));
        }
    }
    if let Some(ranges) = values.get("ranges") {
        let ranges = ranges
            .as_object()
            .ok_or_else(|| super::invalid("invalid sampling ranges"))?;
        for (name, bounds) in ranges {
            if name == "seed" || !names.contains(&name.as_str()) && name != "top_k" {
                return Err(super::invalid(format!("invalid sampling range for {name}")));
            }
            let min = bounds
                .get("min")
                .filter(|v| valid(name, v))
                .and_then(|v| v.as_f64());
            let max = bounds
                .get("max")
                .filter(|v| valid(name, v))
                .and_then(|v| v.as_f64());
            if !min.zip(max).is_some_and(|(min, max)| {
                min <= values[name].as_f64().unwrap() && values[name].as_f64().unwrap() <= max
            }) {
                return Err(super::invalid(format!("invalid sampling range for {name}")));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_metadata(request: &CaptureRequest) -> super::Result<()> {
    if let Some(server) = request.server.as_ref() {
        serving_metadata(&mut Default::default(), server)?;
    }
    Ok(())
}

pub(crate) fn serving_metadata(
    index: &mut super::wire::ModelIndex,
    server: &serde_json::Value,
) -> super::Result<serde_json::Map<String, serde_json::Value>> {
    use super::wire;
    let mut server = server
        .as_object()
        .ok_or_else(|| super::invalid("serving metadata must be an object"))?
        .clone();
    let assets = server
        .remove("assets")
        .ok_or_else(|| super::invalid("serving assets are required"))?;
    let sampling = server
        .remove("sampling_defaults")
        .ok_or_else(|| super::invalid("sampling defaults are required"))?;
    let frontend = server
        .remove("frontend")
        .ok_or_else(|| super::invalid("frontend metadata is required"))?;
    if server.contains_key("generation_defaults") || server.contains_key("eos_token_ids") {
        return Err(super::invalid(
            "sampling and token defaults require typed package fields",
        ));
    }
    validate_sampling(&sampling)?;
    let sampling: wire::DefaultSamplingParameters =
        message("DefaultSamplingParameters", &sampling)?;
    helios_core::runtime_package::validate_sampling_defaults(&sampling)?;
    if sampling.seed.is_none()
        || sampling.temperature.is_none()
        || sampling.top_k.is_none()
        || sampling.top_p.is_none()
        || sampling.min_p.is_none()
        || sampling.repetition_penalty.is_none()
        || sampling.presence_penalty.is_none()
        || sampling.frequency_penalty.is_none()
    {
        return Err(super::invalid(
            "sampling defaults must explicitly declare every parameter",
        ));
    }
    index.sampling_defaults = Some(sampling);
    let frontend: wire::FrontendSettings = message("FrontendSettings", &frontend)?;
    super::wire::PromptEncoder::try_from(frontend.prompt_encoder)
        .map_err(|_| super::invalid("unsupported prompt encoder"))?;
    let expected = if frontend.prompt_encoder == 0 {
        vec!["chat_template", "tokenizer"]
    } else {
        vec!["tokenizer"]
    };
    let assets = assets
        .as_object()
        .ok_or_else(|| super::invalid("serving assets must be an object"))?;
    let mut names = assets.keys().map(String::as_str).collect::<Vec<_>>();
    names.sort();
    if names != expected {
        return Err(super::invalid(
            "runtime assets must match the prompt encoder",
        ));
    }
    index.frontend = Some(frontend);
    if let Some(rebuild) = server.remove("rebuild_manifest") {
        index.rebuild = Some(message("RebuildManifest", &rebuild)?);
    }
    server
        .entry("speculative_proposal_source")
        .or_insert("mtp".into());
    index.contract = Some(message("ServingContract", &server.into())?);
    Ok(assets.clone())
}

#[pyfunction]
fn export_serving<'py>(
    py: Python<'py>,
    index: Vec<u8>,
    server: &str,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    use prost::Message;
    let server: serde_json::Value =
        serde_json::from_str(server).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let encoded = py
        .detach(|| -> super::Result<_> {
            let mut index = super::wire::ModelIndex::decode(index.as_slice())?;
            serving_metadata(&mut index, &server)?;
            Ok(index.encode_to_vec())
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(pyo3::types::PyBytes::new(py, &encoded))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rebuild_catalog() -> (super::super::wire::RebuildWeights, Vec<super::super::wire::Allocation>) {
        use super::super::wire;
        (wire::RebuildWeights {
            version: 1,
            tensors: vec![wire::RebuildTensor {
                name: "weight".into(), dtype: "float32".into(), element_bytes: 4,
                shape: vec![4], stride: vec![1],
                storage: Some(wire::Reference { allocation: "weights".into(), offset: 0 }),
                ..Default::default()
            }],
            unavailable: vec!["discarded".into()],
        }, vec![wire::Allocation { name: "weights".into(), nbytes: 64, ..Default::default() }])
    }
    #[test]
    fn rebuild_weight_validation_preserves_scalar_strided_and_broadcast_views() {
        let (catalog, allocations) = rebuild_catalog();
        for (shape, stride) in [(vec![], vec![]), (vec![2, 3], vec![1, 2]),
            (vec![u64::MAX], vec![0])] {
            let mut catalog = catalog.clone();
            catalog.tensors[0].shape = shape;
            catalog.tensors[0].stride = stride;
            catalog.tensors[0].storage.as_mut().unwrap().offset = 16;
            validate_rebuild_weights(&catalog, &allocations).unwrap();
        }
    }
    #[test]
    fn rebuild_weight_validation_rejects_malformed_catalogs_and_checked_extent_overflow() {
        let (catalog, allocations) = rebuild_catalog();
        for change in 0..19 {
            let mut catalog = catalog.clone();
            let mut allocations = allocations.clone();
            match change {
                0 => catalog.version = 99,
                1 => catalog.tensors[0].name.clear(),
                2 => catalog.tensors.push(catalog.tensors[0].clone()),
                3 => catalog.tensors[0].storage = None,
                4 => catalog.tensors[0].dtype = "unknown".into(),
                5 => catalog.tensors[0].element_bytes = 8,
                6 => catalog.tensors[0].stride.clear(),
                7 => catalog.tensors[0].shape[0] = 0,
                8 => catalog.tensors[0].storage.as_mut().unwrap().allocation = "absent".into(),
                9 => allocations[0].sizing = Some(Default::default()),
                10 => allocations[0].memory = super::super::wire::Memory::Peer as i32,
                11 => catalog.tensors[0].storage.as_mut().unwrap().offset = 49,
                12 => catalog.unavailable.push(String::new()),
                13 => catalog.unavailable.push("weight".into()),
                14 => catalog.unavailable.push("discarded".into()),
                15 => { catalog.tensors[0].shape = vec![u64::MAX]; catalog.tensors[0].stride = vec![u64::MAX]; },
                16 => { catalog.tensors[0].shape = vec![u64::MAX, 2]; catalog.tensors[0].stride = vec![1, 1]; },
                17 => { catalog.tensors[0].shape = vec![u64::MAX]; catalog.tensors[0].stride = vec![1]; },
                18 => catalog.tensors[0].storage.as_mut().unwrap().offset = u64::MAX - 3,
                _ => unreachable!(),
            }
            assert!(validate_rebuild_weights(&catalog, &allocations).is_err(), "change {change}");
        }
    }
    #[test]
    fn native_rank_emission_validates_resolved_rebuild_weights() {
        let request: super::super::CaptureRequest = serde_json::from_value(serde_json::json!({
            "path":"unused", "context":0, "graphs":[], "allocations":[], "execution":null,
            "server":null, "package_data":null, "shard_size":67108864,
            "rebuild_weights":{"version":1,"tensors":[{"name":"weight","dtype":"float32",
                "element_bytes":4,"shape":[4],"stride":[1],"address":4096,"extent":16}],"unavailable":[]}
        })).unwrap();
        let (_, allocations) = rebuild_catalog();
        let output = super::super::RankOutput {
            index: super::super::wire::ModelIndex {
                version: 4, ranks: vec![super::super::wire::BundleIndex { allocations, ..Default::default() }],
                ..Default::default()
            },
            allocations: vec![serde_json::from_value(serde_json::json!({"name":"weights","address":4096,"nbytes":64})).unwrap()],
            payload: vec![], module_globals: vec![], external: vec![], staging: None,
        };
        let mut valid = output.clone();
        finish_rank(&request, &mut valid).unwrap();
        let tensor = &valid.index.ranks[0].rebuild_weights.as_ref().unwrap().tensors[0];
        assert_eq!(tensor.storage.as_ref().unwrap().allocation, "weights");
        for change in 0..6 {
            let mut request = request.clone();
            let weights = request.rebuild_weights.as_mut().unwrap();
            match change {
                0 => weights["version"] = 99.into(),
                1 => weights["tensors"][0]["dtype"] = "unknown".into(),
                2 => weights["tensors"][0]["shape"] = serde_json::json!([17]),
                3 => {
                    let tensor = weights["tensors"][0].clone();
                    weights["tensors"].as_array_mut().unwrap().push(tensor);
                },
                4 => weights["unavailable"] = serde_json::json!(["weight"]),
                5 => weights["tensors"][0]["stride"] = serde_json::json!([]),
                _ => unreachable!(),
            }
            let mut invalid = output.clone();
            assert!(finish_rank(&request, &mut invalid).is_err(), "change {change}");
            assert!(invalid.index.ranks[0].rebuild_weights.is_none());
        }
    }
    fn sampling() -> serde_json::Value {
        serde_json::json!({"seed":0,"temperature":1.,"top_k":-1,"top_p":1.,"min_p":0.,"repetition_penalty":1.,"presence_penalty":0.,"frequency_penalty":0.})
    }
    #[test]
    fn sampling_metadata_rejects_double_precision_out_of_range_before_float_rounding() {
        for (name, value) in [
            ("top_p", serde_json::json!(1.0000000001)),
            ("temperature", serde_json::json!(0.)),
            ("top_k", serde_json::json!(1.5)),
            ("seed", serde_json::json!(-1)),
        ] {
            let mut input = sampling();
            input[name] = value;
            assert!(validate_sampling(&input).is_err());
        }
        let mut input = sampling();
        input["ranges"] = serde_json::json!({"temperature":{"min":0.5,"max":2.}});
        assert!(validate_sampling(&input).is_ok());
        input["ranges"]["temperature"]["min"] = serde_json::json!(1.1);
        assert!(validate_sampling(&input).is_err());
    }
    #[test]
    fn native_metadata_parser_preserves_enum_oneof_and_explicit_defaults() {
        let input = serde_json::json!({"image":{"glm":{}},"prompt_encoder":"DEEPSEEK_V4","eos_token_ids":[2]});
        let parsed: super::super::wire::FrontendSettings =
            message("FrontendSettings", &input).unwrap();
        assert_eq!(parsed.prompt_encoder, 1);
        assert!(matches!(
            parsed.image.unwrap().profile,
            Some(super::super::wire::image_settings::Profile::Glm(_))
        ));
        let parsed: super::super::wire::ServingContract =
            message("ServingContract", &serde_json::json!({"max_active":0})).unwrap();
        assert_eq!(parsed.max_active, Some(0));
        assert!(
            message::<super::super::wire::ServingContract>(
                "ServingContract",
                &serde_json::json!({"scheduler_policy":"model"})
            )
            .is_err()
        );
    }
    #[test]
    fn bundle_only_copy_proposals_retain_their_native_draft_source() {
        let request:super::super::CaptureRequest=serde_json::from_value(serde_json::json!({"path":"unused","context":0,"graphs":[],"allocations":[],"proposal_source":"copy","execution":null,"server":null,"rebuild_weights":null,"package_data":null,"shard_size":67108864})).unwrap();
        let graphs = [
            ("speculative", None),
            ("mixed", Some(true)),
            ("mixed", Some(false)),
            ("packed_speculative", None),
        ]
        .into_iter()
        .map(
            |(kind, speculation_enabled)| super::super::wire::ExecutionGraph {
                kind: kind.into(),
                speculation_enabled,
                draft_source: 1,
                ..Default::default()
            },
        )
        .collect();
        let mut output = super::super::RankOutput {
            index: super::super::wire::ModelIndex {
                version: 4,
                ranks: vec![super::super::wire::BundleIndex {
                    execution: Some(super::super::wire::Execution {
                        graphs,
                        ..Default::default()
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            },
            payload: vec![],
            allocations: vec![],
            module_globals: vec![],
            external: vec![],
            staging: None,
        };
        finish_rank(&request, &mut output).unwrap();
        let sources: Vec<_> = output.index.ranks[0]
            .execution
            .as_ref()
            .unwrap()
            .graphs
            .iter()
            .map(|g| g.draft_source)
            .collect();
        assert_eq!(sources, [2, 2, 1, 1]);
    }
    #[test]
    fn execution_buffers_resolve_allocation_relative_offsets() {
        let allocation = super::super::Allocation {
            name: "storage".into(),
            address: 4096,
            nbytes: 512,
            alignment: 256,
            initialize: "snapshot".into(),
            relocations: vec![],
            preserve_address: true,
            memory: "device".into(),
            snapshot_spans: None,
            sizing: None,
        };
        let input = serde_json::json!({"abi_version":3,"context":42,"device":0,"buffers":[{"name":"token","address":4128,"nbytes":4,"dtype":"int32","shape":[1],"strides":[1]}],"graphs":[{"name":"verify","kind":"speculative","handle":123,"token_capacity":3}]});
        let parsed = execution(&input, &[allocation]).unwrap();
        assert_eq!(parsed.buffers[0].reference.as_ref().unwrap().offset, 32);
        assert_eq!(parsed.graphs[0].draft_tokens, Some(2));
        assert_eq!(parsed.graphs[0].verify_tokens, Some(3));
    }
}

#[pyfunction]
fn export_synchronize(py: Python<'_>, context: usize, owners: Py<PyAny>) -> PyResult<()> {
    if owners.bind(py).is_none() {
        return Err(PyValueError::new_err(
            "CUDA drain requires retained preparation owners",
        ));
    }
    let mut lease = OwnerLease(Some(owners));
    py.detach(|| super::capture::drain_context(context))
        .map_err(|e| {
            PyRuntimeError::new_err(format!(
                "export CUDA drain failed; preparation owners retained: {e}"
            ))
        })?;
    drop(lease.take_after_drain());
    Ok(())
}

fn write_asset(
    stage: &std::path::Path,
    name: &str,
    asset: &serde_json::Value,
    ranks: usize,
) -> super::Result<(std::path::PathBuf, u64, String)> {
    use sha2::{Digest, Sha256};
    use std::io::Write;
    let path = stage.join(format!("asset-{name}.bin"));
    let mut output = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)?;
    let mut digest = Sha256::new();
    let mut length = 0;
    match (
        asset.get("path").and_then(|v| v.as_str()),
        asset.get("text").and_then(|v| v.as_str()),
    ) {
        (Some(source), None) => {
            let nbytes = std::fs::metadata(source)?.len();
            let source = super::io::Source::open(std::path::Path::new(source), 0, nbytes)?;
            let budget = super::io::IoBudget::for_ranks(ranks)?;
            budget.scan(&source, |bytes| {
                digest.update(bytes);
                output.write_all(bytes)?;
                length += bytes.len() as u64;
                Ok(())
            })?;
        }
        (None, Some(text)) => {
            output.write_all(text.as_bytes())?;
            digest.update(text.as_bytes());
            length = text.len() as u64;
        }
        _ => {
            return Err(super::invalid(
                "asset requires exactly one file path or text value",
            ));
        }
    }
    output.sync_all()?;
    let digest = format!("{:x}", digest.finalize());
    if asset
        .get("sha256")
        .and_then(|v| v.as_str())
        .is_some_and(|expected| expected != digest)
    {
        return Err(super::invalid(
            "frontend asset digest differs from preparation metadata",
        ));
    }
    Ok((path, length, digest))
}

#[cfg(test)]
mod asset_tests {
    #[test]
    fn copied_and_inline_frontend_assets_hash_the_exact_staged_bytes() {
        use sha2::{Digest, Sha256};
        let root = tempfile::tempdir().unwrap();
        let bytes = b"{{ messages }}\n";
        let source = root.path().join("template.jinja");
        std::fs::write(&source, bytes).unwrap();
        let expected = format!("{:x}", Sha256::digest(bytes));
        let (copied, nbytes, digest) = super::write_asset(
            root.path(),
            "copied",
            &serde_json::json!({"path":source,"sha256":expected}),
            1,
        )
        .unwrap();
        assert_eq!(std::fs::read(copied).unwrap(), bytes);
        assert_eq!(nbytes, bytes.len() as u64);
        assert_eq!(digest, expected);
        let (inline, _, digest) = super::write_asset(
            root.path(),
            "inline",
            &serde_json::json!({"text":"{{ messages }}\n"}),
            1,
        )
        .unwrap();
        assert_eq!(std::fs::read(inline).unwrap(), bytes);
        assert_eq!(digest, expected);
        assert!(
            super::write_asset(
                root.path(),
                "invalid",
                &serde_json::json!({"path":source,"sha256":"wrong"}),
                1
            )
            .is_err()
        );
    }
}

#[pyclass(frozen, module = "flare_heliopause")]
pub(crate) struct NativeRankOutput {
    pub(crate) output: super::RankOutput,
}

#[pymethods]
impl NativeRankOutput {
    #[staticmethod]
    fn from_sources(py: Python<'_>, path: std::path::PathBuf) -> PyResult<Self> {
        let output = py
            .detach(|| super::package::read_sources(&path))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { output })
    }
    fn summary(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(||->super::Result<String>{
            let bundle=self.output.index.ranks.first().ok_or_else(||super::invalid("rank output lacks resource bundle"))?;
            let cubin_bytes=bundle.modules.iter().try_fold(0u64,|size,module|->super::Result<_>{
                let bytes=bundle.resources.get(&module.file).ok_or_else(||super::invalid("module resource is absent"))?.nbytes;
                size.checked_add(bytes).ok_or_else(||super::invalid("module byte count overflow"))
            })?;
            Ok(serde_json::to_string(&serde_json::json!({"allocations":self.output.allocations,
                "module_globals":self.output.module_globals,"external":self.output.external,
                "export_counts":{"kernels":bundle.functions.len(),"modules":bundle.modules.len(),"cubin_bytes":cubin_bytes}}))?)
        }).map_err(|e|PyRuntimeError::new_err(e.to_string()))
    }
}

#[pyfunction]
fn export_reset_peer_storage(
    py: Python<'_>,
    context: usize,
    ranges: Vec<(u64, u64)>,
    owners: Py<PyAny>,
) -> PyResult<()> {
    if owners.bind(py).is_none() {
        return Err(PyValueError::new_err(
            "collective reset requires retained preparation storage",
        ));
    }
    let mut lease = OwnerLease(Some(owners));
    let (result, drain) = py.detach(|| {
        (
            super::capture::reset_peer_storage(context, &ranges),
            super::capture::drain_context(context),
        )
    });
    if drain.is_ok() {
        drop(lease.take_after_drain());
    }
    if let Err(error) = drain {
        let primary = result
            .as_ref()
            .err()
            .map(|e| format!("{e}; "))
            .unwrap_or_default();
        return Err(PyRuntimeError::new_err(format!(
            "{primary}collective reset CUDA drain failed; preparation owners retained: {error}"
        )));
    }
    result.map_err(|e| PyRuntimeError::new_err(e.to_string()))
}
