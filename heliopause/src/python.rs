#[cfg(target_os = "linux")]
use std::os::fd::BorrowedFd;
use std::sync::Mutex;
#[cfg(any(target_os = "linux", feature = "qualification"))]
use std::time::Duration;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::execution::{
    BufferDescriptor, Dtype, ExecutionManifest, ExecutionWorker, GraphDescriptor, GraphKind,
    InputBinding, OwnedUpload, SamplingMode,
};
use crate::lifetime::OwnerLease;
#[cfg(feature = "qualification")]
use crate::{BufferSpan, GraphBundle, ReplayWorker};

#[cfg(feature = "qualification")]
#[pyclass(module = "flare_heliopause")]
struct ReplayCompletion {
    #[pyo3(get)]
    ticket: u64,
    #[pyo3(get)]
    graph_index: usize,
    #[pyo3(get)]
    repetitions: u32,
    #[pyo3(get)]
    error: Option<String>,
}

/// Borrow a prepared model's executable graphs and replay them on one native stream.
///
/// The owner must retain every captured allocation, executable, and CUDA context.
/// Graph handles must not be reinstantiated and tensor storage must not be resized
/// or mutated concurrently. Captured PyTorch generator-backed RNG is unsupported.
#[cfg(feature = "qualification")]
#[pyclass(module = "flare_heliopause")]
struct GraphReplayWorker {
    worker: ReplayWorker,
    owner: Mutex<OwnerLease<Py<PyAny>>>,
}

#[cfg(feature = "qualification")]
#[pymethods]
impl GraphReplayWorker {
    #[new]
    #[pyo3(signature = (owner, context, device, graph_execs, buffers, queue_capacity=8))]
    fn new(
        py: Python<'_>,
        owner: Py<PyAny>,
        context: usize,
        device: i32,
        graph_execs: Vec<usize>,
        buffers: Vec<(usize, usize)>,
        queue_capacity: usize,
    ) -> PyResult<Self> {
        if owner.bind(py).is_none() {
            return Err(PyValueError::new_err(
                "owner must retain the prepared graph resources",
            ));
        }
        let bundle = GraphBundle {
            context,
            device,
            graph_execs,
            buffers: buffers
                .into_iter()
                .map(|(address, nbytes)| BufferSpan { address, nbytes })
                .collect(),
        };
        bundle
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let worker = py
            .detach(|| {
                // SAFETY: Preparation supplies ready handles and retains their
                // captured allocations in owner. This reference stays live during
                // startup; OwnerLease retains it afterward until confirmed drain.
                unsafe { ReplayWorker::new(bundle, queue_capacity) }
            })
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self {
            worker,
            owner: Mutex::new(OwnerLease(Some(owner))),
        })
    }

    #[pyo3(signature = (graph_index, repetitions=1))]
    fn submit(&self, graph_index: usize, repetitions: u32) -> PyResult<u64> {
        self.worker
            .submit(graph_index, repetitions)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    #[pyo3(signature = (timeout_ms=0))]
    fn poll(&self, py: Python<'_>, timeout_ms: u64) -> PyResult<Option<ReplayCompletion>> {
        py.detach(|| self.worker.poll(Duration::from_millis(timeout_ms)))
            .map(|completion| {
                completion.map(|value| ReplayCompletion {
                    ticket: value.ticket,
                    graph_index: value.graph_index,
                    repetitions: value.repetitions,
                    error: value.error.map(|error| error.to_string()),
                })
            })
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    #[getter]
    fn outstanding(&self) -> usize {
        self.worker.outstanding()
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let shutdown = py.detach(|| self.worker.close());
        if shutdown.drained {
            let owner = self
                .owner
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take_after_drain();
            drop(owner);
        }
        match shutdown.error {
            Some(error) => Err(PyRuntimeError::new_err(error.to_string())),
            None => Ok(()),
        }
    }
}

#[cfg(feature = "qualification")]
impl Drop for GraphReplayWorker {
    fn drop(&mut self) {
        let shutdown = self.worker.close();
        if shutdown.drained {
            let owner = self
                .owner
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .take_after_drain();
            drop(owner);
        }
    }
}

#[pyfunction]
fn current_context() -> PyResult<(usize, i32)> {
    crate::current_context().map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

pub(crate) fn execution_manifest(value: &Bound<'_, PyAny>) -> PyResult<ExecutionManifest> {
    macro_rules! field {
        ($value:expr, $name:literal, $kind:ty) => {
            $value.get_item($name)?.extract::<$kind>()?
        };
    }
    let mut buffers = Vec::new();
    for buffer in value.get_item("buffers")?.try_iter()? {
        let buffer = buffer?;
        buffers.push(BufferDescriptor {
            name: field!(buffer, "name", String),
            address: field!(buffer, "address", usize),
            nbytes: field!(buffer, "nbytes", usize),
            dtype: Dtype::try_from(field!(buffer, "dtype", String).as_str())
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
            shape: field!(buffer, "shape", Vec<usize>),
            strides: field!(buffer, "strides", Vec<usize>),
        });
    }
    let mut graphs = Vec::new();
    for graph in value.get_item("graphs")?.try_iter()? {
        let graph = graph?;
        let mut input_bindings = Vec::new();
        for binding in graph.get_item("input_bindings")?.try_iter()? {
            let binding = binding?;
            input_bindings.push(InputBinding {
                buffer_name: field!(binding, "buffer_name", String),
                host_offset: field!(binding, "host_offset", usize),
                nbytes: field!(binding, "nbytes", usize),
            });
        }
        graphs.push(GraphDescriptor {
            draft_tokens: if field!(graph, "kind", String) == "packed_speculative" { field!(graph, "draft_tokens", usize) } else { 0 },
            name: field!(graph, "name", String),
            kind: GraphKind::try_from(field!(graph, "kind", String).as_str())
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
            sampling_mode: SamplingMode::try_from(field!(graph, "sampling_mode", String).as_str())
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
            handle: field!(graph, "handle", usize),
            batch_capacity: field!(graph, "batch_capacity", usize),
            token_capacity: field!(graph, "token_capacity", usize),
            packed_token_capacity: graph.get_item("packed_token_capacity").ok()
                .map(|value| value.extract::<usize>()).transpose()?.unwrap_or(0),
            prefill_capacity: field!(graph, "prefill_capacity", usize),
            estimated_duration_ns: field!(graph, "estimated_duration_ns", u64),
            input_bindings,
        });
    }
    Ok(ExecutionManifest {
        collectives: None,
        engram: crate::embedding_tables::binding(value)?,
        cache: None,
        offload: None,
        abi_version: field!(value, "abi_version", u32),
        context: field!(value, "context", usize),
        device: field!(value, "device", i32),
        max_active: field!(value, "max_active", usize),
        input_frame_bytes: field!(value, "input_frame_bytes", usize),
        buffers,
        graphs,
    })
}

#[pyclass(module = "flare_heliopause")]
struct ExecutionCompletion {
    #[pyo3(get)]
    fence: u64,
    #[pyo3(get)]
    graph_index: usize,
    #[pyo3(get)]
    proposal_confidence: Option<Vec<[f32; 7]>>,
    #[pyo3(get)]
    speculative_survival: Option<Vec<[f32; 7]>>,
    #[pyo3(get)]
    records: Vec<Vec<i64>>,
    #[pyo3(get)]
    logprobs: Option<Vec<Vec<Option<(u32, f32, Vec<(u32, f32)>)>>>>,
    #[pyo3(get)]
    gpu_ns: [Option<u64>; 16],
    #[pyo3(get)]
    readiness_wait_ns: Option<u64>,
    #[pyo3(get)]
    wall_ns: u64,
    #[pyo3(get)]
    queued: bool,
}

#[pyclass(name = "GraphExecutor", module = "flare_heliopause")]
struct PreparedGraphExecutor {
    worker: ExecutionWorker,
    owner: Mutex<OwnerLease<Py<PyAny>>>,
    drained: std::sync::atomic::AtomicBool,
}

#[pymethods]
impl PreparedGraphExecutor {
    #[new]
    #[pyo3(signature = (owner, manifest, *, peer_fds=None))]
    fn new(
        py: Python<'_>,
        owner: Py<PyAny>,
        manifest: &Bound<'_, PyAny>,
        peer_fds: Option<Vec<i32>>,
    ) -> PyResult<Self> {
        if owner.bind(py).is_none() {
            return Err(PyValueError::new_err(
                "owner must retain prepared execution resources",
            ));
        }
        let manifest = execution_manifest(manifest)?;
        manifest
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let peers = peer_fds
            .map(|fds| {
                fds.into_iter()
                    .map(rank_socket)
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        let worker = py
            .detach(|| {
                // SAFETY: Preparation's owner retains the validated buffers,
                // executable graphs, and context with exclusive native access.
                // Startup holds owner locally; OwnerLease releases it only after
                // close confirms both streams drained, including failed submits.
                match peers {
                    Some(peers) => unsafe { ExecutionWorker::new_parallel(manifest, peers) },
                    None => unsafe { ExecutionWorker::new(manifest) },
                }
            })
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self {
            worker,
            owner: Mutex::new(OwnerLease(Some(owner))),
            drained: std::sync::atomic::AtomicBool::new(false),
        })
    }

    #[pyo3(signature = (graph_index, uploads, grammar_constrained=false))]
    fn submit(
        &self,
        py: Python<'_>,
        graph_index: usize,
        uploads: Vec<(String, usize, Vec<u8>)>,
        grammar_constrained: bool,
    ) -> PyResult<Option<u64>> {
        let uploads = uploads
            .into_iter()
            .map(|(buffer, offset, bytes)| OwnedUpload {
                buffer,
                offset,
                bytes,
            })
            .collect();
        py.detach(|| {
            self.worker
                .submit(graph_index, uploads, grammar_constrained)
        })
        .map(|fence| fence.map(|fence| fence.0))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn poll(&self, py: Python<'_>) -> PyResult<Option<ExecutionCompletion>> {
        py.detach(|| self.worker.poll())
            .map(|completion| {
                completion.map(|completion| ExecutionCompletion {
                    fence: completion.fence.0,
                    graph_index: completion.graph_index,
                    logprobs: completion.logprobs.map(|slots| slots.into_iter().map(|positions|
                        positions.into_iter().map(|entry| entry.map(|entry|
                            (entry.token, entry.logprob, entry.top))).collect()).collect()),
                    proposal_confidence: completion.proposal_confidence
                        .map(|rows| rows[..completion.records.len()].to_vec()),
                    speculative_survival: completion.speculative_survival
                        .map(|rows| rows[..completion.records.len()].to_vec()),
                    gpu_ns: completion.timing.gpu_ns,
                    readiness_wait_ns: completion.timing.readiness.map(|sample| sample.wait_ns()),
                    wall_ns: completion.timing.wall_ns,
                    queued: completion.timing.queued,
                    records: completion
                        .records
                        .into_iter()
                        .map(|record| record.fields())
                        .collect(),
                })
            })
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    #[getter]
    fn outstanding(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| self.worker.outstanding())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }
    #[getter]
    fn drained(&self) -> bool {
        self.drained.load(std::sync::atomic::Ordering::Acquire)
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let shutdown = py.detach(|| self.worker.close());
        if shutdown.drained {
            self.drained
                .store(true, std::sync::atomic::Ordering::Release);
            let owner = self
                .owner
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .take_after_drain();
            drop(owner);
        }
        match shutdown.error {
            Some(error) => Err(PyRuntimeError::new_err(error.to_string())),
            None => Ok(()),
        }
    }
}

#[cfg(target_os = "linux")]
fn rank_socket(fd: i32) -> PyResult<crate::execution::RankStream> {
    if fd < 0 {
        return Err(PyValueError::new_err(
            "rank socket descriptor must be nonnegative",
        ));
    }
    // SAFETY: Python supplies a live socket descriptor and retains it during this
    // call. Duplication gives Rust an independent owner of the same connection.
    let socket = unsafe { BorrowedFd::borrow_raw(fd) }
        .try_clone_to_owned()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    crate::execution::RankStream::from_socket(socket)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[cfg(target_os = "windows")]
fn rank_socket(_fd: i32) -> PyResult<crate::execution::RankStream> {
    Err(PyRuntimeError::new_err(
        "tensor-parallel preparation requires Linux rank sockets",
    ))
}

#[pyfunction]
fn serve_tensor_parallel_rank(
    py: Python<'_>,
    owner: Py<PyAny>,
    manifest: &Bound<'_, PyAny>,
    socket_fd: i32,
    rank: usize,
    world: usize,
) -> PyResult<()> {
    if owner.bind(py).is_none() {
        return Err(PyValueError::new_err(
            "owner must retain prepared rank resources",
        ));
    }
    let manifest = execution_manifest(manifest)?;
    manifest
        .validate()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let socket = rank_socket(socket_fd)?;
    let mut owner = OwnerLease(Some(owner));
    // SAFETY: The lease retains every graph and IPC mapping until the native
    // coordinator confirms that all ranks drained, or intentionally retains them.
    let shutdown =
        py.detach(|| unsafe { crate::execution::serve_rank(manifest, socket, rank, world) });
    if shutdown.drained {
        drop(owner.take_after_drain());
    }
    match shutdown.error {
        Some(error) => {
            let error = PyRuntimeError::new_err(error.to_string());
            error
                .value(py)
                .setattr("resources_retained", !shutdown.drained)?;
            Err(error)
        }
        None => Ok(()),
    }
}

/// Monitor owned child ranks in Rust; unexpected peer exit terminates this process.
#[cfg(target_os = "linux")]
#[pyclass(name = "RankSupervisor", module = "flare_heliopause")]
struct NativeRankSupervisor {
    supervisor: Mutex<crate::processes::RankSupervisor>,
}

#[cfg(target_os = "linux")]
#[pymethods]
impl NativeRankSupervisor {
    #[new]
    fn new(pids: Vec<i32>) -> PyResult<Self> {
        Ok(Self {
            supervisor: Mutex::new(
                crate::processes::RankSupervisor::new(&pids)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
        })
    }

    #[pyo3(signature = (timeout_ms=30000))]
    fn begin_shutdown(&self, timeout_ms: u64) -> PyResult<()> {
        self.supervisor
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .begin_shutdown(Duration::from_millis(timeout_ms))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn finish_shutdown(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            self.supervisor
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .finish_shutdown()
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn abort(&self, reason: &str) {
        self.supervisor
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .abort(reason)
    }
}

#[cfg(target_os = "linux")]
#[pyfunction]
fn bind_parent_lifetime(parent_pid: i32) -> PyResult<()> {
    if parent_pid <= 0 {
        return Err(PyValueError::new_err("parent PID must be positive"));
    }
    crate::processes::bind_parent_lifetime(parent_pid)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyclass(name = "CliSignals", module = "flare_heliopause", unsendable)]
struct NativeCliSignals {
    signals: Option<crate::processes::CliSignals>,
}

#[pymethods]
impl NativeCliSignals {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            signals: Some(
                crate::processes::CliSignals::new()
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
            ),
        })
    }
    fn __enter__(&self) {}
    fn __exit__(&mut self, _kind: Py<PyAny>, _value: Py<PyAny>, _traceback: Py<PyAny>) {
        self.signals.take();
    }
}

impl Drop for PreparedGraphExecutor {
    fn drop(&mut self) {
        let shutdown = self.worker.close();
        if shutdown.drained {
            let owner = self
                .owner
                .get_mut()
                .unwrap_or_else(|error| error.into_inner())
                .take_after_drain();
            drop(owner);
        }
    }
}

#[pymodule]
fn flare_heliopause(module: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::export::binding::register(module)?;
    #[cfg(feature = "qualification")]
    {
        module.add_class::<GraphReplayWorker>()?;
        module.add_class::<ReplayCompletion>()?;
    }
    module.add_class::<PreparedGraphExecutor>()?;
    module.add_class::<crate::embedding_tables::NativeEmbeddingTables>()?;
    module.add_class::<crate::embedding_tables::NativeEmbeddingBinding>()?;
    module.add_class::<ExecutionCompletion>()?;
    #[cfg(target_os = "linux")]
    module.add_class::<NativeRankSupervisor>()?;
    module.add_class::<NativeCliSignals>()?;
    #[cfg(all(feature = "server", feature = "qualification"))]
    module.add_class::<NativeServer>()?;
    module.add_function(wrap_pyfunction!(current_context, module)?)?;
    module.add_function(wrap_pyfunction!(memory_inventory, module)?)?;
    module.add_function(wrap_pyfunction!(serve_tensor_parallel_rank, module)?)?;
    #[cfg(target_os = "linux")]
    module.add_function(wrap_pyfunction!(bind_parent_lifetime, module)?)?;
    Ok(())
}

/// Snapshot device-wide usage and the package allocator's physical mappings.
///
/// Call after preparation has materialized scratch. Imported peer mappings,
/// dynamically backed caches, and allocations outside this allocator are not
/// included in `mappings`. Device-wide usage includes other CUDA processes.
#[pyfunction]
fn memory_inventory(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let inventory = crate::allocator::memory_inventory()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let result = pyo3::types::PyDict::new(py);
    result.set_item("device", inventory.device)?;
    result.set_item("free_bytes", inventory.free)?;
    result.set_item("total_bytes", inventory.total)?;
    result.set_item("mappings", inventory.mappings)?;
    Ok(result)
}

#[cfg(test)]
mod binding_tests {
    use super::*;

    #[test]
    fn raw_replay_exports_require_qualification_feature() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "flare_heliopause").unwrap();
            flare_heliopause(&module).unwrap();
            for name in ["GraphReplayWorker", "ReplayCompletion"] {
                assert_eq!(
                    module.hasattr(name).unwrap(),
                    cfg!(feature = "qualification")
                );
            }
            assert!(module.hasattr("GraphExecutor").unwrap());
            assert!(module.hasattr("current_context").unwrap());
            assert_eq!(
                module.hasattr("Server").unwrap(),
                cfg!(all(feature = "server", feature = "qualification"))
            );
        });
    }
}

#[cfg(all(feature = "server", feature = "qualification"))]
#[pyclass(name = "Server", module = "flare_heliopause")]
struct NativeServer {
    server: crate::server::Server,
    owner: Mutex<OwnerLease<Py<PyAny>>>,
    drained: std::sync::atomic::AtomicBool,
}
#[cfg(all(feature = "server", feature = "qualification"))]
#[pymethods]
impl NativeServer {
    #[new]
    #[pyo3(signature = (owner, manifest, config_json, *, peer_fds=None))]
    fn new(
        py: Python<'_>,
        owner: Py<PyAny>,
        manifest: &Bound<'_, PyAny>,
        config_json: &str,
        peer_fds: Option<Vec<i32>>,
    ) -> PyResult<Self> {
        if owner.bind(py).is_none() {
            return Err(PyValueError::new_err(
                "owner must retain the complete prepared model",
            ));
        }
        let manifest = execution_manifest(manifest)?;
        let peers = peer_fds
            .map(|fds| {
                fds.into_iter()
                    .map(rank_socket)
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        let bootstrap = py
            .detach(|| crate::server::Bootstrap::load(config_json, &manifest))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let server = match py.detach(|| {
            // SAFETY: The prepared owner retains every captured allocation and
            // handle and relinquishes external mutation during native execution.
            // It stays live across startup, is retained on an undrained failure,
            // and is otherwise transferred into OwnerLease until confirmed drain.
            unsafe {
                crate::server::Server::start(
                    manifest,
                    bootstrap.model,
                    bootstrap.frontend,
                    bootstrap.images,
                    bootstrap.server,
                    peers.map(crate::execution::RankPeers::Processes),
                )
            }
        }) {
            Ok(server) => server,
            Err(error) => {
                if !error.drained {
                    std::mem::forget(owner);
                }
                let exception = PyRuntimeError::new_err(error.message.to_string());
                exception
                    .value(py)
                    .setattr("resources_retained", !error.drained)?;
                return Err(exception);
            }
        };
        Ok(Self {
            server,
            owner: Mutex::new(OwnerLease(Some(owner))),
            drained: std::sync::atomic::AtomicBool::new(false),
        })
    }
    #[getter]
    fn address(&self) -> String {
        self.server.address().to_string()
    }
    #[getter]
    fn healthy(&self) -> bool {
        self.server.healthy()
    }
    fn start_dashboard(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| self.server.start_dashboard())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }
    #[pyo3(signature = (timeout_ms=1000))]
    fn wait(&self, py: Python<'_>, timeout_ms: u64) -> PyResult<bool> {
        if timeout_ms > 60_000 {
            return Err(PyValueError::new_err(
                "wait timeout must not exceed 60000 ms",
            ));
        }
        let healthy = py.detach(|| self.server.wait(Duration::from_millis(timeout_ms)));
        py.check_signals()?;
        Ok(healthy)
    }
    #[getter]
    fn drained(&self) -> bool {
        self.drained.load(std::sync::atomic::Ordering::Acquire)
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let shutdown = py.detach(|| self.server.close());
        if shutdown.drained {
            self.drained
                .store(true, std::sync::atomic::Ordering::Release);
            let owner = self
                .owner
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take_after_drain();
            drop(owner);
        }
        shutdown.error.map_or(Ok(()), |error| {
            Err(PyRuntimeError::new_err(error.to_string()))
        })
    }
}
#[cfg(all(feature = "server", feature = "qualification"))]
impl Drop for NativeServer {
    fn drop(&mut self) {
        let shutdown = self.server.close();
        if shutdown.drained {
            let owner = self
                .owner
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .take_after_drain();
            drop(owner);
        }
    }
}
