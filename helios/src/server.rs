//! Native HTTP admission, bounded generation streams and direct graph scheduling.

mod config;
mod dashboard;
mod http;
mod inference;
mod listener;
mod metrics;
mod models;
mod observation;
pub(crate) mod output;
mod logprobs;
mod protocol;
mod request;
pub use request::{PreparedRequest, prepare as prepare_request};
mod runtime;
mod stops;
pub(crate) mod token_spans;
pub use config::Bootstrap;
pub use dashboard::view;
pub use metrics::Snapshot;
pub use runtime::{Generation, GenerationEvent, ModelRuntime, Prompt};

pub fn restore_terminal_for_exit() {
    dashboard::restore_for_exit();
}

use crate::{
    AdmissionError, Error, ExecutionError, Failure, Result, Shutdown,
    execution::ExecutionManifest,
    frontend::{Frontend, images::ImageProcessor},
    model::ModelConfig,
};
use axum::{
    Extension, Json, Router,
    extract::{DefaultBodyLimit, Request, State},
    http::{HeaderMap, StatusCode, Uri, header},
    middleware::{self, Next},
    response::{IntoResponse, Response, Sse, sse::KeepAlive},
    routing::{get, post},
};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    convert::Infallible,
    net::SocketAddr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

pub struct ServerConfig {
    pub bind: SocketAddr,
    pub queue_capacity: usize,
    pub event_capacity: usize,
    pub frontend_workers: usize,
    pub max_body_bytes: usize,
    /// Serving key; Messages also accepts `x-api-key`. `/health` and `/metrics`
    /// are unauthenticated.
    pub api_key: Option<String>,
    /// Names requests may use for the model in addition to its identifier.
    pub served_names: Vec<String>,
    /// Setting that raises the per-request context, appended to context rejections.
    pub context_hint: String,
    /// Setting that raises request or page capacity, appended to capacity rejections.
    pub capacity_hint: String,
    pub scheduling_policy: crate::scheduler::PolicyConfig,
}
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8000".parse().unwrap(),
            queue_capacity: 16,
            event_capacity: 128,
            frontend_workers: 2,
            max_body_bytes: 32 * 1024 * 1024,
            api_key: None,
            served_names: Vec::new(),
            context_hint: String::new(),
            capacity_hint: String::new(),
            scheduling_policy: crate::scheduler::PolicyConfig::default(),
        }
    }
}
const GENERATED_WINDOW: Duration = Duration::from_secs(60);
const GENERATED_SAMPLE_PERIOD: Duration = Duration::from_secs(5);
struct HttpState {
    model: Arc<ModelRuntime>,
    frontend: Arc<Frontend>,
    images: Option<Arc<ImageProcessor>>,
    grammar: Arc<crate::grammar::GrammarCompiler>,
    frontend_slots: Arc<Semaphore>,
    ingress_slots: Arc<Semaphore>,
    max_body_bytes: usize,
    frontend_capacity: usize,
    ingress_capacity: usize,
    ids: AtomicU64,
    api_key: Option<String>,
    context_hint: String,
    capacity_hint: String,
    started: Instant,
    generated: Mutex<VecDeque<(Instant, u64)>>,
}
impl HttpState {
    /// Records the generated-token counter, retaining one sample at or before the window start.
    fn sample_generated(&self) {
        let now = Instant::now();
        let tokens = self.model.metrics().snapshot(false).generated_tokens;
        let mut samples = self.generated.lock().unwrap_or_else(|e| e.into_inner());
        samples.push_back((now, tokens));
        while samples.len() > 2 && now.duration_since(samples[1].0) >= GENERATED_WINDOW {
            samples.pop_front();
        }
    }

    fn generated_last_minute(&self) -> u64 {
        let samples = self.generated.lock().unwrap_or_else(|e| e.into_inner());
        match (samples.front(), samples.back()) {
            (Some(first), Some(last)) => last.1.saturating_sub(first.1),
            _ => 0,
        }
    }
}

pub struct Server {
    address: SocketAddr,
    model: Arc<ModelRuntime>,
    control: Mutex<HttpControl>,
    shutdown: Mutex<Option<Shutdown>>,
    dashboard: Mutex<Option<dashboard::Dashboard>>,
}
struct HttpControl {
    stop: Option<oneshot::Sender<()>>,
    thread: Option<JoinHandle<Result<()>>>,
}

pub struct StartFailure {
    pub message: Error,
    pub drained: bool,
}
impl From<Error> for StartFailure {
    fn from(message: Error) -> Self {
        Self {
            message,
            drained: true,
        }
    }
}
impl From<&str> for StartFailure {
    fn from(message: &str) -> Self {
        Error::from(message).into()
    }
}
impl Server {
    /// # Safety
    /// The caller owns stable, exclusive prepared resources until close confirms
    /// both CUDA streams drained. No Python code runs on either native worker.
    pub unsafe fn start(
        manifest: ExecutionManifest,
        model: ModelConfig,
        frontend: Frontend,
        images: Option<ImageProcessor>,
        config: ServerConfig,
        peers: Option<crate::execution::RankPeers>,
    ) -> std::result::Result<Self, StartFailure> {
        if model.contract.vision.enabled() != images.is_some() {
            return Err("image processor must match prepared vision capabilities".into());
        }
        if !(1..=64).contains(&config.frontend_workers)
            || config.max_body_bytes == 0
            || config.max_body_bytes > 256 * 1024 * 1024
        {
            return Err("invalid HTTP admission limits".into());
        }
        let grammar = Arc::new(crate::grammar::GrammarCompiler::new(
            frontend.tokenizer(),
            model.contract.vocabulary,
            &model.eos_token_ids,
        )?);
        let ingress_capacity = config
            .queue_capacity
            .checked_add(model.limits.max_active)
            .ok_or("HTTP request capacity overflows")?;
        let frontend = Arc::new(frontend);
        // SAFETY: The caller grants exclusive graph/storage access and retains
        // their owners until drain. This server propagates that obligation to the
        // model worker and reports uncertain drain through StartFailure or close.
        let model = Arc::new(unsafe {
            ModelRuntime::start(
                manifest,
                model,
                config.queue_capacity,
                config.event_capacity,
                frontend.clone(),
                peers,
                config.scheduling_policy,
            )?
        });
        let state = Arc::new(HttpState {
            model: model.clone(),
            frontend,
            images: images.map(Arc::new),
            grammar,
            frontend_slots: Arc::new(Semaphore::new(config.frontend_workers)),
            ingress_slots: Arc::new(Semaphore::new(ingress_capacity)),
            max_body_bytes: config.max_body_bytes,
            frontend_capacity: config.frontend_workers,
            ingress_capacity,
            ids: AtomicU64::new(1),
            api_key: config.api_key.clone(),
            context_hint: config.context_hint.clone(),
            capacity_hint: config.capacity_hint.clone(),
            started: Instant::now(),
            generated: Mutex::new(VecDeque::new()),
        });
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let runtime_model = model.clone();
        let thread = match thread::Builder::new()
            .name("flare-http".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .map_err(|error| Error::Execution(ExecutionError::Worker(Failure::caused_by(format!("cannot create HTTP runtime: {error}"), error))))?;
                runtime.block_on(async move {
                    let shutdown_state = state.clone();
                    let sampler = state.clone();
                    tokio::spawn(async move {
                        let mut ticks = tokio::time::interval(GENERATED_SAMPLE_PERIOD);
                        loop {
                            ticks.tick().await;
                            sampler.sample_generated();
                        }
                    });
                    let router = router(state, &config.served_names, config.max_body_bytes);
                    let listener = match tokio::net::TcpListener::bind(config.bind).await {
                        Ok(listener) => listener,
                        Err(error) => {
                            let error = Error::Execution(ExecutionError::Worker(Failure::caused_by(format!("cannot bind HTTP listener: {error}"), error)));
                            let _ = ready_tx.send(Err(error.clone()));
                            return Err(error);
                        }
                    };
                    let address = listener.local_addr().map_err(|error| Error::Execution(ExecutionError::Worker(Failure::caused_by(error.to_string(), error))))?;
                    let listener = listener::HttpListener::new(listener);
                    let _ = ready_tx.send(Ok(address));
                    let (drain_tx, drain_rx) = oneshot::channel();
                    let mut serving = tokio::spawn(async move {
                        axum::serve(listener, router)
                            .with_graceful_shutdown(async {
                                let _ = drain_rx.await;
                            })
                            .await
                    });
                    tokio::select! {
                        result=&mut serving=>{runtime_model.request_stop();return result.map_err(|error| Error::Execution(ExecutionError::Worker(Failure::caused_by(error.to_string(), error))))?.map_err(|error| Error::Execution(ExecutionError::Worker(Failure::caused_by(error.to_string(), error))));},
                        _=shutdown_rx=>{}
                    }
                    shutdown_state.frontend_slots.close();
                    shutdown_state.ingress_slots.close();
                    let _ = drain_tx.send(());
                    match tokio::time::timeout(Duration::from_secs(5), &mut serving).await {
                        Ok(result) => result
                            .map_err(|error| Error::Execution(ExecutionError::Worker(Failure::caused_by(error.to_string(), error))))?
                            .map_err(|error| Error::Execution(ExecutionError::Worker(Failure::caused_by(error.to_string(), error)))),
                        Err(_) => {
                            serving.abort();
                            let _ = serving.await;
                            Ok(())
                        }
                    }
                })
            }) {
            Ok(thread) => thread,
            Err(error) => {
                let shutdown = model.close();
                return Err(StartFailure {
                    message: Error::Execution(ExecutionError::Worker(Failure::caused_by(format!("cannot start HTTP thread: {error}"), error))),
                    drained: shutdown.drained,
                });
            }
        };
        match ready_rx.recv() {
            Ok(Ok(address)) => Ok(Self {
                address,
                model,
                control: Mutex::new(HttpControl {
                    stop: Some(shutdown_tx),
                    thread: Some(thread),
                }),
                shutdown: Mutex::new(None),
                dashboard: Mutex::new(None),
            }),
            result => {
                let returned = thread.join().ok().and_then(Result::err);
                let shutdown = model.close();
                Err(StartFailure {
                    message: match result {
                        Ok(Err(error)) => error,
                        Err(disconnected) => readiness_failure(
                            "HTTP thread exited before readiness",
                            disconnected,
                            returned,
                        ),
                        Ok(Ok(_)) => unreachable!("ready success handled above"),
                    },
                    drained: shutdown.drained,
                })
            }
        }
    }
    pub fn address(&self) -> SocketAddr {
        self.address
    }
    pub fn healthy(&self) -> bool {
        self.model.healthy()
    }
    pub fn wait(&self, timeout: Duration) -> bool {
        self.model.wait(timeout)
    }
    pub fn start_dashboard(&self) -> std::io::Result<bool> {
        let _control = self.control.lock().unwrap_or_else(|e| e.into_inner());
        let mut dashboard = self.dashboard.lock().unwrap_or_else(|e| e.into_inner());
        if dashboard.is_none() && self.healthy() {
            *dashboard = dashboard::Dashboard::start(
                self.model.metrics_handle(),
                format!("{} · {}", self.model.config().model_id, self.address),
                self.model.config().has_learned_drafts(),
            )?;
        }
        Ok(dashboard.is_some())
    }
    pub fn stop_dashboard(&self) {
        self.dashboard
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
    }
    pub fn close(&self) -> Shutdown {
        let mut control = self.control.lock().unwrap_or_else(|e| e.into_inner());
        self.stop_dashboard();
        if let Some(shutdown) = self
            .shutdown
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
        {
            return shutdown;
        }
        if let Some(stop) = control.stop.take() {
            let _ = stop.send(());
        }
        let mut shutdown = self.model.close();
        if let Some(thread) = control.thread.take() {
            match thread.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    shutdown.error.get_or_insert(error);
                }
                Err(_) => {
                    shutdown.error.get_or_insert(Error::execution(
                        ExecutionError::Worker,
                        "HTTP thread panicked",
                    ));
                }
            }
        }
        *self.shutdown.lock().unwrap_or_else(|e| e.into_inner()) = Some(shutdown.clone());
        shutdown
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.close();
    }
}

/// Routes generation and model discovery behind API-key authentication; health
/// and metrics expose no request content and stay unauthenticated.
fn router(state: Arc<HttpState>, served_names: &[String], max_body_bytes: usize) -> Router {
    Router::new()
        .merge(models::router(state.model.config(), served_names))
        .route("/v1/chat/completions", post(http::generate).layer(middleware::from_fn_with_state(state.clone(), admit_http)))
        .route("/v1/completions", post(http::generate).layer(middleware::from_fn_with_state(state.clone(), admit_http)))
        .route("/v1/messages", post(http::generate).layer(middleware::from_fn_with_state(state.clone(), admit_http)))
        .route("/v1/messages/count_tokens", post(http::generate).layer(middleware::from_fn_with_state(state.clone(), admit_http)))
        .route("/v1/responses", post(http::generate).get(http::no_websocket).layer(middleware::from_fn_with_state(state.clone(), admit_http)))
        .route("/v1/systemone", post(protocol::decisions::generate).layer(middleware::from_fn_with_state(state.clone(), admit_http)))
        .route("/v1/decisions", post(protocol::decisions::generate).layer(middleware::from_fn_with_state(state.clone(), admit_http)))
        .route_layer(middleware::from_fn_with_state(state.clone(), authorize))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(state)
}

async fn health(State(state): State<Arc<HttpState>>) -> Response {
    if state.model.healthy() {
        Json(json!({"status":"ok"})).into_response()
    } else {
        error(
            StatusCode::SERVICE_UNAVAILABLE,
            "prepared model runtime is unavailable",
        )
    }
}
async fn metrics(State(state): State<Arc<HttpState>>, uri: Uri, headers: HeaderMap) -> Response {
    let accept = headers.get(header::ACCEPT).and_then(|value| value.to_str().ok());
    let format = match metrics::Format::negotiate(uri.query(), accept) {
        Ok(format) => format,
        Err(message) => return error(StatusCode::BAD_REQUEST, message),
    };
    let input_metrics = metrics::input_metrics(
        state.model.input_store().map(|store| store.stats()).unwrap_or_default(),
        state.images.as_ref().map(|processor| processor.preparation_counts()).unwrap_or_default(),
    );
    if format == metrics::Format::Json {
        let learned_drafter = state.model.config().has_learned_drafts();
        let snapshot = state.model.metrics().snapshot(learned_drafter);
        let mut value = snapshot.json(learned_drafter);
        if let Some(object) = value.as_object_mut() {
            for (name, _, _, count) in input_metrics {
                object.insert(name.into(), count.into());
            }
        }
        return ([(header::VARY, "accept")], Json(value)).into_response();
    }
    let mut text = state.model.metrics().render();
    for (name, kind, help, value) in input_metrics {
        metrics::metric(&mut text, name, kind, help, value);
    }
    for (name, help, value) in [
        (
            "http_inflight_requests",
            "Reserved HTTP admission slots; a batch holds one slot per generation.",
            state.ingress_capacity - state.ingress_slots.available_permits(),
        ),
        (
            "http_request_capacity",
            "Maximum simultaneous reserved HTTP generation slots.",
            state.ingress_capacity,
        ),
        (
            "frontend_active_workers",
            "CPU workers preparing tokenizer, image and grammar inputs.",
            state.frontend_capacity - state.frontend_slots.available_permits(),
        ),
        (
            "frontend_worker_capacity",
            "Maximum simultaneous CPU preparation workers.",
            state.frontend_capacity,
        ),
        (
            "uptime_seconds",
            "Seconds since the HTTP listener started.",
            state.started.elapsed().as_secs() as usize,
        ),
        (
            "generated_tokens_last_minute",
            "Committed generated tokens during the most recent minute, sampled every five seconds.",
            state.generated_last_minute() as usize,
        ),
    ] {
        metrics::metric(&mut text, name, "gauge", help, value as u64);
    }
    (
        [
            (header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8"),
            (header::VARY, "accept"),
        ],
        text,
    )
        .into_response()
}
fn readiness_failure(
    message: &'static str,
    disconnected: std::sync::mpsc::RecvError,
    returned: Option<Error>,
) -> Error {
    let cause = match returned {
        Some(error) => Failure::caused_by(message, error),
        None => Failure::caused_by(message, disconnected),
    };
    ExecutionError::Worker(cause).into()
}

fn status(failure: &Error) -> StatusCode {
    match failure {
        Error::Admission(AdmissionError::Capacity(_) | AdmissionError::Unavailable(_)) => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        Error::Admission(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn error_code(failure: &Error) -> &'static str {
    match failure {
        Error::Admission(AdmissionError::Capacity(_)) => "capacity_exhausted",
        Error::Admission(AdmissionError::Unavailable(_)) => "server_unavailable",
        Error::Admission(AdmissionError::Context(_)) => "context_length_exceeded",
        Error::Admission(AdmissionError::Feature(_)) => "unsupported_feature",
        Error::Admission(AdmissionError::Grammar(_)) => "invalid_grammar",
        Error::Admission(AdmissionError::Image(_)) => "invalid_image",
        Error::Admission(AdmissionError::Invalid(_)) => "invalid_request",
        Error::Execution(ExecutionError::Capacity(_)) => "execution_capacity_exceeded",
        _ => "server_error",
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .fold(0u8, |difference, (a, b)| difference | (a ^ b))
            == 0
}

async fn authorize(State(state): State<Arc<HttpState>>, request: Request, next: Next) -> Response {
    let protocol = protocol::Protocol::from_path(request.uri().path());
    if let Some(key) = &state.api_key {
        let authorization = request.headers().get(axum::http::header::AUTHORIZATION);
        let bearer = authorization.and_then(|value| value.as_bytes().strip_prefix(b"Bearer "));
        let anthropic = (protocol == protocol::Protocol::Messages)
            .then(|| request.headers().get("x-api-key"))
            .flatten()
            .map(|v| v.as_bytes());
        let valid = (authorization.is_none() || bearer.is_some())
            && bearer
                .or(anthropic)
                .is_some_and(|v| constant_time_eq(v, key.as_bytes()))
            && bearer
                .into_iter()
                .chain(anthropic)
                .all(|v| constant_time_eq(v, key.as_bytes()));
        if !valid {
            return http::error(
                protocol,
                StatusCode::UNAUTHORIZED,
                "missing or invalid API key",
            );
        }
    }
    next.run(request).await
}

fn error(status: StatusCode, message: impl std::fmt::Display) -> Response {
    (
        status,
        Json(json!({"error":{"message":message.to_string(),"type":"invalid_request_error"}})),
    )
        .into_response()
}

async fn admit_http(
    State(state): State<Arc<HttpState>>,
    mut request: Request,
    next: Next,
) -> Response {
    let protocol = protocol::Protocol::from_path(request.uri().path());
    if !state.model.healthy() {
        return http::error(
            protocol,
            StatusCode::SERVICE_UNAVAILABLE,
            "prepared model runtime is unavailable",
        );
    }
    let ingress = match state.ingress_slots.clone().try_acquire_owned() {
        Ok(permit) => Arc::new(permit),
        Err(_) => {
            let mut message = format!(
                "native HTTP admission is full: {} requests are active or queued",
                state.ingress_capacity
            );
            if !state.capacity_hint.is_empty() {
                message = format!("{message}; {}", state.capacity_hint);
            }
            return http::error(protocol, StatusCode::TOO_MANY_REQUESTS, message);
        }
    };
    request.extensions_mut().insert(ingress);
    next.run(request).await
}

fn chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish: Value,
    include_usage: bool,
) -> Value {
    let mut value = json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[{"index":0,"delta":delta,"finish_reason":finish}]});
    if include_usage {
        value["usage"] = Value::Null;
    }
    value
}

fn observed_chunk(
    id: &str,
    created: u64,
    model: &str,
    token: u32,
    timestamp_ns: Option<u64>,
    include_usage: bool,
) -> Value {
    let mut value = chunk(id, created, model, json!({}), Value::Null, include_usage);
    value["choices"][0]["token_ids"] = json!([token]);
    if let Some(timestamp) = timestamp_ns {
        value["choices"][0]["token_timestamps_ns"] = json!([timestamp]);
    }
    value
}

#[cfg(test)]
mod observation_tests {
    use super::*;

    #[test]
    fn readiness_failure_keeps_worker_cause_with_unchanged_fallback_message() {
        use std::error::Error as _;
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let joined = std::thread::spawn(move || -> Result<()> {
            drop(tx);
            Err(ExecutionError::Worker(Failure::caused_by(
                "cannot create HTTP runtime: denied",
                std::io::Error::from_raw_os_error(13),
            ))
            .into())
        });
        let disconnected = rx.recv().unwrap_err();
        let returned = joined.join().unwrap().err();
        let error = readiness_failure(
            "HTTP thread exited before readiness",
            disconnected,
            returned,
        );
        assert!(matches!(error, Error::Execution(ExecutionError::Worker(_))));
        assert_eq!(error.to_string(), "HTTP thread exited before readiness");
        let worker = error.source().unwrap().downcast_ref::<Error>().unwrap();
        assert_eq!(
            worker
                .source()
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .raw_os_error(),
            Some(13)
        );
    }

    #[tokio::test]
    async fn typed_status_and_wire_messages_preserve_admission_and_execution_boundaries() {
        let cases = [
            (
                Error::admission(AdmissionError::Invalid, "invalid sampling parameters"),
                StatusCode::BAD_REQUEST,
            ),
            (
                Error::admission(
                    AdmissionError::Feature,
                    "parallel_tool_calls=false is not implemented",
                ),
                StatusCode::BAD_REQUEST,
            ),
            (
                Error::admission(AdmissionError::Grammar, "invalid grammar"),
                StatusCode::BAD_REQUEST,
            ),
            (
                Error::admission(AdmissionError::Image, "invalid image base64"),
                StatusCode::BAD_REQUEST,
            ),
            (
                Error::admission(AdmissionError::Capacity, "native request queue is full"),
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                Error::admission(AdmissionError::Context, "prompt exceeds the context"),
                StatusCode::BAD_REQUEST,
            ),
            (
                Error::admission(
                    AdmissionError::Unavailable,
                    "prepared model runtime is unavailable",
                ),
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                Error::execution(ExecutionError::Poisoned, "executor failed"),
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
        ];
        for (failure, status) in cases {
            let message = failure.to_string();
            let response = error(super::status(&failure), failure);
            assert_eq!(response.status(), status);
            let bytes = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["error"]["message"], message);
        }
    }

    #[test]
    fn raw_metadata_chunks_never_duplicate_text_or_hide_eos() {
        let ordinary = chunk(
            "x",
            1,
            "model",
            json!({"content":"answer"}),
            Value::Null,
            false,
        );
        assert!(ordinary.get("usage").is_none());
        assert!(ordinary["choices"][0].get("token_ids").is_none());
        let observed = observed_chunk("x", 1, "model", 248046, Some(42), true);
        assert_eq!(observed["choices"][0]["delta"], json!({}));
        assert_eq!(observed["choices"][0]["token_ids"], json!([248046]));
        assert_eq!(observed["choices"][0]["token_timestamps_ns"], json!([42]));
        assert!(observed["choices"][0]["finish_reason"].is_null());
        assert!(observed["usage"].is_null());
        let ids_only = observed_chunk("x", 1, "model", 7, None, false);
        assert!(ids_only["choices"][0].get("token_timestamps_ns").is_none());
    }
}
