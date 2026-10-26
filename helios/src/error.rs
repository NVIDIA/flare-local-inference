//! Typed failures across admission, prepared execution and model frontends.

use std::{error::Error as StdError, sync::Arc};

/// Cloneable context retaining the original failure across worker channels.
#[derive(Debug, Clone)]
pub struct Failure {
    message: String,
    source: Option<Arc<dyn StdError + Send + Sync>>,
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl StdError for Failure {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn StdError + 'static))
    }
}

impl Failure {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: None,
        }
    }

    pub fn caused_by(
        message: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self {
            message: message.into(),
            source: Some(Arc::new(source)),
        }
    }

    /// A failure whose cause is already shared, such as one from another crate.
    pub fn shared(
        message: impl Into<String>,
        source: Option<Arc<dyn StdError + Send + Sync>>,
    ) -> Self {
        Self {
            message: message.into(),
            source,
        }
    }

    pub fn boxed(message: impl Into<String>, source: Box<dyn StdError + Send + Sync>) -> Self {
        Self {
            message: message.into(),
            source: Some(Arc::from(source)),
        }
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum AdmissionError {
    #[error(transparent)]
    Invalid(Failure),
    #[error(transparent)]
    Capacity(Failure),
    /// A request's prompt and output limit exceed the context one request may reach.
    #[error(transparent)]
    Context(Failure),
    #[error(transparent)]
    Feature(Failure),
    #[error(transparent)]
    Grammar(Failure),
    #[error(transparent)]
    Image(Failure),
    #[error(transparent)]
    Unavailable(Failure),
}

#[derive(Debug, Clone, thiserror::Error)]
pub struct CudaError {
    pub operation: String,
    pub code: i32,
    pub name: String,
    pub description: String,
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} failed ({}, {}): {}",
            self.operation, self.code, self.name, self.description
        )
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum ExecutionError {
    #[error(transparent)]
    Manifest(Failure),
    #[error(transparent)]
    Upload(Failure),
    #[error(transparent)]
    Completion(Failure),
    #[error(transparent)]
    State(Failure),
    #[error(transparent)]
    Closed(Failure),
    #[error(transparent)]
    Capacity(Failure),
    #[error(transparent)]
    Worker(Failure),
    #[error(transparent)]
    Poisoned(Failure),
    #[error(transparent)]
    Drain(Failure),
    #[error(transparent)]
    Driver(CudaError),
    #[error("prepared model reported device error {code}")]
    Device { code: i64 },
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum FrontendError {
    #[error(transparent)]
    Asset(Failure),
    #[error(transparent)]
    Template(Failure),
    #[error(transparent)]
    Tokenizer(Failure),
    #[error(transparent)]
    ImageDecode(Failure),
    #[error(transparent)]
    Input(Failure),
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Admission(#[from] AdmissionError),
    #[error(transparent)]
    Execution(#[from] ExecutionError),
    #[error(transparent)]
    Frontend(#[from] FrontendError),
}

impl Error {
    pub fn admission(kind: fn(Failure) -> AdmissionError, message: impl Into<String>) -> Self {
        kind(Failure::new(message)).into()
    }
    pub fn execution(kind: fn(Failure) -> ExecutionError, message: impl Into<String>) -> Self {
        kind(Failure::new(message)).into()
    }
    pub fn frontend(kind: FrontendError) -> Self {
        kind.into()
    }
    pub(crate) fn drain(self) -> Self {
        ExecutionError::Drain(Failure::caused_by(self.to_string(), self)).into()
    }

    pub fn admission_context(self, kind: fn(Failure) -> AdmissionError) -> Self {
        kind(Failure::caused_by(self.to_string(), self)).into()
    }
}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Self::Execution(ExecutionError::State(Failure::new(message)))
    }
}
impl From<&str> for Error {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}

/// Host-tier failures are state failures with the same message and cause, except
/// that an error carried across a seam returns as it left.
impl From<flare_selene::Error> for Error {
    fn from(error: flare_selene::Error) -> Self {
        if let Some(carried) = error.carried()
            && let Some(original) = carried.downcast_ref::<Error>()
        {
            return original.clone();
        }
        Self::Execution(ExecutionError::State(Failure::shared(
            error.to_string(),
            error.cause().cloned(),
        )))
    }
}

impl From<Error> for flare_selene::Error {
    fn from(error: Error) -> Self {
        Self::carrying(error)
    }
}

#[cfg(feature = "server")]
impl serde::Serialize for Error {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cloned_context_preserves_io_cause() {
        let error = Error::Admission(AdmissionError::Image(Failure::caused_by(
            "invalid image",
            std::io::Error::from_raw_os_error(2),
        )));
        let copy = error.clone();
        assert_eq!(copy.to_string(), "invalid image");
        let cause = copy.source().unwrap();
        assert_eq!(
            cause
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .raw_os_error(),
            Some(2)
        );
    }
    #[test]
    fn failed_drain_preserves_driver_code_and_wire_message() {
        let driver = Error::Execution(ExecutionError::Driver(CudaError {
            operation: "cuStreamSynchronize".into(),
            code: 700,
            name: "CUDA_ERROR_ILLEGAL_ADDRESS".into(),
            description: "illegal address".into(),
        }));
        let message = driver.to_string();
        let error = driver.drain();
        assert!(matches!(error, Error::Execution(ExecutionError::Drain(_))));
        assert_eq!(error.to_string(), message);
        assert!(matches!(
            error.source().unwrap().downcast_ref::<Error>().unwrap(),
            Error::Execution(ExecutionError::Driver(CudaError { code: 700, .. }))
        ));
    }

    #[test]
    fn errors_cross_the_host_tier_seam_unchanged() {
        let driver = Error::Execution(ExecutionError::Driver(CudaError {
            operation: "cuMemHostRegister_v2".into(),
            code: 700,
            name: "CUDA_ERROR_ILLEGAL_ADDRESS".into(),
            description: "illegal address".into(),
        }));
        let message = driver.to_string();
        let crossed = flare_selene::Error::from(driver).clone();
        assert_eq!(crossed.to_string(), message);
        let returned = Error::from(crossed);
        assert_eq!(returned.to_string(), message);
        assert!(matches!(
            returned,
            Error::Execution(ExecutionError::Driver(CudaError { code: 700, .. }))
        ));
        let drained = Error::from(flare_selene::Error::from(Error::Execution(
            ExecutionError::Drain(Failure::new("copy did not drain")),
        )));
        assert!(matches!(
            drained,
            Error::Execution(ExecutionError::Drain(_))
        ));
        assert_eq!(drained.to_string(), "copy did not drain");
    }

    #[test]
    fn selene_failures_keep_message_and_cause() {
        let cause = std::io::Error::from_raw_os_error(2);
        let failure = flare_selene::Error::caused_by("open record", cause);
        let converted = Error::from(failure.clone());
        assert!(matches!(
            &converted,
            Error::Execution(ExecutionError::State(_))
        ));
        assert_eq!(converted.to_string(), "open record");
        assert_eq!(
            converted
                .source()
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .raw_os_error(),
            Some(2)
        );
        assert_eq!(
            Error::from(flare_selene::Error::new("plain")).to_string(),
            "plain"
        );
    }

    #[test]
    fn driver_code_survives_poison_context() {
        let driver = Error::Execution(ExecutionError::Driver(CudaError {
            operation: "cuGraphLaunch".into(),
            code: 700,
            name: "CUDA_ERROR_ILLEGAL_ADDRESS".into(),
            description: "an illegal memory access was encountered".into(),
        }));
        let poison = Error::Execution(ExecutionError::Poisoned(Failure::caused_by(
            format!("executor failed: {driver}"),
            driver,
        )));
        let cause = poison.source().unwrap().downcast_ref::<Error>().unwrap();
        assert!(matches!(
            cause,
            Error::Execution(ExecutionError::Driver(CudaError { code: 700, .. }))
        ));
    }
}
