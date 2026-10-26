//! The typed failure of host-tier operations.

use std::{error::Error as StdError, sync::Arc};

/// A host-tier failure. Cloneable so one failure can reach every waiter, and
/// able to carry the error of a seam implementation unchanged.
#[derive(Debug, Clone)]
pub struct Error {
    message: String,
    source: Option<Arc<dyn StdError + Send + Sync>>,
    carried: bool,
}

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: None,
            carried: false,
        }
    }

    pub fn caused_by(
        message: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self {
            message: message.into(),
            source: Some(Arc::new(source)),
            carried: false,
        }
    }

    /// Carry an error produced behind a seam, so the caller on the far side can
    /// recover it exactly. The message is the carried error's own.
    pub fn carrying(error: impl StdError + Send + Sync + 'static) -> Self {
        Self {
            message: error.to_string(),
            source: Some(Arc::new(error)),
            carried: true,
        }
    }

    /// The error given to [`Error::carrying`], if this error was made that way.
    pub fn carried(&self) -> Option<&(dyn StdError + Send + Sync + 'static)> {
        self.source.as_deref().filter(|_| self.carried)
    }

    /// The cause of a failure that is not a carried error.
    pub fn cause(&self) -> Option<&Arc<dyn StdError + Send + Sync>> {
        self.source.as_ref().filter(|_| !self.carried)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn StdError + 'static))
    }
}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Self::new(message)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
