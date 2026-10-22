//! Wire adapters. Model execution consumes `inference::Request` only.

mod decode;
pub(in crate::server) mod decisions;
mod encode;
use super::{inference, observation::ObservationOptions};
pub(super) use decode::decode;
pub(super) use encode::{Encoder, Frame, PromptUsage};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Protocol {
    Chat,
    Completions,
    Messages,
    Responses,
}
impl Protocol {
    pub fn from_path(path: &str) -> Self {
        match path {
            "/v1/messages" | "/v1/messages/count_tokens" => Self::Messages,
            "/v1/responses" => Self::Responses,
            "/v1/completions" => Self::Completions,
            _ => Self::Chat,
        }
    }
    pub fn error(self, status: axum::http::StatusCode, message: impl std::fmt::Display) -> Value {
        let kind = match status.as_u16() {
            401 => "authentication_error",
            403 => "permission_error",
            404 => "not_found_error",
            413 => "request_too_large",
            429 => "rate_limit_error",
            500..=599 if self == Self::Messages => "api_error",
            500..=599 => "server_error",
            _ => "invalid_request_error",
        };
        let mut body = serde_json::json!({"error":{"type":kind,"message":message.to_string()}});
        if self == Self::Messages {
            body["type"] = "error".into();
        } else {
            body["error"]["param"] = Value::Null;
            body["error"]["code"] = kind.into();
        }
        body
    }
    pub fn failure(self, failure: &crate::Error) -> Value {
        let mut body = self.error(super::status(failure), failure);
        if self != Self::Messages {
            body["error"]["code"] = super::error_code(failure).into();
        }
        body
    }
}
pub(super) struct Decoded {
    pub request: inference::Request,
    pub stream: bool,
    pub echo: bool,
    pub observation: ObservationOptions,
    pub metadata: Value,
}

#[cfg(test)]
mod tests;
