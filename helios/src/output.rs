//! Reusable output syntax and transport-neutral incremental events.

#[cfg(feature = "server")]
mod argument_tags;
#[cfg(feature = "server")]
mod gemma_call;
#[cfg(feature = "server")]
mod minimax_xml;
#[cfg(feature = "server")]
pub mod chatml;
#[cfg(feature = "server")]
mod pipeline;
pub mod tagged;
#[cfg(feature = "server")]
mod xtml;
#[cfg(feature = "server")]
pub use pipeline::{OutputConfig, OutputPipeline};

use crate::Result;

/// Presented token IDs and stable decoded deltas are independent ordered
/// inputs. A token may produce no text or complete UTF-8 bytes emitted by
/// preceding tokens. Runtime stop filtering precedes this interface; raw
/// generation observations retain IDs suppressed by that filtering.
#[derive(Clone, Copy, Debug)]
pub enum Input<'a> {
    Token(u32),
    Text(&'a str),
}

/// Ordered presentation input after worker-owned string-stop filtering.
/// Token IDs accompany only fully visible spans; clipped spans carry text.
#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    #[cfg(feature = "server")]
    Logprob(crate::server::token_spans::ReturnedLogprob),
    Token(u32),
    Text(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Termination {
    Eos,
    OutputLimit,
    UserStop,
    Cancelled,
    ExecutionError,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    #[cfg(feature = "server")]
    TokenLogprobs(Vec<crate::server::token_spans::ReturnedLogprob>),
    #[cfg(feature = "server")]
    ContentLogprobs(Vec<crate::server::token_spans::ReturnedLogprob>),
    Content(String),
    Reasoning(String),
    ToolBegin {
        index: usize,
        name: String,
        id: Option<String>,
    },
    ToolArguments {
        index: usize,
        text: String,
    },
    ToolEnd {
        index: usize,
    },
}

/// An active protocol frame may end within a decoded span. Its composer owns
/// the remainder and decides which parser receives it.
#[derive(Default)]
pub struct Progress {
    pub events: Vec<Event>,
    pub remainder: Option<String>,
}

pub trait ReasoningParser {
    fn push(&mut self, input: Input<'_>) -> Result<Progress>;
    fn finish(&mut self, cause: Termination) -> Result<Vec<Event>>;
}

pub trait ToolCallParser {
    fn push(&mut self, input: Input<'_>) -> Result<Progress>;
    fn finish(&mut self, cause: Termination) -> Result<Vec<Event>>;
}

/// Concrete associated types keep model selection outside the token loop.
pub trait ModelOutput {
    /// One span per Content event from the latest push/finish, in event order.
    /// Ranges use absolute byte offsets in concatenated Input::Text strings;
    /// token inputs add no bytes. None marks synthetic content without source bytes.
    fn content_spans(&self) -> Vec<Option<std::ops::Range<usize>>>;
    /// Monotonic byte watermark in the same coordinates: future content cannot
    /// refer to earlier bytes. Attribution entries starting below this watermark
    /// may be discarded after processing the latest events and their spans.
    fn consumed_bytes(&self) -> usize;
    type Config;
    fn new(config: Self::Config) -> Self;
    #[cfg(feature = "server")]
    fn grammar_envelope(_: &Self::Config) -> Option<crate::grammar::OutputEnvelope> {
        None
    }
    fn push(&mut self, input: Input<'_>) -> Result<Vec<Event>>;
    fn finish(&mut self, cause: Termination) -> Result<Vec<Event>>;
}

#[cfg(feature = "server")]
pub(crate) mod dsml;
