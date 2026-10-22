//! Model-specific conversation rendering and token encoding.
//!
//! V4 and V4.1 provide prompt rendering. Attach a tokenizer to an encoding to
//! encode conversations into token IDs.

use deepseek_recipe_core::conversation::Conversation;
use deepseek_recipe_core::multimodal::ImageSource;

pub use tokenizer::TokenizerEncoder;
pub use v4::{dsv4, dsv41};

pub mod v4;

mod tokenizer;

/// A rendered prompt and the images its placeholders refer to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedPrompt {
    /// Model input string.
    pub prompt: String,
    /// Image sources in the order their placeholders appear in the prompt.
    pub image_sources: Vec<ImageSource>,
    /// UTF-8 byte ends of complete normalized messages, before the generation header.
    pub message_boundaries: Vec<usize>,
    pub checkpoint_hints: Vec<usize>,
}

/// Failure to encode a conversation.
#[derive(Debug, thiserror::Error)]
pub enum EncodingError {
    /// The rendered prompt could not be encoded.
    #[error("failed to encode conversation: {0}")]
    Encode(String),
    /// The encoding has no attached tokenizer.
    #[error("no tokenizer is attached to the encoding")]
    MissingTokenizer,
}

/// Render model prompts and encode conversations into token IDs.
pub trait PromptEncoding {
    /// Encode a conversation into model token IDs using the attached tokenizer.
    ///
    /// The tokenizer must be attached to the encoding before this call, through
    /// the encoding's `with_tokenizer` method. Returns
    /// [`EncodingError::MissingTokenizer`] when no tokenizer is attached.
    fn encode(&self, conversation: &Conversation) -> Result<Vec<u32>, EncodingError>;

    /// Render the conversation and the prefix for the next assistant turn.
    fn render_conversation(
        &self,
        conversation: &Conversation,
        hints: &[(usize, usize)],
    ) -> RenderedPrompt;
}
