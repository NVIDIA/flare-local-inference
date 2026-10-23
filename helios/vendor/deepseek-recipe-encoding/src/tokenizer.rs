//! Encode rendered prompts into token ids.

use std::sync::Arc;

use tokenizers::Tokenizer;

/// Encode text into model token ids.
///
/// Implemented for [`Tokenizer`]; other tokenizers can implement it to back a
/// prompt encoding.
pub trait TokenizerEncoder: Send + Sync {
    /// Encode `text` into token ids without added special tokens.
    fn encode_ids(&self, text: &str) -> Result<Vec<u32>, String>;
}

impl TokenizerEncoder for Tokenizer {
    fn encode_ids(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| error.to_string())
    }
}

impl<T: TokenizerEncoder> TokenizerEncoder for Arc<T> {
    fn encode_ids(&self, text: &str) -> Result<Vec<u32>, String> {
        (**self).encode_ids(text)
    }
}
