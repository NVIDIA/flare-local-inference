use deepseek_recipe_core::conversation::ReasoningEffort;

use super::{EncodingV4, SYSTEM_SP_TOKEN};
use crate::TokenizerEncoder;

fn reasoning_effort_template(effort: Option<ReasoningEffort>) -> String {
    let effort_score = match effort {
        Some(ReasoningEffort::Low) => 50,
        None | Some(ReasoningEffort::High) | Some(ReasoningEffort::Xhigh) => 75,
        Some(ReasoningEffort::Max) => 100,
    };
    format!(
        "Reasoning Effort: {effort_score} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    )
}

/// Prompt rendering and token encoding for DeepSeek V4.1.
pub struct DeepseekV41Encoding {
    tokenizer: Option<Box<dyn TokenizerEncoder>>,
}

impl DeepseekV41Encoding {
    /// Create an encoding without an attached tokenizer.
    ///
    /// Token encoding requires a tokenizer, attached with `with_tokenizer`.
    pub fn new() -> Self {
        Self { tokenizer: None }
    }

    /// Attach the tokenizer used by token encoding.
    pub fn with_tokenizer(mut self, tokenizer: impl TokenizerEncoder + 'static) -> Self {
        self.tokenizer = Some(Box::new(tokenizer));
        self
    }
}

impl Default for DeepseekV41Encoding {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodingV4 for DeepseekV41Encoding {
    fn tokenizer(&self) -> Option<&dyn TokenizerEncoder> {
        self.tokenizer.as_deref()
    }

    fn supports_mid_conversation_system(&self) -> bool {
        true
    }

    fn system_token(&self) -> &'static str {
        SYSTEM_SP_TOKEN
    }

    fn tool_calls_block_name(&self) -> &'static str {
        " calls"
    }

    fn tool_call_tag_name(&self) -> &'static str {
        " invoke"
    }

    fn tool_parameter_tag_name(&self) -> &'static str {
        " parameter"
    }

    fn render_reasoning_effort(
        &self,
        index: usize,
        thinking_mode: bool,
        effort: Option<ReasoningEffort>,
    ) -> String {
        if index == 0 && thinking_mode {
            return reasoning_effort_template(effort);
        }
        String::new()
    }
}
