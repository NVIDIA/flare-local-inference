use deepseek_recipe_core::conversation::ReasoningEffort;

use super::EncodingV4;
use crate::TokenizerEncoder;

const REASONING_EFFORT_HIGH: &str = r"Reasoning Effort: Absolute maximum with no shortcuts permitted.
You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.
Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.

";
const REASONING_EFFORT_MAX: &str = r"Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.
You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.
Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.

";

/// Prompt rendering and token encoding for DeepSeek V4.
pub struct DeepseekV4Encoding {
    tokenizer: Option<Box<dyn TokenizerEncoder>>,
}

impl DeepseekV4Encoding {
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

impl Default for DeepseekV4Encoding {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodingV4 for DeepseekV4Encoding {
    fn tokenizer(&self) -> Option<&dyn TokenizerEncoder> {
        self.tokenizer.as_deref()
    }

    fn supports_mid_conversation_system(&self) -> bool {
        false
    }

    fn system_token(&self) -> &'static str {
        ""
    }

    fn tool_calls_block_name(&self) -> &'static str {
        "tool_calls"
    }

    fn tool_call_tag_name(&self) -> &'static str {
        "invoke"
    }

    fn tool_parameter_tag_name(&self) -> &'static str {
        "parameter"
    }

    fn render_reasoning_effort(
        &self,
        index: usize,
        thinking_mode: bool,
        effort: Option<ReasoningEffort>,
    ) -> String {
        if index != 0 || !thinking_mode {
            return String::new();
        }
        match effort {
            Some(ReasoningEffort::Low) => String::new(),
            None | Some(ReasoningEffort::High) | Some(ReasoningEffort::Xhigh) => {
                REASONING_EFFORT_HIGH.to_string()
            }
            Some(ReasoningEffort::Max) => REASONING_EFFORT_MAX.to_string(),
        }
    }
}
