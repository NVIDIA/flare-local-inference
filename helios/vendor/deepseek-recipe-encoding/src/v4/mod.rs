use deepseek_recipe_core::conversation::{Conversation, ReasoningEffort, ResponseFormat};
use deepseek_recipe_core::messages::{InputMessage, ToolCall};
use deepseek_recipe_core::tools::{ToolChoice, ToolDefinition};
use deepseek_recipe_core::util::json_formatter::stringify_python_style;

use crate::EncodingError;
use crate::PromptEncoding;
use crate::RenderedPrompt;
use crate::TokenizerEncoder;

pub mod dsv4;
pub mod dsv41;

/// Marks the start of the prompt.
pub const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
/// Starts the reasoning content of an assistant turn.
pub const THINKING_START_TOKEN: &str = "<think>";
/// Ends the reasoning content of an assistant turn.
pub const THINKING_END_TOKEN: &str = "</think>";

/// Starts a system message in a V4.1 prompt.
pub const SYSTEM_SP_TOKEN: &str = "<｜System｜>";
/// Starts a user message.
pub const USER_SP_TOKEN: &str = "<｜User｜>";
/// Starts an assistant message.
pub const ASSISTANT_SP_TOKEN: &str = "<｜Assistant｜>";
/// Starts the latest reminder message.
pub const LATEST_REMINDER_SP_TOKEN: &str = "<｜latest_reminder｜>";
/// Terminates a message.
pub const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
/// Tag name prefix of the markup that structures tool calls. The angle brackets
/// come from the surrounding template, as in `<｜DSML｜tool_calls>` for V4 and
/// `<｜DSML｜ calls>` for V4.1.
pub const DSML_SP_TOKEN: &str = "｜DSML｜";

fn parameter_template(
    dsml_token: &str,
    tool_parameter_tag_name: &str,
    key: &str,
    is_str: &str,
    value: &str,
) -> String {
    format!(
        "<{dsml_token}{tool_parameter_tag_name} name=\"{key}\" string=\"{is_str}\">{value}</{dsml_token}{tool_parameter_tag_name}>"
    )
}

fn render_tool_arguments(tool_call: &ToolCall, tool_parameter_tag_name: &str) -> String {
    let arguments =
        serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&tool_call.arguments)
            .unwrap_or_else(|err| {
                tracing::warn!(?err, "invalid tool call arguments");
                serde_json::Map::from_iter([(
                    "arguments".to_owned(),
                    tool_call.arguments.clone().into(),
                )])
            });
    arguments
        .iter()
        .map(|(key, value)| {
            let (is_str, kv_str) = match value.as_str() {
                Some(s) => ("true", s.to_owned()),
                None => ("false", stringify_python_style(value)),
            };
            parameter_template(DSML_SP_TOKEN, tool_parameter_tag_name, key, is_str, &kv_str)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) trait EncodingV4 {
    fn tokenizer(&self) -> Option<&dyn TokenizerEncoder>;

    fn supports_mid_conversation_system(&self) -> bool;

    fn system_token(&self) -> &'static str;

    fn tool_calls_block_name(&self) -> &'static str;

    fn tool_call_tag_name(&self) -> &'static str;

    fn tool_parameter_tag_name(&self) -> &'static str;

    fn render_reasoning_effort(
        &self,
        index: usize,
        thinking_mode: bool,
        effort: Option<ReasoningEffort>,
    ) -> String;
}

fn tool_call_template(encoding: &impl EncodingV4, name: &str, arguments: &str) -> String {
    let tool_call_tag_name = encoding.tool_call_tag_name();
    format!(
        "<{DSML_SP_TOKEN}{tool_call_tag_name} name=\"{name}\">\n{arguments}\n</{DSML_SP_TOKEN}{tool_call_tag_name}>"
    )
}

fn tool_calls_template(encoding: &impl EncodingV4, tool_calls: &str) -> String {
    let tc_block_name = encoding.tool_calls_block_name();
    format!("<{DSML_SP_TOKEN}{tc_block_name}>\n{tool_calls}\n</{DSML_SP_TOKEN}{tc_block_name}>")
}

fn render_tool_calls(encoding: &impl EncodingV4, tool_calls: &[ToolCall]) -> String {
    tool_calls
        .iter()
        .map(|tool_call| {
            tool_call_template(
                encoding,
                &tool_call.name,
                &render_tool_arguments(tool_call, encoding.tool_parameter_tag_name()),
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_message(
    encoding: &impl EncodingV4,
    messages: &[InputMessage],
    index: usize,
    thinking_mode: bool,
    reasoning_effort: Option<ReasoningEffort>,
) -> String {
    let msg = &messages[index];
    let prev = messages[..index].last();
    let reasoning_effort_prompt =
        encoding.render_reasoning_effort(index, thinking_mode, reasoning_effort);
    let mut prompt = if index == 0
        && (!reasoning_effort_prompt.is_empty() || matches!(msg, InputMessage::System { .. }))
    {
        encoding.system_token().to_string()
    } else {
        String::new()
    };
    prompt += &reasoning_effort_prompt;
    match msg {
        InputMessage::System { content } => {
            if index > 0 && encoding.supports_mid_conversation_system() {
                prompt += encoding.system_token();
            }
            prompt += content;
        }
        InputMessage::User { content, .. } => {
            if matches!(
                prev,
                Some(InputMessage::User { .. } | InputMessage::Tool { .. })
            ) {
                prompt += "\n\n";
            } else {
                prompt += USER_SP_TOKEN;
            }
            prompt += content;
        }
        InputMessage::LatestReminder { content } => {
            prompt += LATEST_REMINDER_SP_TOKEN;
            prompt += content;
        }
        InputMessage::Tool { content, .. } => {
            if matches!(
                prev,
                Some(InputMessage::User { .. } | InputMessage::Tool { .. })
            ) {
                prompt += "\n\n";
            } else {
                prompt += USER_SP_TOKEN;
            }
            prompt += &format!("<tool_result>{content}</tool_result>");
        }
        InputMessage::Assistant {
            content,
            reasoning_content,
            tool_calls,
        } => {
            let tool_calls_content = match tool_calls {
                Some(tool_calls) if !tool_calls.is_empty() => {
                    format!(
                        "\n\n{}",
                        tool_calls_template(encoding, &render_tool_calls(encoding, tool_calls))
                    )
                }
                _ => String::new(),
            };
            let mut thinking_part = String::new();
            if thinking_mode && index > 0 {
                if let Some(reasoning_content) = reasoning_content {
                    thinking_part += reasoning_content;
                }
                thinking_part += THINKING_END_TOKEN;
            }
            prompt += ASSISTANT_SP_TOKEN;
            prompt += if !thinking_part.is_empty() {
                THINKING_START_TOKEN
            } else {
                THINKING_END_TOKEN
            };
            prompt += &thinking_part;
            prompt += content;
            prompt += &tool_calls_content;
            prompt += EOS_TOKEN;
        }
    }
    prompt
}

impl<T: EncodingV4> PromptEncoding for T {
    fn encode(&self, conversation: &Conversation) -> Result<Vec<u32>, EncodingError> {
        let tokenizer = self.tokenizer().ok_or(EncodingError::MissingTokenizer)?;
        let rendered = self.render_conversation(conversation);
        tokenizer
            .encode_ids(&rendered.prompt)
            .map_err(EncodingError::Encode)
    }

    fn render_conversation(&self, conversation: &Conversation) -> RenderedPrompt {
        let mut messages = normalize_messages(self, &conversation.messages);
        let has_tools =
            conversation.tool_choice != ToolChoice::None && !conversation.tools.is_empty();
        let format_schema = match &conversation.response_format {
            ResponseFormat::Text => None,
            ResponseFormat::JsonObject => Some(stringify_python_style(&serde_json::json!({
                "type": "json_object"
            }))),
        };
        if has_tools || format_schema.is_some() {
            if !matches!(messages.first(), Some(InputMessage::System { .. })) {
                messages.insert(
                    0,
                    InputMessage::System {
                        content: String::new(),
                    },
                );
            }
            if let Some(InputMessage::System { content }) = messages.first_mut() {
                if has_tools {
                    content.push_str("\n\n");
                    content.push_str(&render_tool_prompt(self, &conversation.tools));
                }
                if let Some(schema) = format_schema {
                    content.push_str("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
                    content.push_str(&schema);
                }
            }
        }
        let mut prompt = BOS_TOKEN.to_string();
        let mut message_boundaries = Vec::with_capacity(messages.len() + 1);
        message_boundaries.push(prompt.len());
        for index in 0..messages.len() {
            prompt += &render_message(
                self,
                &messages,
                index,
                conversation.thinking_mode,
                conversation.reasoning_effort,
            );
            message_boundaries.push(prompt.len());
        }
        prompt.push_str(ASSISTANT_SP_TOKEN);
        prompt.push_str(if conversation.thinking_mode {
            THINKING_START_TOKEN
        } else {
            THINKING_END_TOKEN
        });
        if conversation.tool_choice == ToolChoice::Required && !conversation.tools.is_empty() {
            prompt.push_str(&format!(
                "\n\n<{DSML_SP_TOKEN}{}>\n",
                self.tool_calls_block_name()
            ));
        }
        let image_sources = messages
            .iter()
            .filter_map(|message| match message {
                InputMessage::User { image_sources, .. }
                | InputMessage::Tool { image_sources, .. } => Some(image_sources.as_slice()),
                _ => None,
            })
            .flatten()
            .cloned()
            .collect();
        RenderedPrompt {
            prompt,
            image_sources,
            message_boundaries,
        }
    }
}

fn render_tool_prompt(encoding: &impl EncodingV4, tools: &[ToolDefinition]) -> String {
    let tool_schemas = tools
        .iter()
        .map(|tool| {
            stringify_python_style(&serde_json::json!({
              "name": tool.name,
              "description": tool.description.as_deref().unwrap_or_default(),
              "parameters": tool.parameters,
            }))
        })
        .collect::<Vec<_>>()
        .join("\n");
    let dsml_token = DSML_SP_TOKEN;
    let tc_block_name = encoding.tool_calls_block_name();
    let tool_call_tag_name = encoding.tool_call_tag_name();
    let tool_parameter_tag_name = encoding.tool_parameter_tag_name();
    let thinking_start_token = THINKING_START_TOKEN;
    let thinking_end_token = THINKING_END_TOKEN;
    format!(
        r#"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}{tc_block_name}>" block like the following:

<{dsml_token}{tc_block_name}>
<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME">
<{dsml_token}{tool_parameter_tag_name} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}{tool_parameter_tag_name}>
...
</{dsml_token}{tool_call_tag_name}>
<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME2">
...
</{dsml_token}{tool_call_tag_name}>
</{dsml_token}{tc_block_name}>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"#
    )
}

fn normalize_messages(encoding: &impl EncodingV4, messages: &[InputMessage]) -> Vec<InputMessage> {
    let mut normalized = Vec::new();
    let mut has_non_system = false;
    for message in messages.iter().cloned() {
        match message {
            InputMessage::System { content } if !encoding.supports_mid_conversation_system() => {
                if !has_non_system {
                    if let Some(InputMessage::System { content: head }) = normalized.last_mut() {
                        if !head.is_empty() && !content.is_empty() {
                            head.push_str("\n\n");
                        }
                        head.push_str(&content);
                    } else {
                        normalized.push(InputMessage::System { content });
                    }
                } else if !content.is_empty() {
                    normalized.push(InputMessage::User {
                        content,
                        image_sources: Vec::new(),
                    });
                }
            }
            InputMessage::User {
                content,
                image_sources,
            } => {
                has_non_system = true;
                if let Some(InputMessage::User {
                    content: previous,
                    image_sources: previous_image_sources,
                }) = normalized.last_mut()
                {
                    previous.push_str("\n\n");
                    previous.push_str(&content);
                    previous_image_sources.extend(image_sources);
                } else {
                    normalized.push(InputMessage::User {
                        content,
                        image_sources,
                    });
                }
            }
            message => {
                has_non_system |= !matches!(message, InputMessage::System { .. });
                normalized.push(message);
            }
        }
    }
    sort_tool_results_by_call_order(&mut normalized);
    normalized
}

fn sort_tool_results_by_call_order(messages: &mut [InputMessage]) {
    let mut order: Vec<String> = Vec::new();
    let mut idx = 0;
    while idx < messages.len() {
        match &messages[idx] {
            InputMessage::Assistant {
                tool_calls: Some(tool_calls),
                ..
            } if !tool_calls.is_empty() => {
                order = tool_calls.iter().map(|tc| tc.id.clone()).collect();
                idx += 1;
            }
            InputMessage::User { .. } | InputMessage::Tool { .. } => {
                let start = idx;
                while idx < messages.len()
                    && matches!(
                        messages[idx],
                        InputMessage::User { .. } | InputMessage::Tool { .. }
                    )
                {
                    idx += 1;
                }
                let tool_idxs: Vec<usize> = (start..idx)
                    .filter(|&i| matches!(messages[i], InputMessage::Tool { .. }))
                    .collect();
                if tool_idxs.len() > 1 && !order.is_empty() {
                    let mut tools: Vec<InputMessage> = tool_idxs
                        .iter()
                        .map(|&i| {
                            std::mem::replace(
                                &mut messages[i],
                                InputMessage::LatestReminder {
                                    content: String::new(),
                                },
                            )
                        })
                        .collect();
                    tools.sort_by_key(|m| match m {
                        InputMessage::Tool { tool_call_id, .. } => {
                            order.iter().position(|id| id == tool_call_id).unwrap_or(0)
                        }
                        _ => 0,
                    });
                    for (&i, tool) in tool_idxs.iter().zip(tools) {
                        messages[i] = tool;
                    }
                }
            }
            _ => idx += 1,
        }
    }
}
