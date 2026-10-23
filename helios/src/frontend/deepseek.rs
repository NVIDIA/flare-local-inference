//! Official DeepSeek conversation encoding with normalized message positions.

use super::{Frontend, Json, Map, PromptEncoder, RenderedPrompt};
use crate::{Error, Failure, FrontendError, Result};
use deepseek_recipe_core::{
    conversation::{Conversation, ReasoningEffort, ResponseFormat},
    messages::{InputMessage, ToolCall},
    multimodal::{IMAGE_SPECIAL_TOKEN, ImageDetail, ImageSource},
    tools::{ToolChoice, ToolDefinition},
};
use deepseek_recipe_encoding::{
    PromptEncoding, dsv4::DeepseekV4Encoding, dsv41::DeepseekV41Encoding,
};

pub(super) fn invalid(message: impl Into<String>) -> Error {
    Error::frontend(FrontendError::Input(Failure::new(message.into())))
}

fn content(value: &Json, hints: &[Json]) -> Result<(String, Vec<ImageSource>, Vec<usize>)> {
    if value.is_null() {
        return Ok((String::new(), vec![], vec![]));
    }
    if let Some(text) = value.as_str() {
        return Ok((
            text.to_owned(),
            vec![],
            hints.iter().map(|_| text.len()).collect(),
        ));
    }
    let parts = value
        .as_array()
        .ok_or_else(|| invalid("message content must be text or content blocks"))?;
    let mut text = Vec::new();
    let mut images = Vec::new();
    let mut positions = Vec::new();
    let mut end = 0;
    for (index, part) in parts.iter().enumerate() {
        match part["type"].as_str() {
            Some("text") => text.push(
                part["text"]
                    .as_str()
                    .ok_or_else(|| invalid("text content must be a string"))?
                    .to_owned(),
            ),
            Some("image_url") => {
                let url = part["image_url"]["url"]
                    .as_str()
                    .ok_or_else(|| invalid("image_url.url must be a string"))?;
                if !url.starts_with("data:image/") {
                    return Err(invalid("image input requires a data URL"));
                }
                text.push(IMAGE_SPECIAL_TOKEN.to_owned());
                images.push(ImageSource::DataUrl {
                    data_url: url.to_owned(),
                    detail: ImageDetail::Auto,
                });
            }
            _ => return Err(invalid("unsupported DeepSeek content block")),
        }
        if index != 0 {
            end += 2;
        }
        end += text.last().unwrap().len();
        if hints.contains(&Json::from(index + 1))
            || (hints == [Json::from(0)] && index + 1 == parts.len())
        {
            positions.push(end);
        }
    }
    Ok((text.join("\n\n"), images, positions))
}

fn tool_arguments(value: &Json) -> Result<String> {
    let parsed = if let Some(text) = value.as_str() {
        serde_json::from_str::<Json>(text)
            .map_err(|_| invalid("tool arguments must contain a JSON object"))?
    } else {
        value.clone()
    };
    if !parsed.is_object() {
        return Err(invalid("tool arguments must contain a JSON object"));
    }
    serde_json::to_string(&parsed).map_err(|e| invalid(e.to_string()))
}

fn conversation(
    frontend: &Frontend,
    messages: &[Json],
    options: &Map<String, Json>,
) -> Result<(Conversation, Vec<(usize, usize)>)> {
    if messages.is_empty() {
        return Err(invalid("cannot encode an empty conversation"));
    }
    for key in options.keys() {
        if matches!(
            key.as_str(),
            "bos_token"
                | "eos_token"
                | "unk_token"
                | "sep_token"
                | "pad_token"
                | "cls_token"
                | "mask_token"
                | "additional_special_tokens"
        ) {
            return Err(invalid(format!(
                "unsupported DeepSeek prompt option: {key}"
            )));
        }
    }
    let mut options = {
        let mut merged = frontend.defaults.clone();
        if options.contains_key("enable_thinking") && !options.contains_key("thinking_mode") {
            merged.remove("thinking_mode");
        }
        if options.contains_key("thinking_mode") {
            merged.remove("enable_thinking");
        }
        merged.extend(options.clone());
        merged
    };
    for key in options.keys() {
        if !matches!(
            key.as_str(),
            "thinking_mode"
                | "enable_thinking"
                | "reasoning_effort"
                | "add_generation_prompt"
                | "continue_final_message"
                | "drop_thinking"
                | "tools"
                | "tool_choice"
                | "response_format"
                | "documents"
                | "bos_token"
                | "eos_token"
                | "unk_token"
                | "sep_token"
                | "pad_token"
                | "cls_token"
                | "mask_token"
                | "additional_special_tokens"
        ) {
            return Err(invalid(format!(
                "unsupported DeepSeek prompt option: {key}"
            )));
        }
    }
    if !options.contains_key("thinking_mode") && !options.contains_key("enable_thinking") {
        options.insert("thinking_mode".into(), "thinking".into());
    }
    if let Some(value) = options.get("continue_final_message") {
        if !matches!(value, Json::Null | Json::Bool(false)) && value.as_str() != Some("") {
            return Err(invalid(
                "DeepSeek prompt encoding does not support continue_final_message",
            ));
        }
    }
    if options.get("add_generation_prompt") != Some(&Json::Bool(true)) {
        return Err(invalid(
            "DeepSeek prompt encoding requires add_generation_prompt=true",
        ));
    }
    if options.contains_key("drop_thinking") {
        return Err(invalid(
            "DeepSeek prompt encoding does not support drop_thinking",
        ));
    }
    if options.get("documents").is_some_and(|v| !v.is_null()) {
        return Err(invalid(
            "DeepSeek prompt encoding does not support documents",
        ));
    }
    let thinking_mode = match options.get("thinking_mode").filter(|v| !v.is_null()) {
        Some(Json::String(value)) if value == "thinking" => true,
        Some(Json::String(value)) if value == "chat" => false,
        Some(_) => return Err(invalid("thinking_mode must be thinking or chat")),
        None => match options.get("enable_thinking").filter(|v| !v.is_null()) {
            None => true,
            Some(Json::Bool(value)) => *value,
            _ => return Err(invalid("enable_thinking must be a boolean")),
        },
    };
    let reasoning_effort = Some(
        match options.get("reasoning_effort").filter(|v| !v.is_null()) {
            None => ReasoningEffort::High,
            Some(Json::String(value)) => match value.as_str() {
                "low" => ReasoningEffort::Low,
                "high" => ReasoningEffort::High,
                "xhigh" => ReasoningEffort::Xhigh,
                "max" => ReasoningEffort::Max,
                _ => return Err(invalid("reasoning_effort must be low, high, xhigh, or max")),
            },
            _ => {
                return Err(invalid(
                    "numeric reasoning_effort is not supported by DeepSeek prompt encoding",
                ));
            }
        },
    );
    let tool_choice = match options
        .get("tool_choice")
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_str()
                .ok_or_else(|| invalid("tool_choice must be a string"))
        })
        .transpose()?
        .unwrap_or("auto")
    {
        "auto" => ToolChoice::Auto,
        "none" => ToolChoice::None,
        "required" => ToolChoice::Required,
        _ => return Err(invalid("unsupported tool_choice")),
    };
    let mut tools = Vec::new();
    if let Some(value) = options.get("tools").filter(|v| !v.is_null()) {
        for tool in value
            .as_array()
            .ok_or_else(|| invalid("tools must be an array"))?
        {
            let function = tool.get("function").unwrap_or(tool);
            tools.push(ToolDefinition {
                name: function["name"]
                    .as_str()
                    .ok_or_else(|| invalid("tool requires a name"))?
                    .to_owned(),
                description: function
                    .get("description")
                    .filter(|v| !v.is_null())
                    .map(|v| {
                        v.as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| invalid("tool description must be text"))
                    })
                    .transpose()?,
                parameters: function
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({"type":"object","properties":{}})),
                strict: function
                    .get("strict")
                    .filter(|v| !v.is_null())
                    .map(|v| {
                        v.as_bool()
                            .ok_or_else(|| invalid("tool strict must be a boolean"))
                    })
                    .transpose()?,
            });
        }
    }
    if tool_choice == ToolChoice::Required && tools.is_empty() {
        return Err(invalid("required tool choice needs at least one tool"));
    }
    if tool_choice == ToolChoice::Required && thinking_mode {
        return Err(invalid(
            "thinking mode does not support required tool choice",
        ));
    }
    let response_format = match options.get("response_format").filter(|v| !v.is_null()) {
        None => ResponseFormat::Text,
        Some(value) if value == "text" || value["type"] == "text" => ResponseFormat::Text,
        Some(value) if value == "json_object" || value["type"] == "json_object" => {
            ResponseFormat::JsonObject
        }
        _ => return Err(invalid("unsupported DeepSeek response_format")),
    };
    let supplied_text = format!(
        "{}{}",
        serde_json::to_string(&messages).unwrap(),
        options
            .get("tools")
            .map(Json::to_string)
            .unwrap_or_default()
    );
    let package_marker = frontend
        .image_placeholder_token_id
        .and_then(|id| frontend.tokenizer.id_to_token(id));
    if supplied_text.contains(IMAGE_SPECIAL_TOKEN)
        || package_marker
            .as_ref()
            .is_some_and(|marker| supplied_text.contains(marker))
    {
        return Err(invalid(
            "literal DeepSeek image placeholder is ambiguous; use an image_url content block",
        ));
    }
    let mut converted = Vec::new();
    let mut hints = Vec::new();
    for message in messages {
        if message.get("task").is_some() || message.get("prefix").is_some() {
            return Err(invalid(
                "DeepSeek task and assistant-prefill fields are unsupported",
            ));
        }
        let block_hints = message["_checkpoint_hints"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let (text, image_sources, positions) = content(&message["content"], block_hints)?;
        let suffix = if message["role"] == "tool" {
            "</tool_result>".len()
        } else {
            0
        };
        hints.extend(
            positions
                .into_iter()
                .map(|position| (converted.len(), position + suffix)),
        );
        let role = message["role"]
            .as_str()
            .ok_or_else(|| invalid("message requires a role"))?;
        if !image_sources.is_empty() && !matches!(role, "user" | "tool") {
            return Err(invalid("images require user or tool messages"));
        }
        converted.push(match role {
            "system" | "developer" => InputMessage::System { content: text },
            "user" => InputMessage::User {
                content: text,
                image_sources,
            },
            "tool" => InputMessage::Tool {
                content: text,
                image_sources,
                tool_call_id: message["tool_call_id"]
                    .as_str()
                    .ok_or_else(|| invalid("tool message requires tool_call_id"))?
                    .to_owned(),
            },
            "assistant" => {
                let reasoning_content = message
                    .get("reasoning_content")
                    .or_else(|| message.get("thinking"))
                    .filter(|v| !v.is_null())
                    .map(|v| {
                        v.as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| invalid("assistant reasoning must be text"))
                    })
                    .transpose()?;
                let tool_calls = message
                    .get("tool_calls")
                    .filter(|v| !v.is_null())
                    .map(|calls| {
                        calls
                            .as_array()
                            .ok_or_else(|| invalid("tool_calls must be an array"))?
                            .iter()
                            .map(|call| {
                                let function = &call["function"];
                                Ok(ToolCall {
                                    id: call["id"].as_str().unwrap_or("").to_owned(),
                                    name: function["name"]
                                        .as_str()
                                        .ok_or_else(|| invalid("tool call requires a name"))?
                                        .to_owned(),
                                    arguments: tool_arguments(&function["arguments"])?,
                                })
                            })
                            .collect::<Result<Vec<_>>>()
                    })
                    .transpose()?;
                InputMessage::Assistant {
                    content: text,
                    reasoning_content,
                    tool_calls,
                }
            }
            _ => return Err(invalid("unsupported DeepSeek message role")),
        });
    }
    Ok((
        Conversation {
            messages: converted,
            thinking_mode,
            tools,
            tool_choice,
            reasoning_effort,
            response_format,
        },
        hints,
    ))
}

pub(super) fn validate_defaults(frontend: &Frontend) -> Result<()> {
    conversation(
        frontend,
        &[serde_json::json!({"role":"user","content":""})],
        &Map::new(),
    )
    .map(|_| ())
}

fn official(
    encoder: PromptEncoder,
    conversation: &Conversation,
    hints: &[(usize, usize)],
) -> deepseek_recipe_encoding::RenderedPrompt {
    match encoder {
        PromptEncoder::DeepseekV4 => {
            DeepseekV4Encoding::new().render_conversation(conversation, hints)
        }
        PromptEncoder::DeepseekV41 => {
            DeepseekV41Encoding::new().render_conversation(conversation, hints)
        }
        PromptEncoder::Jinja => unreachable!(),
    }
}

pub(super) fn render(
    frontend: &Frontend,
    messages: &[Json],
    options: &Map<String, Json>,
) -> Result<RenderedPrompt> {
    let (conversation, hints) = conversation(frontend, messages, options)?;
    let rendered = official(frontend.prompt_encoder, &conversation, &hints);
    let image_sources = rendered
        .image_sources
        .into_iter()
        .map(|source| match source {
            ImageSource::DataUrl { data_url, .. } => Ok(data_url),
            _ => Err(invalid("image input requires a data URL")),
        })
        .collect::<Result<Vec<_>>>()?;
    let mut token_ids = Vec::new();
    let mut offsets = Vec::new();
    let mut byte_start = 0;
    let spans: Vec<_> = rendered.prompt.split(IMAGE_SPECIAL_TOKEN).collect();
    if spans.len() - 1 != image_sources.len() {
        return Err(invalid(
            "image placeholder count does not match image sources",
        ));
    }
    let placeholder =
        if image_sources.is_empty() {
            None
        } else {
            Some(frontend.image_placeholder_token_id.ok_or_else(|| {
                invalid("image inputs require a prepared image placeholder token ID")
            })?)
        };
    for (index, span) in spans.into_iter().enumerate() {
        if index != 0 {
            token_ids.push(placeholder.unwrap());
            offsets.push((byte_start, byte_start + IMAGE_SPECIAL_TOKEN.len()));
            byte_start += IMAGE_SPECIAL_TOKEN.len();
        }
        let encoding = frontend.tokenizer.encode(span, false).map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(error.to_string(), error)))
        })?;
        token_ids.extend_from_slice(encoding.get_ids());
        offsets.extend(encoding.get_offsets().iter().map(|&(start,end)| (byte_start+start,byte_start+end)));
        byte_start += span.len();
    }
    let message_boundaries = super::boundaries::token_ends(
        &rendered.message_boundaries,
        &rendered.checkpoint_hints,
        &offsets,
        rendered.prompt.len(),
    )?;
    Ok(RenderedPrompt {
        text: rendered.prompt,
        token_ids,
        image_sources,
        image_checkpoint_hints: Vec::new(),
        message_boundaries,
    })
}

impl Frontend {
    pub fn image_placeholder_token_id(&self) -> Result<u32> {
        if let Some(id) = self.image_placeholder_token_id {
            return Ok(id);
        }
        let ids = self.encode("<｜deepseek_image｜>", false)?;
        if ids.len() != 1 {
            return Err(invalid("image placeholder must encode to one token"));
        }
        Ok(ids[0])
    }

}
