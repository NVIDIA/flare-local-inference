//! Protocol-neutral inputs lowered only at the model frontend boundary.

use crate::{AdmissionError, Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Clone, Debug, PartialEq)]
pub enum Input {
    Conversation(Vec<Message>),
    Raw(Vec<RawPrompt>),
}
#[derive(Clone, Debug, PartialEq)]
pub enum RawPrompt {
    Text(String),
    Tokens(Vec<u32>),
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub role: Role,
    pub parts: Vec<Part>,
    pub phase: Option<String>,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Part {
    Text(String),
    Image(String),
    Reasoning(String),
    ToolCall {
        id: Option<String>,
        name: String,
        arguments: Value,
    },
    ToolResult {
        id: Option<String>,
        content: Vec<Part>,
        is_error: bool,
    },
}
#[derive(Clone, Debug, PartialEq)]
pub struct Tool {
    pub name: String,
    pub namespace: Option<String>,
    pub description: String,
    pub parameters: Value,
    pub custom: bool,
    pub grammar: Option<ToolGrammar>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ToolGrammar {
    pub syntax: String,
    pub source: String,
}
#[derive(Clone, Debug, PartialEq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Named(String),
}
impl Tool {
    pub fn native_name(&self) -> String {
        match &self.namespace {
            Some(ns) => format!("{ns}__{}", self.name),
            None => self.name.clone(),
        }
    }
}
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct Sampling {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<serde_json::Number>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Constraint {
    JsonObject,
    Json(Value),
    Lark(String),
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Effort {
    Low,
    High,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogprobOptions {
    pub top: usize,
    pub token_ids: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Options {
    pub sampling: Sampling,
    pub logprobs: Option<LogprobOptions>,
    pub prompt_logprobs: Option<usize>,
    pub effort: Option<Effort>,
    pub output_limit: Option<u64>,
    pub stops: Vec<String>,
    pub token_stops: Vec<Vec<u32>>,
    pub constraint: Option<Constraint>,
    pub ignore_eos: bool,
    pub template: Map<String, Value>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub input: Input,
    // Message index and exclusive content-part end.
    pub checkpoint_hints: Vec<(usize, usize)>,
    pub options: Options,
    pub tools: Vec<Tool>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
    pub choices: usize,
}
impl Request {
    pub fn validate(&self) -> Result<()> {
        let mut names = std::collections::HashSet::new();
        for tool in &self.tools {
            if tool.name.is_empty()
                || !tool.parameters.is_object()
                || !names.insert(tool.native_name())
            {
                return Err(
                    "tool names must be nonempty and unambiguous, with object schemas".into(),
                );
            }
        }
        if self.tool_choice == ToolChoice::Required && self.tools.is_empty() {
            return Err("required tool choice needs at least one tool".into());
        }
        if let ToolChoice::Named(name) = &self.tool_choice {
            if !names.contains(name) {
                return Err(format!("selected tool {name:?} is not declared in tools").into());
            }
        }
        Ok(())
    }
    pub fn generation_count(&self) -> Result<usize> {
        let prompts = match &self.input {
            Input::Raw(prompts) => prompts.len(),
            _ => 1,
        };
        prompts
            .checked_mul(self.choices)
            .filter(|n| *n > 0)
            .ok_or_else(|| {
                Error::from("prompt and choice counts must be positive and fit in memory")
            })
    }
}
pub fn unsupported(message: impl Into<String>) -> Error {
    Error::admission(AdmissionError::Feature, message.into())
}

fn content(parts: &[Part]) -> Result<Value> {
    let mut values = Vec::new();
    for part in parts {
        match part {
            Part::Text(text) => values.push(json!({"type":"text","text":text})),
            Part::Image(url) => values.push(json!({"type":"image_url","image_url":{"url":url}})),
            _ => return Err("tool result content must contain text or images".into()),
        }
    }
    if values.is_empty() {
        Ok(Value::String(String::new()))
    } else if values.len() == 1 && values[0]["type"] == "text" {
        Ok(values[0]["text"].clone())
    } else {
        Ok(Value::Array(values))
    }
}

/// Converts semantic parts to the message context consumed by prepared templates.
pub fn template_messages(
    messages: &[Message],
    tool_choice: ToolChoice,
    hints: &[(usize, usize)],
) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    let lower_instructions = messages
        .iter()
        .any(|message| message.role == Role::Developer);
    let instructions: Vec<_> = messages
        .iter()
        .filter(|m| lower_instructions && matches!(m.role, Role::System | Role::Developer))
        .collect();
    if !instructions.is_empty() {
        let labeled = instructions.len() > 1 || instructions[0].role == Role::Developer;
        let mut sections = Vec::new();
        for instruction in instructions {
            let mut text = String::new();
            for part in &instruction.parts {
                let Part::Text(value) = part else {
                    return Err("system and developer instructions require text".into());
                };
                text.push_str(value);
            }
            if labeled {
                let role = if instruction.role == Role::System {
                    "System"
                } else {
                    "Developer"
                };
                sections.push(format!("{role} instructions:\n{text}"));
            } else {
                sections.push(text);
            }
        }
        out.push(json!({"role":"system","content":sections.join("\n\n")}));
    }
    let mut assistant_start = None;
    for (message_index, message) in messages.iter().enumerate() {
        if lower_instructions && matches!(message.role, Role::System | Role::Developer) {
            continue;
        }
        if message.role == Role::Assistant {
            if let Some(start) = assistant_start {
                order_tool_result_batch(&mut out[start..]);
            }
            assistant_start = Some(out.len());
        }
        let before = out.len();
        let mut current = json!({"role": message.role, "content": ""});
        let mut ordinary = Vec::new();
        let mut ordinary_hints = Vec::new();
        let flush = |current: &mut Value,
                     ordinary: &mut Vec<Part>,
                     hints: &mut Vec<usize>,
                     out: &mut Vec<Value>|
         -> Result<()> {
            current["content"] = content(ordinary)?;
            ordinary.clear();
            if !hints.is_empty() {
                current["_checkpoint_hints"] = json!(hints);
                hints.clear();
            }
            if current["content"] != ""
                || current.get("reasoning_content").is_some()
                || current.get("tool_calls").is_some()
            {
                out.push(current.clone());
            }
            *current = json!({"role": message.role, "content":""});
            Ok(())
        };
        for (part_index, part) in message.parts.iter().enumerate() {
            match part {
                Part::Text(_) | Part::Image(_) => {
                    if current.get("tool_calls").is_some() {
                        flush(&mut current, &mut ordinary, &mut ordinary_hints, &mut out)?;
                    }
                    ordinary.push(part.clone());
                    if hints.contains(&(message_index, part_index + 1)) {
                        ordinary_hints.push(ordinary.len());
                    }
                }
                Part::Reasoning(text) => {
                    if !ordinary.is_empty() || current.get("tool_calls").is_some() {
                        flush(&mut current, &mut ordinary, &mut ordinary_hints, &mut out)?;
                    }
                    let previous = current
                        .get("reasoning_content")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    current["reasoning_content"] = json!(format!("{previous}{text}"));
                }
                Part::ToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    if current.get("tool_calls").is_none() {
                        current["tool_calls"] = json!([]);
                    }
                    let mut call =
                        json!({"type":"function","function":{"name":name,"arguments":arguments}});
                    if let Some(id) = id {
                        call["id"] = json!(id);
                    }
                    current["tool_calls"].as_array_mut().unwrap().push(call);
                }
                Part::ToolResult {
                    id,
                    content: parts,
                    is_error,
                } => {
                    flush(&mut current, &mut ordinary, &mut ordinary_hints, &mut out)?;
                    let mut parts = parts.clone();
                    if *is_error {
                        parts.insert(0, Part::Text("Tool execution failed:".into()));
                    }
                    let mut result =
                        json!({"role":"tool","content":content(&parts)?,"is_error":is_error});
                    if let Some(id) = id {
                        result["tool_call_id"] = json!(id);
                    }
                    if hints.contains(&(message_index, part_index + 1)) {
                        // Zero denotes a complete tool result, including its wrapper.
                        result["_checkpoint_hints"] = json!([0]);
                    }
                    out.push(result);
                }
            }
        }
        flush(&mut current, &mut ordinary, &mut ordinary_hints, &mut out)?;
        if before == out.len() {
            out.push(json!({"role":message.role,"content":""}));
        }
    }
    if let Some(start) = assistant_start {
        order_tool_result_batch(&mut out[start..]);
    }
    if tool_choice == ToolChoice::Required {
        let instruction = "You must call at least one of the provided tools in your next response. Choose the appropriate tool and arguments.";
        append_system_instruction(&mut out, instruction)?;
    }
    Ok(out)
}

/// Keeps result payloads in call order before templates can reorder their markers.
fn order_tool_result_batch(messages: &mut [Value]) {
    let mut ranks = std::collections::HashMap::new();
    // One semantic assistant turn can lower to several template messages when
    // text follows a call. All of its calls belong to the same result batch.
    for call in messages
        .iter()
        .filter(|message| message["role"] == "assistant")
        .filter_map(|message| message.get("tool_calls").and_then(Value::as_array))
        .flatten()
    {
        let Some(id) = call
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
        else {
            return;
        };
        if ranks.insert(id, ranks.len()).is_some() {
            return;
        }
    }
    let mut slots = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (index, message) in messages.iter().enumerate().skip(1) {
        if message["role"] != "tool" {
            continue;
        }
        let Some(id) = message.get("tool_call_id").and_then(Value::as_str) else {
            return;
        };
        let Some(&rank) = ranks.get(id) else {
            return;
        };
        if !seen.insert(id) {
            return;
        }
        slots.push((index, rank));
    }
    // ID-less or ambiguous histories retain their template-defined behavior.
    // Only complete messages move; intervening text and each payload stay intact.
    let mut results: Vec<_> = slots
        .iter()
        .map(|&(index, rank)| (rank, std::mem::take(&mut messages[index])))
        .collect();
    results.sort_by_key(|&(rank, _)| rank);
    for ((index, _), (_, result)) in slots.into_iter().zip(results) {
        messages[index] = result;
    }
}

pub(super) fn append_system_instruction(
    messages: &mut Vec<Value>,
    instruction: &str,
) -> Result<()> {
    if let Some(message) = messages.first_mut().filter(|m| m["role"] == "system") {
        match &mut message["content"] {
            Value::String(text) => {
                text.push_str("\n\n");
                text.push_str(instruction);
            }
            Value::Array(parts) => parts.push(json!({"type":"text","text":instruction})),
            _ => return Err("system instructions require text".into()),
        }
    } else {
        messages.insert(0, json!({"role":"system","content":instruction}));
    }
    Ok(())
}

#[cfg(test)]
mod tool_result_order_tests {
    use super::*;

    fn calls(ids: &[Option<&str>]) -> Message {
        Message {
            role: Role::Assistant,
            parts: ids
                .iter()
                .enumerate()
                .map(|(index, id)| Part::ToolCall {
                    id: id.map(str::to_owned),
                    name: format!("lookup_{index}"),
                    arguments: json!({}),
                })
                .collect(),
            phase: None,
        }
    }

    fn result(id: Option<&str>, label: &str) -> Message {
        Message {
            role: Role::Tool,
            parts: vec![Part::ToolResult {
                id: id.map(str::to_owned),
                content: vec![
                    Part::Text(label.into()),
                    Part::Image(format!("image:{label}:1")),
                    Part::Image(format!("image:{label}:2")),
                ],
                is_error: false,
            }],
            phase: None,
        }
    }

    fn user(text: &str) -> Message {
        Message {
            role: Role::User,
            parts: vec![Part::Text(text.into())],
            phase: None,
        }
    }

    fn lower(messages: &[Message]) -> Vec<Value> {
        template_messages(messages, ToolChoice::Auto, &[]).unwrap()
    }

    #[test]
    fn complete_payloads_follow_call_order_across_fixed_text_slots() {
        let assistant = calls(&[Some("a"), Some("b")]);
        let a = result(Some("a"), "alpha");
        let b = result(Some("b"), "beta");
        let text = Message {
            role: Role::User,
            parts: vec![
                Part::Text("between".into()),
                Part::Image("image:user".into()),
            ],
            phase: None,
        };
        let reversed = lower(&[assistant.clone(), b, text.clone(), a]);
        let expected = lower(&[
            assistant,
            result(Some("a"), "alpha"),
            text,
            result(Some("b"), "beta"),
        ]);
        assert_eq!(reversed, expected);
        assert_eq!(
            reversed[1]["content"][1]["image_url"]["url"],
            "image:alpha:1"
        );
        assert_eq!(
            reversed[1]["content"][2]["image_url"]["url"],
            "image:alpha:2"
        );
        assert_eq!(reversed[2]["content"][1]["image_url"]["url"], "image:user");
        assert_eq!(reversed[3]["content"][0]["text"], "beta");
    }

    #[test]
    fn reused_ids_are_bound_to_their_own_assistant_batch() {
        let messages = lower(&[
            calls(&[Some("a"), Some("b")]),
            result(Some("b"), "first beta"),
            result(Some("a"), "first alpha"),
            calls(&[Some("b"), Some("a")]),
            result(Some("a"), "second alpha"),
            result(Some("b"), "second beta"),
        ]);
        assert_eq!(messages[1]["content"][0]["text"], "first alpha");
        assert_eq!(messages[2]["content"][0]["text"], "first beta");
        assert_eq!(messages[4]["content"][0]["text"], "second beta");
        assert_eq!(messages[5]["content"][0]["text"], "second alpha");
    }

    #[test]
    fn partial_batches_sort_without_requiring_all_results() {
        let messages = lower(&[
            calls(&[Some("a"), Some("b"), Some("c")]),
            result(Some("c"), "gamma"),
            user("fixed"),
            result(Some("a"), "alpha"),
        ]);
        assert_eq!(messages[1]["tool_call_id"], "a");
        assert_eq!(messages[2]["content"], "fixed");
        assert_eq!(messages[3]["tool_call_id"], "c");
    }

    #[test]
    fn split_assistant_fragments_share_the_originating_call_batch() {
        let mut assistant = calls(&[Some("a"), Some("b"), Some("c")]);
        assistant.parts.insert(1, Part::Text("continuing".into()));
        let messages = lower(&[
            assistant,
            result(Some("c"), "gamma"),
            result(Some("a"), "alpha"),
            result(Some("b"), "beta"),
        ]);
        assert_eq!(messages[0]["tool_calls"][0]["id"], "a");
        assert_eq!(messages[1]["tool_calls"][0]["id"], "b");
        assert_eq!(messages[2]["tool_call_id"], "a");
        assert_eq!(messages[3]["tool_call_id"], "b");
        assert_eq!(messages[4]["tool_call_id"], "c");
    }

    #[test]
    fn ambiguous_batches_retain_every_message_in_arrival_order() {
        let valid = json!({"role":"assistant", "tool_calls":[{"id":"a"},{"id":"b"}]});
        for (assistant, results) in [
            (
                json!({"role":"assistant","tool_calls":[{}, {"id":"b"}]}),
                vec![
                    json!({"role":"tool","tool_call_id":"b"}),
                    json!({"role":"tool"}),
                ],
            ),
            (
                json!({"role":"assistant","tool_calls":[{"id":"a"},{"id":"a"}]}),
                vec![
                    json!({"role":"tool","tool_call_id":"a"}),
                    json!({"role":"tool","tool_call_id":"a"}),
                ],
            ),
            (
                valid.clone(),
                vec![
                    json!({"role":"tool","tool_call_id":"b"}),
                    json!({"role":"tool"}),
                ],
            ),
            (
                valid.clone(),
                vec![
                    json!({"role":"tool","tool_call_id":"b"}),
                    json!({"role":"tool","tool_call_id":"unknown"}),
                ],
            ),
            (
                valid,
                vec![
                    json!({"role":"tool","tool_call_id":"b"}),
                    json!({"role":"tool","tool_call_id":"a"}),
                    json!({"role":"tool","tool_call_id":"b"}),
                ],
            ),
        ] {
            let mut messages = vec![assistant];
            messages.extend(results);
            let original = messages.clone();
            order_tool_result_batch(&mut messages);
            assert_eq!(messages, original);
        }
    }

    #[test]
    fn assistant_without_calls_ends_the_binding_scope() {
        let messages = lower(&[
            calls(&[Some("a"), Some("b")]),
            Message {
                role: Role::Assistant,
                parts: vec![Part::Text("done".into())],
                phase: None,
            },
            result(Some("b"), "beta"),
            result(Some("a"), "alpha"),
        ]);
        assert_eq!(messages[2]["tool_call_id"], "b");
        assert_eq!(messages[3]["tool_call_id"], "a");
    }

    #[test]
    fn normalized_messages_keep_image_payload_and_marker_order() {
        let ordered = vec![
            calls(&[Some("a"), Some("b")]),
            result(Some("a"), "alpha"),
            result(Some("b"), "beta"),
        ];
        let mut reversed = ordered.clone();
        reversed.swap(1, 2);
        let render = |messages: &[Message]| {
            let mut normalized = lower(messages);
            let mut images = Vec::new();
            for message in &mut normalized {
                if let Some(parts) = message["content"].as_array_mut() {
                    for part in parts {
                        if part["type"] == "image_url" {
                            images.push(part["image_url"]["url"].as_str().unwrap().to_owned());
                            *part = json!({"type":"image"});
                        }
                    }
                }
            }
            let rendered = normalized.iter().map(|message| message["content"].to_string())
                .collect::<Vec<_>>().join("\n");
            (rendered, images)
        };
        let expected = render(&ordered);
        assert_eq!(render(&reversed), expected);
        assert_eq!(
            expected.1,
            [
                "image:alpha:1",
                "image:alpha:2",
                "image:beta:1",
                "image:beta:2"
            ]
        );
        assert_eq!(expected.0.matches("\"type\":\"image\"").count(), 4);
        assert!(expected.0.find("alpha").unwrap() < expected.0.find("beta").unwrap());
    }
}
