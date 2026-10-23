//! Incremental think/tool envelopes with statically selected model payload syntax.

use super::{Event, Input, ModelOutput, Progress, ReasoningParser, Termination, ToolCallParser};
use crate::Result;
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputStart {
    Content,
    Reasoning,
    ToolBody,
}

#[derive(Clone)]
pub struct OutputConfig {
    pub structured: bool,
    pub start: OutputStart,
    pub tools: Vec<Value>,
}

pub trait ToolDialect {
    const NAME: &'static str;
    const TRIM_NEWLINES: bool;
    const OPEN: &'static str;
    const CLOSE: &'static str;
    const FUNCTION: Option<(&'static str, &'static str)> = None;
    const TOOL_ENDS_REASONING: bool = false;
    fn parse(body: &str, tools: &[Value]) -> Result<Vec<(String, String)>>;
}

pub struct ThinkReasoning<D: ToolDialect> {
    dialect: std::marker::PhantomData<D>,
    buffer: String,
    open: String,
    close: String,
    trim_newlines: usize,
    tools: bool,
    trim_wrap: bool,
    tool_ends_reasoning: bool,
}
impl<D: ToolDialect> ReasoningParser for ThinkReasoning<D> {
    fn push(&mut self, input: Input<'_>) -> Result<Progress> {
        let Input::Text(text) = input else {
            return Ok(Progress::default());
        };
        self.buffer.push_str(text);
        while self.trim_newlines > 0 && !self.buffer.is_empty() {
            if self.buffer.starts_with('\n') {
                self.buffer.remove(0);
                self.trim_newlines -= 1;
            } else {
                self.trim_newlines = 0;
            }
        }
        let mut progress = Progress::default();
        let mut boundary = self
            .buffer
            .find(&self.close)
            .map(|at| (at, self.close.len()));
        if self.tools && self.tool_ends_reasoning {
            if let Some(at) = self.buffer.find(D::OPEN) {
                if boundary.is_none_or(|(end, _)| at < end) {
                    boundary = Some((at, 0));
                }
            }
        }
        if let Some((end, consumed)) = boundary {
            let text = &self.buffer[..end];
            let text = if self.trim_wrap {
                text.strip_suffix('\n').unwrap_or(text)
            } else {
                text
            }
            .to_owned();
            let text = if self.tool_ends_reasoning {
                text.replace(&self.open, "")
            } else {
                text
            };
            if !text.is_empty() {
                progress.events.push(Event::Reasoning(text));
            }
            self.buffer.drain(..end + consumed);
            progress.remainder = Some(std::mem::take(&mut self.buffer));
        } else {
            let mut hold =
                partial_marker(&self.buffer, &[&self.close, &format!("\n{}", self.close)]);
            if self.tool_ends_reasoning {
                hold = hold.max(partial_marker(&self.buffer, &[&self.open]));
                if self.tools {
                    hold = hold.max(partial_marker(&self.buffer, &[D::OPEN]));
                }
            }
            let visible = self.buffer.len() - hold;
            let text: String = self.buffer.drain(..visible).collect();
            let text = if self.tool_ends_reasoning {
                text.replace(&self.open, "")
            } else {
                text
            };
            if !text.is_empty() {
                progress.events.push(Event::Reasoning(text));
            }
        }
        Ok(progress)
    }
    fn finish(&mut self, cause: Termination) -> Result<Vec<Event>> {
        let text = std::mem::take(&mut self.buffer);
        Ok(
            if text.is_empty()
                || matches!(cause, Termination::Cancelled | Termination::ExecutionError)
            {
                vec![]
            } else {
                vec![Event::Reasoning(text)]
            },
        )
    }
}

pub struct TaggedTools<D: ToolDialect> {
    dialect: std::marker::PhantomData<D>,
    buffer: String,
    tools: Arc<[Value]>,
    index: usize,
}
impl<D: ToolDialect> ToolCallParser for TaggedTools<D> {
    fn push(&mut self, input: Input<'_>) -> Result<Progress> {
        let Input::Text(text) = input else {
            return Ok(Progress::default());
        };
        self.buffer.push_str(text);
        if self.buffer.len() > 1024 * 1024 {
            return Err("tool call exceeds the 1 MiB parser bound".into());
        }
        let mut boundary = self
            .buffer
            .find(D::CLOSE)
            .map(|at| (at, at + D::CLOSE.len()));
        if let Some((_, close)) = D::FUNCTION {
            if let Some(at) = self.buffer.find(close) {
                if boundary.is_none_or(|(end, _)| at < end) {
                    boundary = Some((at + close.len(), at + close.len()));
                }
            }
        }
        let Some((end, consumed)) = boundary else {
            return Ok(Progress::default());
        };
        let calls = D::parse(&self.buffer[..end], &self.tools)?;
        self.buffer.drain(..consumed);
        let events = calls
            .into_iter()
            .enumerate()
            .flat_map(|(offset, (name, arguments))| {
                let index = self.index + offset;
                [
                    Event::ToolBegin {
                        index,
                        name,
                        id: None,
                    },
                    Event::ToolArguments {
                        index,
                        text: arguments,
                    },
                    Event::ToolEnd { index },
                ]
            })
            .collect();
        Ok(Progress {
            events,
            remainder: Some(std::mem::take(&mut self.buffer)),
        })
    }
    fn finish(&mut self, cause: Termination) -> Result<Vec<Event>> {
        if matches!(
            cause,
            Termination::Cancelled
                | Termination::ExecutionError
                | Termination::UserStop
                | Termination::OutputLimit
        ) {
            self.buffer.clear();
            return Ok(vec![]);
        }
        Err(format!("incomplete {} tool call", D::NAME).into())
    }
}

enum Mode {
    Verbatim,
    Content,
    Reasoning,
    Tool,
}

pub struct TaggedOutput<D: ToolDialect> {
    received: usize,
    spans: Vec<Option<std::ops::Range<usize>>>,
    reasoning_enabled: bool,
    initial_close: bool,
    mode: Mode,
    structured: bool,
    outer_tool_pending: bool,
    buffer: String,
    trim_newlines: usize,
    tools: Arc<[Value]>,
    reasoning: ThinkReasoning<D>,
    tool: Option<TaggedTools<D>>,
    calls: usize,
}
impl<D: ToolDialect> TaggedOutput<D> {
    pub fn with_reasoning(
        mut self,
        syntax: Option<&crate::model::contract::ReasoningParser>,
    ) -> Self {
        self.reasoning_enabled = syntax.is_some();
        if let Some(syntax) = syntax {
            self.reasoning.open.clone_from(&syntax.open);
            self.reasoning.close.clone_from(&syntax.close);
            self.initial_close = matches!(self.mode, Mode::Content)
                && syntax.kind == crate::model::contract::ReasoningParserKind::MmThink;
        }
        self.reasoning.trim_wrap = syntax.is_some_and(|s| s.wrapping_newlines > 0);
        self.reasoning.tool_ends_reasoning =
            syntax.is_some_and(|s| s.terminal_markers.iter().any(|m| m == D::OPEN));
        self
    }
}
impl<D: ToolDialect> ModelOutput for TaggedOutput<D> {
    type Config = OutputConfig;
    fn new(config: OutputConfig) -> Self {
        let tools = !config.tools.is_empty();
        let definitions: Arc<[Value]> = config.tools.into();
        Self {
            received: 0,
            spans: vec![],
            reasoning_enabled: true,
            initial_close: false,
            mode: match config.start {
                OutputStart::Content => Mode::Content,
                OutputStart::Reasoning => Mode::Reasoning,
                OutputStart::ToolBody => Mode::Tool,
            },
            structured: config.structured,
            outer_tool_pending: false,
            buffer: String::new(),
            trim_newlines: 0,
            tools: definitions.clone(),
            reasoning: ThinkReasoning {
                buffer: String::new(),
                open: "<think>".into(),
                close: "</think>".into(),
                trim_newlines: 0,
                tools,
                trim_wrap: D::TRIM_NEWLINES,
                tool_ends_reasoning: D::TOOL_ENDS_REASONING,
                dialect: std::marker::PhantomData,
            },
            tool: (config.start == OutputStart::ToolBody).then(|| TaggedTools {
                dialect: std::marker::PhantomData,
                buffer: String::new(),
                tools: definitions,
                index: 0,
            }),
            calls: 0,
        }
    }
    #[cfg(feature = "server")]
    fn grammar_envelope(config: &Self::Config) -> Option<crate::grammar::OutputEnvelope> {
        Some(crate::grammar::OutputEnvelope {
            reasoning_end: (config.start == OutputStart::Reasoning).then_some("</think>"),
            tool: (!config.tools.is_empty()).then_some((D::OPEN, D::CLOSE)),
            function: D::FUNCTION,
            tool_ends_reasoning: D::TOOL_ENDS_REASONING,
            response: None,
            tool_body_start: config.start == OutputStart::ToolBody,
        })
    }
    fn content_spans(&self) -> Vec<Option<std::ops::Range<usize>>> {
        self.spans.clone()
    }
    fn consumed_bytes(&self) -> usize {
        // Nested reasoning/tool buffers cannot become content, even before their close marker.
        self.received - self.buffer.len()
    }
    fn push(&mut self, input: Input<'_>) -> Result<Vec<Event>> {
        self.spans.clear();
        if let Input::Token(_) = input {
            return match self.mode {
                Mode::Reasoning => self.reasoning.push(input).map(|p| p.events),
                Mode::Tool => self.tool.as_mut().unwrap().push(input).map(|p| p.events),
                _ => Ok(vec![]),
            };
        }
        let Input::Text(text) = input else {
            unreachable!()
        };
        self.received += text.len();
        self.buffer.push_str(text);
        let mut output = Vec::new();
        loop {
            while self.trim_newlines > 0 {
                if self.buffer.is_empty() {
                    return Ok(output);
                }
                if self.buffer.starts_with('\n') {
                    self.buffer.remove(0);
                    self.trim_newlines -= 1;
                } else {
                    self.trim_newlines = 0;
                }
            }
            match self.mode {
                Mode::Verbatim => {
                    let start = self.received - self.buffer.len();
                    let text = std::mem::take(&mut self.buffer);
                    if !text.is_empty() {
                        self.spans.push(Some(start..start + text.len()));
                        output.push(Event::Content(text));
                    }
                    return Ok(output);
                }
                Mode::Reasoning | Mode::Tool => {
                    let text = std::mem::take(&mut self.buffer);
                    let reasoning = matches!(self.mode, Mode::Reasoning);
                    let progress = if reasoning {
                        self.reasoning.push(Input::Text(&text))?
                    } else {
                        self.tool.as_mut().unwrap().push(Input::Text(&text))?
                    };
                    self.calls += progress
                        .events
                        .iter()
                        .filter(|event| matches!(event, Event::ToolBegin { .. }))
                        .count();
                    output.extend(progress.events);
                    let Some(remainder) = progress.remainder else {
                        return Ok(output);
                    };
                    self.buffer = remainder;
                    self.mode = Mode::Content;
                    if reasoning {
                        self.trim_newlines = if self.reasoning.trim_wrap { 2 } else { 0 };
                    } else {
                        self.tool = None;
                        self.outer_tool_pending = D::FUNCTION.is_some();
                    }
                }
                Mode::Content => {
                    if self.initial_close {
                        if self.reasoning.close.starts_with(&self.buffer)
                            && self.buffer.len() < self.reasoning.close.len()
                        {
                            return Ok(output);
                        }
                        self.initial_close = false;
                        if self.buffer.starts_with(self.reasoning.close.as_str()) {
                            self.buffer.drain(..self.reasoning.close.len());
                            continue;
                        }
                    }
                    if self.outer_tool_pending {
                        let text = self.buffer.trim_start();
                        if text.is_empty()
                            || (D::CLOSE.starts_with(text) && text.len() < D::CLOSE.len())
                            || D::FUNCTION.is_some_and(|(open, _)| {
                                open.starts_with(text) && text.len() < open.len()
                            })
                        {
                            return Ok(output);
                        }
                        self.outer_tool_pending = false;
                        if text.starts_with(D::CLOSE) {
                            let at = self.buffer.find(D::CLOSE).unwrap();
                            self.buffer.drain(..at + D::CLOSE.len());
                            continue;
                        }
                        if D::FUNCTION.is_some_and(|(open, _)| text.starts_with(open)) {
                            let whitespace = self.buffer.len() - text.len();
                            self.buffer.drain(..whitespace);
                        }
                    }
                    if self.structured {
                        let text = self.buffer.trim_start();
                        if text.is_empty() {
                            return Ok(output);
                        }
                        let markers = [
                            D::OPEN,
                            D::CLOSE,
                            D::FUNCTION.map_or(D::OPEN, |(open, _)| open),
                            self.reasoning.close.as_str(),
                        ];
                        let control = markers
                            .iter()
                            .filter(|marker| !marker.is_empty())
                            .any(|marker| text.starts_with(marker) || marker.starts_with(text));
                        if !control {
                            self.mode = Mode::Verbatim;
                            continue;
                        }
                    }
                    if self.reasoning.tool_ends_reasoning {
                        if let Some(at) = self.buffer.find(self.reasoning.close.as_str()) {
                            let earlier = [
                                self.reasoning.open.as_str(),
                                D::OPEN,
                                D::FUNCTION.map_or(D::OPEN, |(open, _)| open),
                            ]
                            .iter()
                            .any(|marker| self.buffer.find(marker).is_some_and(|start| start < at));
                            if !earlier {
                                let start = self.received - self.buffer.len();
                                let prefix: String = self.buffer.drain(..at).collect();
                                self.buffer.drain(..self.reasoning.close.len());
                                if !prefix.is_empty() {
                                    self.spans.push(Some(start..start + prefix.len()));
                                    output.push(Event::Content(prefix));
                                }
                                continue;
                            }
                        }
                    }
                    let mut marker = self
                        .reasoning_enabled
                        .then(|| self.buffer.find(self.reasoning.open.as_str()))
                        .flatten()
                        .map(|at| (at, false));
                    let mut bare_function = false;
                    if !self.tools.is_empty() {
                        if let Some(at) = self.buffer.find(D::OPEN) {
                            if marker.is_none_or(|(first, _)| at < first) {
                                marker = Some((at, true));
                            }
                        }
                        if let Some((open, _)) = D::FUNCTION {
                            if let Some(at) = self.buffer.find(open) {
                                if marker.is_none_or(|(first, _)| at < first) {
                                    marker = Some((at, true));
                                    bare_function = true;
                                }
                            }
                        }
                    }
                    if let Some((at, tool)) = marker {
                        let prefix = &self.buffer[..at];
                        let prefix = if tool && D::TRIM_NEWLINES {
                            if self.calls == 0 {
                                prefix.strip_suffix("\n\n").unwrap_or(prefix)
                            } else {
                                prefix.strip_suffix('\n').unwrap_or(prefix)
                            }
                        } else {
                            prefix
                        };
                        if !prefix.is_empty() {
                            let start = self.received - self.buffer.len();
                            self.spans.push(Some(start..start + prefix.len()));
                            output.push(Event::Content(prefix.to_owned()));
                        }
                        self.buffer.drain(
                            ..at + if bare_function {
                                0
                            } else if tool {
                                D::OPEN.len()
                            } else {
                                self.reasoning.open.len()
                            },
                        );
                        self.mode = if tool { Mode::Tool } else { Mode::Reasoning };
                        if tool {
                            self.tool = Some(TaggedTools {
                                dialect: std::marker::PhantomData,
                                buffer: String::new(),
                                tools: self.tools.clone(),
                                index: self.calls,
                            });
                        } else {
                            self.reasoning = ThinkReasoning {
                                dialect: std::marker::PhantomData,
                                buffer: String::new(),
                                open: self.reasoning.open.clone(),
                                close: self.reasoning.close.clone(),
                                trim_newlines: u8::from(self.reasoning.trim_wrap) as usize,
                                trim_wrap: self.reasoning.trim_wrap,
                                tool_ends_reasoning: self.reasoning.tool_ends_reasoning,
                                tools: !self.tools.is_empty(),
                            };
                        }
                    } else {
                        let newline = format!("\n{}", D::OPEN);
                        let double_newline = format!("\n\n{}", D::OPEN);
                        let markers: &[&str] = match (self.reasoning_enabled, self.tools.is_empty())
                        {
                            (true, true) => &[self.reasoning.open.as_str()],
                            (true, false) => &[
                                self.reasoning.open.as_str(),
                                D::OPEN,
                                &double_newline,
                                &newline,
                            ],
                            (false, false) => &[D::OPEN, &double_newline, &newline],
                            (false, true) => &[],
                        };
                        let mut hold = partial_marker(&self.buffer, markers);
                        if self.reasoning.tool_ends_reasoning {
                            hold = hold.max(partial_marker(
                                &self.buffer,
                                &[self.reasoning.close.as_str()],
                            ));
                        }
                        if !self.tools.is_empty()
                            && let Some((open, _)) = D::FUNCTION
                        {
                            hold = hold.max(partial_marker(&self.buffer, &[open, D::CLOSE]));
                        }
                        let start = self.received - self.buffer.len();
                        let text: String = self.buffer.drain(..self.buffer.len() - hold).collect();
                        if !text.is_empty() {
                            self.spans.push(Some(start..start + text.len()));
                            output.push(Event::Content(text));
                        }
                        return Ok(output);
                    }
                }
            }
        }
    }
    fn finish(&mut self, cause: Termination) -> Result<Vec<Event>> {
        if matches!(cause, Termination::Cancelled | Termination::ExecutionError) {
            self.buffer.clear();
            return match self.mode {
                Mode::Reasoning => self.reasoning.finish(cause),
                Mode::Tool => self.tool.as_mut().unwrap().finish(cause),
                _ => Ok(vec![]),
            };
        }
        let mut output = self.push(Input::Text(""))?;
        match self.mode {
            Mode::Reasoning => output.extend(self.reasoning.finish(cause)?),
            Mode::Tool => output.extend(self.tool.as_mut().unwrap().finish(cause)?),
            _ => {
                let start = self.received - self.buffer.len();
                let text = std::mem::take(&mut self.buffer);
                if !text.is_empty() {
                    self.spans.push(Some(start..start + text.len()));
                    output.push(Event::Content(text));
                }
            }
        }
        Ok(output)
    }
}

fn partial_marker(text: &str, markers: &[&str]) -> usize {
    markers
        .iter()
        .flat_map(|marker| {
            marker
                .char_indices()
                .map(|(index, _)| index)
                .filter(|index| *index > 0)
                .rev()
                .filter(|length| text.ends_with(&marker[..*length]))
                .take(1)
        })
        .max()
        .unwrap_or(0)
}

pub(crate) fn parameter_value(raw: &str, schema: Option<&Value>) -> Result<Value> {
    let Some(schema) = schema else {
        return Ok(Value::String(raw.into()));
    };
    let types = schema.get("type");
    if types == Some(&Value::String("string".into())) {
        return Ok(Value::String(raw.into()));
    }
    if types.is_none() {
        for key in ["anyOf", "oneOf"] {
            if let Some(branches) = schema.get(key).and_then(Value::as_array) {
                for branch in branches {
                    if let Ok(value) = parameter_value(raw, Some(branch)) {
                        return Ok(value);
                    }
                }
                return Err("tool parameter does not match any declared type".into());
            }
        }
        return Ok(Value::String(raw.into()));
    }
    let kinds: Vec<_> = match types {
        Some(Value::String(kind)) => vec![kind.as_str()],
        Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).collect(),
        _ => return Err("invalid tool parameter type declaration".into()),
    };
    let parsed: std::result::Result<Value, _> = serde_json::from_str(raw);
    for kind in kinds {
        if kind == "string" {
            return Ok(Value::String(raw.into()));
        }
        if let Ok(value) = &parsed {
            let valid = match kind {
                "null" => value.is_null(),
                "boolean" => value.is_boolean(),
                "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
                "number" => value.is_number(),
                "array" => value.is_array(),
                "object" => value.is_object(),
                _ => false,
            };
            if valid {
                return Ok(value.clone());
            }
        }
    }
    Err("tool parameter value does not match its declared JSON type".into())
}
