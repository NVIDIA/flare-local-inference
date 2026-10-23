//! Translate the official DeepSeek parser's actions into shared output events.

use super::{
    Event, Input, ModelOutput, Termination,
    tagged::{OutputConfig, OutputStart},
};
use crate::Result;
use deepseek_recipe::stream::state_machine::{
    OutputAction, OutputActionSegment, ParsingOptions, ReasoningStage, StateMachine,
};
use serde::de::{Deserializer, MapAccess, Visitor};
use serde_json::Value;
use std::collections::HashSet;

const BUFFER_LIMIT: usize = 1024 * 1024;

#[derive(Default)]
struct Call {
    name: String,
    arguments: String,
}

pub struct DsmlOutput<const SPACED: bool> {
    consumed: usize,
    spans: Vec<Option<std::ops::Range<usize>>>,
    machine: Option<StateMachine>,
    pending: String,
    last_action: OutputAction,
    call: Option<Call>,
    calls: usize,
    structured: bool,
    verbatim: bool,
}

impl<const SPACED: bool> DsmlOutput<SPACED> {
    fn apply(&mut self, actions: Vec<OutputActionSegment>, events: &mut Vec<Event>) -> Result<()> {
        let mut consumed = 0;
        for segment in actions {
            let end = consumed + segment.len;
            let source = self
                .pending
                .get(consumed..end)
                .ok_or("DeepSeek parser returned an invalid source span")?;
            let repeated = self.last_action == segment.action;
            self.last_action = segment.action;
            match segment.action {
                OutputAction::Raw => {
                    if !source.is_empty() {
                        self.spans.push(Some(self.consumed + consumed..self.consumed + end));
                        events.push(Event::Content(source.to_owned()));
                        self.verbatim = self.structured && !source.trim().is_empty();
                    }
                }
                OutputAction::Reasoning => {
                    if !source.is_empty() {
                        events.push(Event::Reasoning(source.to_owned()));
                    }
                }
                OutputAction::ToolCallBegin if !repeated => {
                    if self.call.is_some() {
                        return Err(
                            "DeepSeek parser began a tool call before completing its predecessor"
                                .into(),
                        );
                    }
                    self.call = Some(Call::default());
                }
                OutputAction::ToolName => {
                    self.call
                        .as_mut()
                        .ok_or("DeepSeek tool name has no call")?
                        .name
                        .push_str(source);
                }
                OutputAction::ToolNameEnd if !repeated => {
                    let name = &self
                        .call
                        .as_ref()
                        .ok_or("DeepSeek tool name has no call")?
                        .name;
                    if name.is_empty() {
                        return Err("tool function name is empty".into());
                    }
                }
                OutputAction::LabelToolCallArguments { label, .. } if !repeated => {
                    self.call
                        .as_mut()
                        .ok_or("DeepSeek tool arguments have no call")?
                        .arguments
                        .push_str(label);
                }
                OutputAction::RawToolCallArguments { string } => {
                    let arguments = &mut self
                        .call
                        .as_mut()
                        .ok_or("DeepSeek tool arguments have no call")?
                        .arguments;
                    if string {
                        let escaped = serde_json::to_string(source)
                            .expect("serializing a string cannot fail");
                        arguments.push_str(&escaped[1..escaped.len() - 1]);
                    } else {
                        arguments.push_str(source);
                    }
                }
                OutputAction::ToolCallArgumentsEnd { output } if !repeated => {
                    let mut call = self
                        .call
                        .take()
                        .ok_or("DeepSeek tool arguments have no call")?;
                    call.arguments.push_str(output.unwrap_or_default());
                    validate_arguments(&call.arguments)?;
                    let index = self.calls;
                    events.extend([
                        Event::ToolBegin {
                            index,
                            name: call.name,
                            id: None,
                        },
                        Event::ToolArguments {
                            index,
                            text: call.arguments,
                        },
                        Event::ToolEnd { index },
                    ]);
                    self.calls += 1;
                }
                OutputAction::Label(label) if !repeated => {
                    self.spans.push(None);
                    events.push(Event::Content(label.into()))
                }
                OutputAction::ToSpace => {
                    self.spans.push(Some(self.consumed + consumed..self.consumed + end));
                    events.push(Event::Content(" ".into()));
                },
                _ => {}
            }
            if self.call.as_ref().is_some_and(|call| {
                call.name.len().saturating_add(call.arguments.len()) > BUFFER_LIMIT
            }) {
                return Err("DeepSeek tool call exceeds the 1 MiB buffer bound".into());
            }
            consumed = end;
            if self.verbatim {
                break;
            }
        }
        self.pending.drain(..consumed);
        self.consumed += consumed;
        if self.verbatim && !self.pending.is_empty() {
            self.spans.push(Some(self.consumed..self.consumed + self.pending.len()));
            self.consumed += self.pending.len();
            events.push(Event::Content(std::mem::take(&mut self.pending)));
        }
        if self.pending.len() > BUFFER_LIMIT {
            return Err("DeepSeek parser exceeds the 1 MiB buffer bound".into());
        }
        Ok(())
    }

    fn feed(&mut self, text: &str, events: &mut Vec<Event>) -> Result<()> {
        self.pending.push_str(text);
        let actions = self.machine.as_mut().unwrap().feed(text);
        self.apply(actions, events)
    }
}

impl<const SPACED: bool> ModelOutput for DsmlOutput<SPACED> {
    type Config = OutputConfig;

    fn new(config: OutputConfig) -> Self {
        Self {
            consumed: 0,
            spans: vec![],
            machine: Some(StateMachine::new(ParsingOptions {
                parse_tool_calls: !config.tools.is_empty(),
                tool_call_initial_stage: config.start == OutputStart::ToolBody,
                // Helios grammars also support scalar JSON and non-JSON languages.
                parse_json_output: false,
                reasoning_initial_stage: (config.start == OutputStart::Reasoning)
                    .then_some(ReasoningStage::Start),
                stop_sequences: vec![],
            })),
            pending: String::new(),
            last_action: OutputAction::Skip,
            call: None,
            calls: 0,
            structured: config.structured,
            verbatim: false,
        }
    }

    fn grammar_envelope(config: &Self::Config) -> Option<crate::grammar::OutputEnvelope> {
        Some(crate::grammar::OutputEnvelope {
            reasoning_end: (config.start == OutputStart::Reasoning).then_some("</think>"),
            tool: (!config.tools.is_empty()).then_some(if SPACED {
                ("<｜DSML｜ calls>", "</｜DSML｜ calls>")
            } else {
                ("<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>")
            }),
            function: None,
            tool_ends_reasoning: false,
            tool_starts_line: false,
            response: None,
            tool_body_start: config.start == OutputStart::ToolBody,
        })
    }

    fn content_spans(&self) -> Vec<Option<std::ops::Range<usize>>> {
        self.spans.clone()
    }
    fn consumed_bytes(&self) -> usize {
        self.consumed
    }
    fn push(&mut self, input: Input<'_>) -> Result<Vec<Event>> {
        self.spans.clear();
        if self.machine.is_none() {
            return Err("output presentation is finished".into());
        }
        let Input::Text(text) = input else {
            return Ok(vec![]);
        };
        let mut events = Vec::new();
        let mut rest = text;
        // Stop interpreting markup as soon as a grammar-constrained answer begins.
        while self.structured && !self.verbatim && !rest.is_empty() {
            let length = rest.chars().next().unwrap().len_utf8();
            self.feed(&rest[..length], &mut events)?;
            rest = &rest[length..];
        }
        if !rest.is_empty() {
            if self.verbatim {
                self.spans.push(Some(self.consumed..self.consumed + rest.len()));
                self.consumed += rest.len();
                events.push(Event::Content(rest.to_owned()));
            } else {
                self.feed(rest, &mut events)?;
            }
        }
        Ok(events)
    }

    fn finish(&mut self, cause: Termination) -> Result<Vec<Event>> {
        self.spans.clear();
        let machine = self
            .machine
            .take()
            .ok_or("output presentation is finished")?;
        if matches!(cause, Termination::Cancelled | Termination::ExecutionError) {
            self.pending.clear();
            self.call = None;
            return Ok(vec![]);
        }
        let mut events = Vec::new();
        if !self.verbatim {
            self.apply(machine.finish(), &mut events)?;
        }
        if self.call.take().is_some() && cause == Termination::Eos {
            return Err("incomplete DeepSeek tool call at end of response".into());
        }
        Ok(events)
    }
}

fn validate_arguments(arguments: &str) -> Result<()> {
    struct Arguments;
    impl<'de> Visitor<'de> for Arguments {
        type Value = ();
        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a tool argument object with nonempty, unique keys")
        }
        fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> std::result::Result<(), M::Error> {
            let mut keys = HashSet::new();
            while let Some(key) = map.next_key::<String>()? {
                if key.is_empty() || !keys.insert(key) {
                    return Err(serde::de::Error::custom(
                        "tool argument names must be nonempty and unique",
                    ));
                }
                map.next_value::<Value>()?;
            }
            Ok(())
        }
    }
    let mut decoder = serde_json::Deserializer::from_str(arguments);
    decoder
        .deserialize_map(Arguments)
        .map_err(|error| format!("invalid DeepSeek tool arguments: {error}"))?;
    decoder
        .end()
        .map_err(|error| format!("invalid DeepSeek tool arguments: {error}"))?;
    Ok(())
}

#[cfg(test)]
mod deepseek41_syntax_tests {
    use super::*;
    use crate::output::tagged::OutputConfig;
    type DeepSeekOutput = DsmlOutput<true>;
    use crate::output::{Input, ModelOutput, Termination};
    use crate::server::output::Accumulator;
    use serde_json::json;

    #[test]
    fn every_utf8_split_preserves_multiple_calls_and_typed_values() {
        let text = "thought</think><｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">東京</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"lookup\"><｜DSML｜ parameter name=\"n\" string=\"false\">3</｜DSML｜ parameter></｜DSML｜ invoke>\n</｜DSML｜ calls>";
        for split in text.char_indices().map(|(offset, _)| offset) {
            let mut parser = DeepSeekOutput::new(OutputConfig {
                structured: false,
                start: crate::output::tagged::OutputStart::Reasoning,
                tools: vec![json!({"type":"function", "function":{"name":"lookup"}})],
            });
            let mut output = Accumulator::new("request".into());
            output
                .apply(parser.push(Input::Text(&text[..split])).unwrap())
                .unwrap();
            output
                .apply(parser.push(Input::Text(&text[split..])).unwrap())
                .unwrap();
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["reasoning_content"], "thought");
            assert_eq!(message["tool_calls"].as_array().unwrap().len(), 2);
            assert_eq!(
                message["tool_calls"][0]["function"]["arguments"],
                "{\"city\": \"東京\"}"
            );
            assert_eq!(
                message["tool_calls"][1]["function"]["arguments"],
                "{\"n\": 3}"
            );
        }
    }
}

#[cfg(test)]
mod deepseek4_syntax_tests {
    use super::*;
    use crate::output::tagged::OutputConfig;
    type DeepSeekOutput = DsmlOutput<false>;
    use crate::output::{Input, ModelOutput, Termination};
    use crate::server::output::Accumulator;
    use serde_json::json;

    #[test]
    fn every_utf8_split_preserves_multiple_calls_and_typed_values() {
        let text = "thought</think><｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"lookup\">\n<｜DSML｜parameter name=\"city\" string=\"true\">東京</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"lookup\"><｜DSML｜parameter name=\"n\" string=\"false\">3</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        for split in text.char_indices().map(|(offset, _)| offset) {
            let mut parser = DeepSeekOutput::new(OutputConfig {
                structured: false,
                start: crate::output::tagged::OutputStart::Reasoning,
                tools: vec![json!({"type":"function", "function":{"name":"lookup"}})],
            });
            let mut output = Accumulator::new("request".into());
            output
                .apply(parser.push(Input::Text(&text[..split])).unwrap())
                .unwrap();
            output
                .apply(parser.push(Input::Text(&text[split..])).unwrap())
                .unwrap();
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["reasoning_content"], "thought");
            assert_eq!(message["tool_calls"].as_array().unwrap().len(), 2);
            assert_eq!(
                message["tool_calls"][0]["function"]["arguments"],
                "{\"city\": \"東京\"}"
            );
            assert_eq!(
                message["tool_calls"][1]["function"]["arguments"],
                "{\"n\": 3}"
            );
        }
    }
}

#[cfg(test)]
mod prefixed_tool_tests {
    use super::*;
    use crate::output::tagged::{OutputConfig, OutputStart};
    use crate::output::{Input, ModelOutput, Termination};
    use crate::server::output::Accumulator;
    use serde_json::json;

    fn check<const SPACED: bool>() {
        let separator = if SPACED { " " } else { "" };
        let close = if SPACED {
            "</｜DSML｜ calls>"
        } else {
            "</｜DSML｜tool_calls>"
        };
        let text = format!(
            "\n<｜DSML｜{separator}invoke name=\"lookup\"><｜DSML｜{separator}parameter name=\"city\" string=\"true\">東京</｜DSML｜{separator}parameter></｜DSML｜{separator}invoke>{close}"
        );
        for split in text
            .char_indices()
            .map(|(offset, _)| offset)
            .chain([text.len()])
        {
            let mut parser = DsmlOutput::<SPACED>::new(OutputConfig {
                structured: true,
                start: OutputStart::ToolBody,
                tools: vec![json!({"type":"function", "function":{"name":"lookup"}})],
            });
            let mut output = Accumulator::new("request".into());
            output
                .apply(parser.push(Input::Text(&text[..split])).unwrap())
                .unwrap();
            output
                .apply(parser.push(Input::Text(&text[split..])).unwrap())
                .unwrap();
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["tool_calls"][0]["function"]["name"], "lookup");
            assert_eq!(
                message["tool_calls"][0]["function"]["arguments"],
                "{\"city\": \"東京\"}"
            );
            assert!(message.get("reasoning_content").is_none_or(Value::is_null));
        }
    }

    #[test]
    fn compact_tool_body_accepts_every_utf8_split() {
        check::<false>();
    }

    #[test]
    fn spaced_tool_body_accepts_every_utf8_split() {
        check::<true>();
    }
}
