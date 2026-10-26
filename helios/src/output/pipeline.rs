//! Independent prepared syntax selection for streamed reasoning and tool calls.

use super::{
    Event, Input, ModelOutput, Termination,
    tagged::{OutputConfig as TaggedConfig, TaggedOutput},
};
use crate::{
    Result,
    model::contract::{FrontendContract, ReasoningParserKind, ToolParserKind},
};

#[derive(Clone)]
pub struct OutputConfig {
    pub syntax: FrontendContract,
    pub content: TaggedConfig,
    /// Delimiters that occupy one tokenizer token and count only when generated as it.
    pub control_tokens: Vec<(u32, String)>,
}

pub enum OutputPipeline {
    FunctionTags(TaggedOutput<super::chatml::FunctionTags<true>>),
    CompactFunctionTags(TaggedOutput<super::chatml::FunctionTags<false>>),
    ArgumentTags(TaggedOutput<super::argument_tags::ArgumentTags>),
    Dsml(super::dsml::DsmlOutput<false>),
    SpacedDsml(super::dsml::DsmlOutput<true>),
    Xtml(super::xtml::XtmlOutput),
    MiniMaxXml(TaggedOutput<super::minimax_xml::MiniMaxXml>),
    GemmaCall(TaggedOutput<super::gemma_call::GemmaCall>),
    Text(TaggedOutput<PlainText>),
}

pub struct PlainText;
impl super::tagged::ToolDialect for PlainText {
    const NAME: &'static str = "plain text";
    const TRIM_NEWLINES: bool = false;
    const OPEN: &'static str = "";
    const CLOSE: &'static str = "";
    fn parse(_: &str, _: &[serde_json::Value]) -> Result<Vec<(String, String)>> {
        Err("tool parsing is not configured".into())
    }
}

macro_rules! dispatch {
    ($self:expr, $parser:ident => $body:expr) => {
        match $self {
            OutputPipeline::FunctionTags($parser) => $body,
            OutputPipeline::CompactFunctionTags($parser) => $body,
            OutputPipeline::ArgumentTags($parser) => $body,
            OutputPipeline::Dsml($parser) => $body,
            OutputPipeline::SpacedDsml($parser) => $body,
            OutputPipeline::Xtml($parser) => $body,
            OutputPipeline::MiniMaxXml($parser) => $body,
            OutputPipeline::GemmaCall($parser) => $body,
            OutputPipeline::Text($parser) => $body,
        }
    };
}

impl OutputPipeline {
    pub fn new(config: OutputConfig) -> Self {
        let reasoning = config.syntax.reasoning.as_ref();
        match config
            .syntax
            .tools
            .as_ref()
            .map(|t| (t.kind, t.wrapping_newlines, t.tag_separator.as_str()))
        {
            Some((ToolParserKind::QwenCoder, 0, _)) => Self::CompactFunctionTags(
                TaggedOutput::new(config.content).with_reasoning(reasoning, &config.control_tokens),
            ),
            Some((ToolParserKind::QwenCoder, _, _)) => Self::FunctionTags(
                TaggedOutput::new(config.content).with_reasoning(reasoning, &config.control_tokens),
            ),
            Some((ToolParserKind::GlmArgumentTags, _, _)) => Self::ArgumentTags(
                TaggedOutput::new(config.content).with_reasoning(reasoning, &config.control_tokens),
            ),
            Some((ToolParserKind::Dsml, _, " ")) => {
                Self::SpacedDsml(super::dsml::DsmlOutput::new(config.content))
            }
            Some((ToolParserKind::Dsml, _, _)) => {
                Self::Dsml(super::dsml::DsmlOutput::new(config.content))
            }
            Some((ToolParserKind::Xtml, _, _)) => Self::Xtml(super::xtml::XtmlOutput::new(
                config.content,
                config.control_tokens,
            )),
            Some((ToolParserKind::MiniMaxXml, _, _)) => Self::MiniMaxXml(
                TaggedOutput::new(config.content).with_reasoning(reasoning, &config.control_tokens),
            ),
            Some((ToolParserKind::GemmaCall, _, _)) => Self::GemmaCall(
                TaggedOutput::new(config.content).with_reasoning(reasoning, &config.control_tokens),
            ),
            None => Self::Text(
                TaggedOutput::new(config.content).with_reasoning(reasoning, &config.control_tokens),
            ),
        }
    }

    /// Whether the parser `new` selects gates reasoning delimiters by token identity.
    /// DSML recognizes its markup in the recipe crate's text state machine, and XTML
    /// carries its own control tokens.
    pub fn gates_reasoning_tokens(syntax: &FrontendContract) -> bool {
        match syntax.tools.as_ref().map(|t| t.kind) {
            Some(ToolParserKind::Dsml | ToolParserKind::Xtml) => false,
            Some(
                ToolParserKind::QwenCoder
                | ToolParserKind::GlmArgumentTags
                | ToolParserKind::MiniMaxXml
                | ToolParserKind::GemmaCall,
            )
            | None => true,
        }
    }

    pub fn grammar_envelope(config: &OutputConfig) -> Option<crate::grammar::OutputEnvelope> {
        let mut envelope = match config
            .syntax
            .tools
            .as_ref()
            .map(|t| (t.kind, t.wrapping_newlines, t.tag_separator.as_str()))
        {
            Some((ToolParserKind::QwenCoder, 0, _)) => {
                TaggedOutput::<super::chatml::FunctionTags<false>>::grammar_envelope(
                    &config.content,
                )
            }
            Some((ToolParserKind::QwenCoder, _, _)) => {
                TaggedOutput::<super::chatml::FunctionTags<true>>::grammar_envelope(&config.content)
            }
            Some((ToolParserKind::GlmArgumentTags, _, _)) => {
                TaggedOutput::<super::argument_tags::ArgumentTags>::grammar_envelope(
                    &config.content,
                )
            }
            Some((ToolParserKind::Dsml, _, " ")) => {
                super::dsml::DsmlOutput::<true>::grammar_envelope(&config.content)
            }
            Some((ToolParserKind::Dsml, _, _)) => {
                super::dsml::DsmlOutput::<false>::grammar_envelope(&config.content)
            }
            Some((ToolParserKind::Xtml, _, _)) => Some(crate::grammar::OutputEnvelope {
                reasoning_end: config
                    .content
                    .start
                    .eq(&super::tagged::OutputStart::Reasoning)
                    .then_some("<|close|>think<|sep|><|open|>response<|sep|>"),
                tool: (!config.content.tools.is_empty())
                    .then_some((super::xtml::TOOLS_OPEN, super::xtml::TOOLS_CLOSE)),
                function: None,
                tool_ends_reasoning: false,
                response: Some(("", super::xtml::RESPONSE_CLOSE)),
                tool_body_start: false,
            }),
            Some((ToolParserKind::MiniMaxXml, _, _)) => {
                TaggedOutput::<super::minimax_xml::MiniMaxXml>::grammar_envelope(&config.content)
            }
            Some((ToolParserKind::GemmaCall, _, _)) => {
                TaggedOutput::<super::gemma_call::GemmaCall>::grammar_envelope(&config.content)
            }
            None => Some(crate::grammar::OutputEnvelope {
                reasoning_end: (config.content.start == super::tagged::OutputStart::Reasoning)
                    .then_some("</think>"),
                tool: None,
                function: None,
                tool_ends_reasoning: false,
                response: None,
                tool_body_start: false,
            }),
        }?;
        if config.content.start == super::tagged::OutputStart::Reasoning {
            match config.syntax.reasoning.as_ref().map(|r| r.kind) {
                Some(ReasoningParserKind::MmThink) => envelope.reasoning_end = Some("</mm:think>"),
                Some(ReasoningParserKind::ChannelThought) => {
                    envelope.reasoning_end = Some("<channel|>")
                }
                _ => {}
            }
        }
        envelope.tool_ends_reasoning = config.syntax.reasoning.as_ref().is_some_and(|r| {
            envelope
                .tool
                .is_some_and(|(open, _)| r.terminal_markers.iter().any(|m| m == open))
        });
        Some(envelope)
    }

    pub fn push(&mut self, input: Input<'_>) -> Result<Vec<Event>> {
        dispatch!(self, parser => parser.push(input))
    }
    pub fn finish(&mut self, cause: Termination) -> Result<Vec<Event>> {
        dispatch!(self, parser => parser.finish(cause))
    }
}

impl ModelOutput for OutputPipeline {
    type Config = OutputConfig;
    fn new(config: Self::Config) -> Self {
        OutputPipeline::new(config)
    }
    fn grammar_envelope(config: &Self::Config) -> Option<crate::grammar::OutputEnvelope> {
        OutputPipeline::grammar_envelope(config)
    }
    fn content_spans(&self) -> Vec<Option<std::ops::Range<usize>>> {
        dispatch!(self, parser => parser.content_spans())
    }
    fn consumed_bytes(&self) -> usize {
        dispatch!(self, parser => parser.consumed_bytes())
    }
    fn push(&mut self, input: Input<'_>) -> Result<Vec<Event>> {
        OutputPipeline::push(self, input)
    }
    fn finish(&mut self, cause: Termination) -> Result<Vec<Event>> {
        OutputPipeline::finish(self, cause)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::contract::{ReasoningParser, ReasoningParserKind, ToolParser};
    use crate::server::output::Accumulator;
    use serde_json::json;

    fn run(
        syntax: FrontendContract,
        text: &str,
        tools: Vec<serde_json::Value>,
        structured: bool,
    ) -> serde_json::Value {
        let mut parser = OutputPipeline::new(OutputConfig {
            syntax,
            control_tokens: vec![],
            content: TaggedConfig {
                structured,
                start: crate::output::tagged::OutputStart::Content,
                tools,
            },
        });
        let mut output = Accumulator::new("contract".into());
        for character in text.chars() {
            output
                .apply(parser.push(Input::Text(&character.to_string())).unwrap())
                .unwrap();
        }
        output
            .apply(parser.finish(Termination::Eos).unwrap())
            .unwrap();
        output.finish().unwrap();
        output.message()
    }

    #[test]
    fn absent_parsers_preserve_plain_and_structured_content() {
        for structured in [false, true] {
            let text = "{\"literal\":\"<think>content</think><tool_call>\"}";
            assert_eq!(
                run(
                    crate::model::testing::metadata().frontend,
                    text,
                    vec![],
                    structured
                )["content"],
                text
            );
        }
    }

    #[test]
    fn tool_parser_does_not_implicitly_enable_reasoning() {
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.tools = Some(ToolParser {
            kind: ToolParserKind::QwenCoder,
            wrapping_newlines: 0,
            tag_separator: String::new(),
        });
        let tools = vec![json!({"type":"function","function":{"name":"lookup"}})];
        let result = run(
            syntax.clone(),
            "<think>literal</think><tool_call><function=lookup></function></tool_call>",
            tools,
            false,
        );
        assert_eq!(result["content"], "<think>literal</think>");
        assert!(result.get("reasoning_content").is_none());
        assert_eq!(result["tool_calls"][0]["function"]["name"], "lookup");
        syntax.tools = None;
        syntax.reasoning = Some(ReasoningParser {
            kind: ReasoningParserKind::ThinkTags,
            open: "<think>".into(),
            close: "</think>".into(),
            wrapping_newlines: 0,
            terminal_markers: vec![],
        });
        let result = run(syntax, "<think>plan</think>answer", vec![], false);
        assert_eq!(result["reasoning_content"], "plan");
        assert_eq!(result["content"], "answer");
    }
    #[test]
    fn mm_think_and_namespace_tools_stream_at_every_utf8_boundary() {
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.tools = Some(ToolParser {
            kind: ToolParserKind::MiniMaxXml,
            wrapping_newlines: 0,
            tag_separator: String::new(),
        });
        syntax.reasoning = Some(ReasoningParser {
            kind: ReasoningParserKind::MmThink,
            open: "<mm:think>".into(),
            close: "</mm:think>".into(),
            wrapping_newlines: 0,
            terminal_markers: vec![],
        });
        let tools = vec![
            json!({"function":{"name":"lookup","parameters":{"properties":{"n":{"type":"integer"}}}}}),
        ];
        let text = "</mm:think><mm:think>café 世界</mm:think>Searching. ]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name='lookup'>]<]minimax[>[<n>3]<]minimax[>[</n>]<]minimax[>[</invoke>\n]<]minimax[>[<invoke name=lookup>]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>";
        for split in text.char_indices().map(|(at, _)| at) {
            let config = OutputConfig {
                syntax: syntax.clone(),
                control_tokens: vec![],
                content: TaggedConfig {
                    start: crate::output::tagged::OutputStart::Content,
                    structured: false,
                    tools: tools.clone(),
                },
            };
            let mut parser = OutputPipeline::new(config);
            let mut output = Accumulator::new("xml".into());
            for part in [&text[..split], &text[split..]] {
                output
                    .apply(parser.push(Input::Text(part)).unwrap())
                    .unwrap();
            }
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let result = output.message();
            assert_eq!(result["reasoning_content"], "café 世界", "split {split}");
            assert_eq!(result["content"], "Searching. ", "split {split}");
            assert_eq!(
                result["tool_calls"][0]["function"]["arguments"],
                "{\"n\":3}"
            );
            assert_eq!(result["tool_calls"][1]["function"]["arguments"], "{}");
        }
        assert_eq!(
            run(
                syntax.clone(),
                "</mm:think>answer</mm:think>",
                vec![],
                false
            )["content"],
            "answer</mm:think>"
        );
        let config = OutputConfig {
            syntax,
            control_tokens: vec![],
            content: TaggedConfig {
                start: crate::output::tagged::OutputStart::Reasoning,
                structured: true,
                tools: vec![],
            },
        };
        assert_eq!(
            OutputPipeline::grammar_envelope(&config)
                .unwrap()
                .reasoning_end,
            Some("</mm:think>")
        );
        let mut parser = OutputPipeline::new(config.clone());
        assert_eq!(
            parser
                .push(Input::Text("plan</mm:think>{\"ok\":true}"))
                .unwrap(),
            vec![
                Event::Reasoning("plan".into()),
                Event::Content("{\"ok\":true}".into())
            ]
        );
    }

    #[test]
    fn channel_thought_and_gemma_calls_stream_at_every_utf8_boundary() {
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.tools = Some(ToolParser {
            kind: ToolParserKind::GemmaCall,
            wrapping_newlines: 0,
            tag_separator: String::new(),
        });
        syntax.reasoning = Some(ReasoningParser {
            kind: ReasoningParserKind::ChannelThought,
            open: "<|channel>thought".into(),
            close: "<channel|>".into(),
            wrapping_newlines: 1,
            terminal_markers: vec!["<|tool_call>".into()],
        });
        let tools = vec![
            json!({"function":{"name":"lookup","parameters":{"properties":{"n":{"type":"integer"}}}}}),
        ];
        let stream = |text: &str, split: usize, start| {
            let mut parser = OutputPipeline::new(OutputConfig {
                syntax: syntax.clone(),
                control_tokens: vec![],
                content: TaggedConfig {
                    start,
                    structured: false,
                    tools: tools.clone(),
                },
            });
            let mut output = Accumulator::new("gemma".into());
            for part in [&text[..split], &text[split..]] {
                output
                    .apply(parser.push(Input::Text(part)).unwrap())
                    .unwrap();
            }
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            output.message()
        };
        let content = crate::output::tagged::OutputStart::Content;
        let text = "<|channel>thought\ncafé 世界\n<channel|>Checking. <|tool_call>call:lookup{n:3,q:<|\"|>a,b<|\"|>}<tool_call|><|tool_call>call:other{}<tool_call|>";
        for split in text.char_indices().map(|(at, _)| at) {
            let result = stream(text, split, content);
            assert_eq!(result["reasoning_content"], "café 世界", "split {split}");
            assert_eq!(result["content"], "Checking. ", "split {split}");
            assert_eq!(
                result["tool_calls"][0]["function"]["arguments"],
                "{\"n\":3,\"q\":\"a,b\"}"
            );
            assert_eq!(result["tool_calls"][1]["function"]["name"], "other");
            assert_eq!(result["tool_calls"][1]["function"]["arguments"], "{}");
        }
        let text = "<|channel>thought\nplan<|tool_call>call:lookup{n:1}<tool_call|>";
        for split in text.char_indices().map(|(at, _)| at) {
            let result = stream(text, split, content);
            assert_eq!(result["reasoning_content"], "plan", "split {split}");
            assert_eq!(result["tool_calls"][0]["function"]["arguments"], "{\"n\":1}");
        }
        let result = stream("<channel|>answer", 0, content);
        assert_eq!(result["content"], "answer");
        assert!(result.get("reasoning_content").is_none());
        let reasoning = crate::output::tagged::OutputStart::Reasoning;
        let text = "resume\n<channel|>Done.";
        for split in text.char_indices().map(|(at, _)| at) {
            let result = stream(text, split, reasoning);
            assert_eq!(result["reasoning_content"], "resume", "split {split}");
            assert_eq!(result["content"], "Done.", "split {split}");
        }
        let envelope = OutputPipeline::grammar_envelope(&OutputConfig {
            syntax: syntax.clone(),
            control_tokens: vec![],
            content: TaggedConfig {
                start: reasoning,
                structured: true,
                tools: tools.clone(),
            },
        })
        .unwrap();
        assert_eq!(envelope.reasoning_end, Some("<channel|>"));
        assert_eq!(envelope.tool, Some(("<|tool_call>", "<tool_call|>")));
        assert!(envelope.tool_ends_reasoning);
    }

    /// Pieces marked `true` arrive as their delimiter token and then its text.
    /// Feed 0 sends each text after its token, feed 1 delays delimiter text into
    /// the next push, and feed 2 sends one character per push.
    fn delimited(
        syntax: &FrontendContract,
        start: crate::output::tagged::OutputStart,
        tools: &[serde_json::Value],
        pieces: &[(bool, &str)],
        feed: usize,
    ) -> serde_json::Value {
        let reasoning = syntax.reasoning.as_ref().unwrap();
        let control_tokens = [(1, reasoning.open.clone()), (2, reasoning.close.clone())];
        delimited_by(syntax, &control_tokens, start, tools, pieces, feed)
    }

    fn delimited_by(
        syntax: &FrontendContract,
        control_tokens: &[(u32, String)],
        start: crate::output::tagged::OutputStart,
        tools: &[serde_json::Value],
        pieces: &[(bool, &str)],
        feed: usize,
    ) -> serde_json::Value {
        let control_tokens = control_tokens.to_vec();
        let mut parser = OutputPipeline::new(OutputConfig {
            syntax: syntax.clone(),
            control_tokens: control_tokens.clone(),
            content: TaggedConfig {
                structured: false,
                start,
                tools: tools.to_vec(),
            },
        });
        let mut output = Accumulator::new("delimited".into());
        let mut push = |input: Input<'_>| {
            output.apply(parser.push(input).unwrap()).unwrap();
        };
        let mut delayed = String::new();
        for &(token, text) in pieces {
            if token {
                let id = control_tokens.iter().find(|(_, m)| m == text).unwrap().0;
                push(Input::Token(id));
            }
            match feed {
                0 => push(Input::Text(text)),
                1 if token => delayed.push_str(text),
                1 => push(Input::Text(&(std::mem::take(&mut delayed) + text))),
                _ => {
                    for character in text.chars() {
                        push(Input::Text(&character.to_string()));
                    }
                }
            }
        }
        push(Input::Text(&delayed));
        output
            .apply(parser.finish(Termination::Eos).unwrap())
            .unwrap();
        output.finish().unwrap();
        output.message()
    }

    #[test]
    fn finish_releases_text_held_for_a_delimiter_token() {
        use crate::output::tagged::OutputStart;
        let qwen = serde_json::from_str::<crate::model::PreparedContract>(include_str!(
            "../../tests/support/contracts/qwen38.json"
        ))
        .unwrap()
        .frontend;
        let reasoning = qwen.reasoning.as_ref().unwrap();
        let control_tokens = vec![(1, reasoning.open.clone()), (2, reasoning.close.clone())];
        for (start, token, held) in [
            (OutputStart::Reasoning, 2, "</thi"),
            (OutputStart::Content, 1, "<thi"),
        ] {
            for cause in [
                Termination::Eos,
                Termination::OutputLimit,
                Termination::UserStop,
                Termination::Cancelled,
                Termination::ExecutionError,
            ] {
                let mut parser = OutputPipeline::new(OutputConfig {
                    syntax: qwen.clone(),
                    control_tokens: control_tokens.clone(),
                    content: TaggedConfig {
                        structured: false,
                        start,
                        tools: vec![],
                    },
                });
                let mut events = parser.push(Input::Text("plan")).unwrap();
                events.extend(parser.push(Input::Token(token)).unwrap());
                events.extend(parser.push(Input::Text(held)).unwrap());
                events.extend(parser.finish(cause).unwrap());
                let text: String = events
                    .iter()
                    .map(|event| match event {
                        Event::Reasoning(text) if start == OutputStart::Reasoning => text.as_str(),
                        Event::Content(text) if start == OutputStart::Content => text.as_str(),
                        event => panic!("{event:?} for {start:?} {cause:?}"),
                    })
                    .collect();
                let discarded =
                    matches!(cause, Termination::Cancelled | Termination::ExecutionError);
                let expected = if discarded {
                    "plan".to_owned()
                } else {
                    format!("plan{held}")
                };
                assert_eq!(text, expected, "{start:?} {cause:?}");
            }
        }
    }

    #[test]
    fn single_token_reasoning_delimiters_leave_spelled_tags_as_text() {
        use crate::output::tagged::OutputStart;
        let contract = |source: &str| {
            serde_json::from_str::<crate::model::PreparedContract>(source)
                .unwrap()
                .frontend
        };
        let qwen = contract(include_str!("../../tests/support/contracts/qwen38.json"));
        let glm = contract(include_str!("../../tests/support/contracts/glm53.json"));
        let mimo = contract(include_str!("../../tests/support/contracts/mimo_v2.json"));
        let (open, close) = ((true, "<think>"), (true, "</think>"));
        let text = |text| (false, text);
        let cases: &[(&[(bool, &str)], &str, &str)] = &[
            (
                &[
                    text("Closes with </think> first.\nKeep it short."),
                    close,
                    text("The answer is 42."),
                ],
                "Closes with </think> first.\nKeep it short.",
                "The answer is 42.",
            ),
            (
                &[text("Mention <think> here."), close, text("OK")],
                "Mention <think> here.",
                "OK",
            ),
            (
                &[text("plan"), close, text("Write </think> to close.")],
                "plan",
                "Write </think> to close.",
            ),
            (
                &[
                    text("plan"),
                    close,
                    text("Write <think> to open, then text."),
                ],
                "plan",
                "Write <think> to open, then text.",
            ),
            (
                &[
                    text("plan"),
                    close,
                    text("The `<think>` and `</think>` tags mark reasoning."),
                ],
                "plan",
                "The `<think>` and `</think>` tags mark reasoning.",
            ),
            (
                &[
                    text("plan"),
                    close,
                    text("answer "),
                    open,
                    text("more"),
                    close,
                    text("done"),
                ],
                "planmore",
                "answer done",
            ),
        ];
        for syntax in [&qwen, &glm, &mimo] {
            for (pieces, reasoning, content) in cases {
                for feed in 0..3 {
                    let message = delimited(syntax, OutputStart::Reasoning, &[], pieces, feed);
                    assert_eq!(
                        message["reasoning_content"], *reasoning,
                        "{pieces:?} feed {feed}"
                    );
                    assert_eq!(message["content"], *content, "{pieces:?} feed {feed}");
                }
            }
        }
        for feed in 0..3 {
            let message = delimited(
                &qwen,
                OutputStart::Reasoning,
                &[],
                &[
                    text("Closes with </think> first.\n"),
                    close,
                    text("\n\nThe answer is 42."),
                ],
                feed,
            );
            assert_eq!(message["reasoning_content"], "Closes with </think> first.");
            assert_eq!(message["content"], "The answer is 42.");
        }

        let tools = [json!({"type":"function","function":{"name":"lookup"}})];
        let call = "<tool_call><function=lookup></function></tool_call>";
        let implicit = format!("Look up Paris.{call}");
        let tool_cases: &[(&[(bool, &str)], &str, &str, usize)] = &[
            (
                &[open, text("Look up Paris."), close, close, text(call)],
                "Look up Paris.",
                "",
                1,
            ),
            (&[open, text(&implicit)], "Look up Paris.", "", 1),
            (
                &[text("plan"), close, text("Note: </think> stays.")],
                "plan",
                "Note: </think> stays.",
                0,
            ),
            (
                &[text("Use <think> literally."), close, text("ok")],
                "Use <think> literally.",
                "ok",
                0,
            ),
        ];
        for syntax in [&qwen, &mimo] {
            for (pieces, reasoning, content, calls) in tool_cases {
                for feed in 0..3 {
                    let message = delimited(syntax, OutputStart::Reasoning, &tools, pieces, feed);
                    assert_eq!(
                        message["reasoning_content"], *reasoning,
                        "{pieces:?} feed {feed}"
                    );
                    assert_eq!(
                        message["content"].as_str().unwrap_or_default(),
                        *content,
                        "{pieces:?} feed {feed}"
                    );
                    assert_eq!(
                        message
                            .get("tool_calls")
                            .map_or(0, |c| c.as_array().unwrap().len()),
                        *calls,
                        "{pieces:?} feed {feed}"
                    );
                }
            }
        }

        let mut minimax = crate::model::testing::metadata().frontend;
        minimax.tools = Some(ToolParser {
            kind: ToolParserKind::MiniMaxXml,
            wrapping_newlines: 0,
            tag_separator: String::new(),
        });
        minimax.reasoning = Some(ReasoningParser {
            kind: ReasoningParserKind::MmThink,
            open: "<mm:think>".into(),
            close: "</mm:think>".into(),
            wrapping_newlines: 0,
            terminal_markers: vec![],
        });
        for (pieces, reasoning, content) in [
            (
                &[(true, "</mm:think>"), text("answer</mm:think>")][..],
                None,
                "answer</mm:think>",
            ),
            (&[text("</mm:think>answer")][..], None, "</mm:think>answer"),
            (
                &[
                    (true, "<mm:think>"),
                    text("Quote </mm:think> here."),
                    (true, "</mm:think>"),
                    text("done"),
                ][..],
                Some("Quote </mm:think> here."),
                "done",
            ),
        ] {
            for feed in 0..3 {
                let message = delimited(&minimax, OutputStart::Content, &[], pieces, feed);
                assert_eq!(
                    message.get("reasoning_content").and_then(|r| r.as_str()),
                    reasoning,
                    "{pieces:?} feed {feed}"
                );
                assert_eq!(message["content"], content, "{pieces:?} feed {feed}");
            }
        }
    }

    #[test]
    fn channel_thought_gates_only_its_single_token_close() {
        use crate::output::tagged::OutputStart;
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.tools = Some(ToolParser {
            kind: ToolParserKind::GemmaCall,
            wrapping_newlines: 0,
            tag_separator: String::new(),
        });
        syntax.reasoning = Some(ReasoningParser {
            kind: ReasoningParserKind::ChannelThought,
            open: "<|channel>thought".into(),
            close: "<channel|>".into(),
            wrapping_newlines: 1,
            terminal_markers: vec!["<|tool_call>".into()],
        });
        let control_tokens = [(2, "<channel|>".to_owned())];
        let tools = [
            json!({"function":{"name":"lookup","parameters":{"properties":{"n":{"type":"integer"}}}}}),
        ];
        let close = (true, "<channel|>");
        let text = |text| (false, text);
        let call = "<|tool_call>call:lookup{n:1}<tool_call|>";
        let cases: &[(OutputStart, &[(bool, &str)], Option<&str>, &str, usize)] = &[
            (
                OutputStart::Reasoning,
                &[text("Quote <channel|> here.\n"), close, text("Answer.")],
                Some("Quote <channel|> here."),
                "Answer.",
                0,
            ),
            (
                OutputStart::Content,
                &[
                    text("<|channel>thought\nWrite <channel|> to close.\n"),
                    close,
                    text("Done."),
                ],
                Some("Write <channel|> to close."),
                "Done.",
                0,
            ),
            (
                OutputStart::Content,
                &[text("Answer with <channel|> spelled.")],
                None,
                "Answer with <channel|> spelled.",
                0,
            ),
            (
                OutputStart::Content,
                &[text("Use <|chan to begin.")],
                None,
                "Use <|chan to begin.",
                0,
            ),
            (
                OutputStart::Content,
                &[close, text("answer")],
                None,
                "answer",
                0,
            ),
            (
                OutputStart::Content,
                &[text("<channel|>answer")],
                None,
                "<channel|>answer",
                0,
            ),
            (
                OutputStart::Reasoning,
                &[text("Quote <channel|> first."), text(call)],
                Some("Quote <channel|> first."),
                "",
                1,
            ),
            (
                OutputStart::Content,
                &[
                    text("<|channel>thought\nplan\n"),
                    close,
                    text("Calling. "),
                    text(call),
                ],
                Some("plan"),
                "Calling. ",
                1,
            ),
        ];
        for (start, pieces, reasoning, content, calls) in cases {
            for feed in 0..3 {
                let message = delimited_by(&syntax, &control_tokens, *start, &tools, pieces, feed);
                assert_eq!(
                    message.get("reasoning_content").and_then(|r| r.as_str()),
                    *reasoning,
                    "{pieces:?} feed {feed}"
                );
                assert_eq!(
                    message["content"].as_str().unwrap_or_default(),
                    *content,
                    "{pieces:?} feed {feed}"
                );
                assert_eq!(
                    message
                        .get("tool_calls")
                        .map_or(0, |c| c.as_array().unwrap().len()),
                    *calls,
                    "{pieces:?} feed {feed}"
                );
            }
        }
    }
}
