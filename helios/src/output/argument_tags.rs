//! GLM think envelopes and raw-string/JSON argument tags from the chat template.

use crate::{
    Result,
    output::tagged::{ToolDialect, parameter_value},
};
use serde_json::{Map, Value};

#[cfg(test)]
use crate::output::tagged::{OutputConfig, TaggedOutput};

pub struct ArgumentTags;
impl ToolDialect for ArgumentTags {
    const NAME: &'static str = "GLM";
    const TRIM_NEWLINES: bool = false;
    const OPEN: &'static str = "<tool_call>";
    const CLOSE: &'static str = "</tool_call>";

    fn parse(body: &str, tools: &[Value]) -> Result<Vec<(String, String)>> {
        let body = body.trim();
        let split = body.find("<arg_key>").unwrap_or(body.len());
        let name = body[..split].trim();
        if name.is_empty() {
            return Err("tool function name is empty".into());
        }
        let mut rest = &body[split..];
        let function = tools
            .iter()
            .filter_map(|tool| tool.get("function"))
            .find(|function| function.get("name").and_then(Value::as_str) == Some(name));
        let properties = function
            .and_then(|function| function.get("parameters"))
            .and_then(|parameters| parameters.get("properties"))
            .and_then(Value::as_object);
        let mut arguments = Map::new();
        while !rest.trim().is_empty() {
            let key = rest
                .trim_start()
                .strip_prefix("<arg_key>")
                .ok_or("GLM tool call requires an argument key")?;
            let (key, tail) = key
                .split_once("</arg_key>")
                .ok_or("unterminated GLM argument key")?;
            if key.is_empty() || arguments.contains_key(key) {
                return Err("GLM argument names must be nonempty and unique".into());
            }
            let value = tail
                .trim_start()
                .strip_prefix("<arg_value>")
                .ok_or("GLM argument requires a value tag")?;
            let (value, tail) = value
                .split_once("</arg_value>")
                .ok_or("unterminated GLM argument value")?;
            arguments.insert(
                key.to_owned(),
                parameter_value(value, properties.and_then(|p| p.get(key)))?,
            );
            rest = tail;
        }
        Ok(vec![(
            name.to_owned(),
            serde_json::to_string(&arguments).map_err(|error| {
                crate::FrontendError::Input(crate::Failure::caused_by(error.to_string(), error))
            })?,
        )])
    }
    fn starts_call(text: &str) -> Option<bool> {
        // Calls continue with the function name and an argument key or closing tag;
        // quoted tags followed by prose remain text.
        let text = text.strip_prefix(Self::OPEN).unwrap_or(text).trim_start();
        if text.is_empty() {
            return None;
        }
        let end = text.find(|c: char| c == '<' || c == '>' || c.is_whitespace())?;
        if end == 0 {
            return Some(false);
        }
        let next = text[end..].trim_start();
        if ["<arg_key>", "</tool_call>"]
            .iter()
            .any(|tag| next.starts_with(tag))
        {
            Some(true)
        } else if ["<arg_key>", "</tool_call>"]
            .iter()
            .any(|tag| tag.starts_with(next))
        {
            None
        } else {
            Some(false)
        }
    }
}

#[cfg(test)]
type ArgumentTagOutput = TaggedOutput<ArgumentTags>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        output::{Input, ModelOutput, Termination},
        server::output::Accumulator,
    };
    use serde_json::json;

    fn config() -> OutputConfig {
        OutputConfig {
            structured: false,
            start: crate::output::tagged::OutputStart::Reasoning,
            tools: vec![json!({"type":"function", "function": {
            "name":"run", "parameters":{"type":"object", "properties":{
                "text":{"type":"string"}, "count":{"type":"integer"}, "enabled":{"type":"boolean"},
                "items":{"type":"array"}, "options":{"type":"object"}, "empty":{"type":"null"}
            }}}})],
        }
    }

    #[test]
    fn split_streams_preserve_reasoning_and_decode_typed_tool_arguments() {
        let text = "\nreason α\n</think>\nAnswer\n<tool_call>run<arg_key>text</arg_key><arg_value>\nhello α\n</arg_value><arg_key>count</arg_key><arg_value>2</arg_value><arg_key>enabled</arg_key><arg_value>true</arg_value><arg_key>items</arg_key><arg_value>[1,2]</arg_value><arg_key>options</arg_key><arg_value>{\"x\":1}</arg_value><arg_key>empty</arg_key><arg_value>null</arg_value></tool_call><tool_call>run</tool_call>\nend";
        for boundary in text
            .char_indices()
            .map(|(index, _)| index)
            .chain([text.len()])
        {
            let mut parser = ArgumentTagOutput::new(config());
            let mut output = Accumulator::new("test".into());
            output
                .apply(parser.push(Input::Text(&text[..boundary])).unwrap())
                .unwrap();
            output
                .apply(parser.push(Input::Text(&text[boundary..])).unwrap())
                .unwrap();
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["reasoning_content"], "\nreason α\n");
            assert_eq!(message["content"], "\nAnswer\n\nend");
            let calls = message["tool_calls"].as_array().unwrap();
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0]["function"]["name"], "run");
            let arguments: Value =
                serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(
                arguments,
                json!({"text":"\nhello α\n", "count":2, "enabled":true, "items":[1,2], "options":{"x":1}, "empty":null})
            );
            assert_eq!(calls[1]["function"]["arguments"], "{}");
        }
    }

    #[test]
    fn malformed_calls_fail_without_coercion() {
        for body in [
            "",
            "run<arg_key></arg_key><arg_value>x</arg_value>",
            "run<arg_key>count</arg_key><arg_value>two</arg_value>",
            "run<arg_key>text</arg_key><arg_value>a</arg_value><arg_key>text</arg_key><arg_value>b</arg_value>",
            "run<arg_key>text</arg_key><arg_value>unfinished",
            "run<arg_key>x</arg_key>text",
        ] {
            assert!(
                ArgumentTags::parse(body, &config().tools).is_err(),
                "{body}"
            );
        }
    }

    #[test]
    fn prose_tool_tag_mentions_remain_answer_text() {
        let call = "<tool_call>run<arg_key>text</arg_key><arg_value>x</arg_value></tool_call>";
        for (text, content, calls) in [
            (
                "</think>The `<tool_call>` tag wraps calls. END".to_owned(),
                "The `<tool_call>` tag wraps calls. END",
                0,
            ),
            (
                "</think>Write <tool_call> or `<tool_call>run` literally.".to_owned(),
                "Write <tool_call> or `<tool_call>run` literally.",
                0,
            ),
            (
                format!("</think>Quote `<tool_call>` here.\n{call}"),
                "Quote `<tool_call>` here.\n",
                1,
            ),
        ] {
            for split in text.char_indices().map(|(at, _)| at) {
                let mut parser = ArgumentTagOutput::new(config());
                let mut output = Accumulator::new("test".into());
                for part in [&text[..split], &text[split..]] {
                    output
                        .apply(parser.push(Input::Text(part)).unwrap())
                        .unwrap();
                }
                output
                    .apply(parser.finish(Termination::Eos).unwrap())
                    .unwrap();
                output.finish().unwrap();
                let message = output.message();
                assert_eq!(message["content"], content, "{text:?} at {split}");
                let tools = message["tool_calls"]
                    .as_array()
                    .map_or(&[][..], Vec::as_slice);
                assert_eq!(tools.len(), calls, "{text:?} at {split}");
                if let Some(tool) = tools.first() {
                    assert_eq!(tool["function"]["name"], "run");
                    assert_eq!(tool["function"]["arguments"], "{\"text\":\"x\"}");
                }
            }
        }
    }

    #[test]
    fn truncation_and_structured_output_keep_protocol_semantics() {
        let mut parser = ArgumentTagOutput::new(config());
        parser
            .push(Input::Text("</think><tool_call>run<arg_key>text</arg_key>"))
            .unwrap();
        assert!(parser.finish(Termination::OutputLimit).unwrap().is_empty());
        let mut parser = ArgumentTagOutput::new(OutputConfig {
            structured: true,
            start: crate::output::tagged::OutputStart::Content,
            ..config()
        });
        assert_eq!(
            parser
                .push(Input::Text("<think>{\"x\":1}</think>"))
                .unwrap(),
            vec![crate::output::Event::Content(
                "<think>{\"x\":1}</think>".into()
            )]
        );
    }
}
